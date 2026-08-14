from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from .config import QCConfig, VerificationConfig


@dataclass
class HumanVerification:
    person_count: int = 0
    missing_count: int = 0
    uncertain_count: int = 0
    coverages: list[float] = field(default_factory=list)
    center_coverages: list[float] = field(default_factory=list)
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    @property
    def person_detector_zero(self) -> bool:
        return self.person_count == 0


@dataclass
class SAMVerification:
    ran: bool = False
    prompted_boxes: int = 0
    checked_people: int = 0
    min_recall: float | None = None
    min_iou: float | None = None
    missing_count: int = 0
    reasons: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)


class TorchvisionPersonVerifier:
    """Fast COCO person detector used for advisory QC and SAM prompts only."""

    def __init__(self, qc: QCConfig, device: str = "cuda"):
        self.qc, self.device_name, self.model = qc, device, None

    def load(self):
        from torchvision.models.detection import (
            SSDLite320_MobileNet_V3_Large_Weights,
            ssdlite320_mobilenet_v3_large,
        )

        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.model = ssdlite320_mobilenet_v3_large(weights=weights).eval().to(self.device_name)
        return self

    def verify(self, image: Image.Image, alpha: np.ndarray) -> HumanVerification:
        import torch

        if self.model is None:
            self.load()
        tensor = self.transforms(image).to(self.device_name)
        with torch.inference_mode():
            prediction = self.model([tensor])[0]
        keep = (prediction["labels"] == 1) & (prediction["scores"] >= self.qc.person_confidence)
        boxes = prediction["boxes"][keep].round().int().cpu().tolist()
        scores = prediction["scores"][keep].float().cpu().tolist()
        h, w = alpha.shape
        foreground = alpha >= self.qc.foreground_threshold
        result = HumanVerification()
        for raw, score in zip(boxes, scores):
            x1, y1, x2, y2 = max(0, raw[0]), max(0, raw[1]), min(w, raw[2]), min(h, raw[3])
            if x2 <= x1 or y2 <= y1:
                continue
            region = foreground[y1:y2, x1:x2]
            margin_x, margin_y = int((x2 - x1) * 0.22), int((y2 - y1) * 0.10)
            center = foreground[y1 + margin_y:y2 - margin_y, x1 + margin_x:x2 - margin_x]
            coverage = float(region.mean())
            center_coverage = float(center.mean()) if center.size else 0.0
            result.boxes.append((x1, y1, x2, y2))
            result.scores.append(float(score))
            result.coverages.append(coverage)
            result.center_coverages.append(center_coverage)
            if coverage < self.qc.min_person_mask_coverage or center_coverage < self.qc.min_person_center_coverage:
                result.missing_count += 1
            elif coverage < self.qc.min_person_mask_coverage * 1.35:
                result.uncertain_count += 1
        result.person_count = len(result.boxes)
        return result


def should_run_sam(
    qc_reasons: list[str], human: HumanVerification, config: VerificationConfig
) -> bool:
    """Escalate only strong mask/semantic signals; frame contact alone is excluded."""

    if not config.sam_enabled or human.person_detector_zero:
        return False
    if any(reason in {"mask_issue", "low_confidence"} for reason in qc_reasons):
        return True
    if human.missing_count > 0:
        return True
    if any(value < config.sam_trigger_box_coverage for value in human.coverages):
        return True
    if any(value < config.sam_trigger_center_coverage for value in human.center_coverages):
        return True
    if human.person_count > 1 and any(
        value < config.sam_trigger_multiple_coverage for value in human.coverages
    ):
        return True
    return False


class SAM21PersonVerifier:
    """Lazy prompted SAM 2.1 verifier. It never supplies the production alpha."""

    def __init__(self, config: VerificationConfig):
        self.config = config
        self.predictor = None

    @property
    def model(self):
        # BatchProcessor uses this property to distinguish lazy from preloaded models.
        return self.predictor

    def load(self):
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.predictor = SAM2ImagePredictor.from_pretrained(
            self.config.sam_model,
            revision=self.config.sam_revision,
        )
        return self

    def verify(
        self, image: Image.Image, alpha: np.ndarray, human: HumanVerification
    ) -> SAMVerification:
        prompts = [
            (box, score)
            for box, score in zip(human.boxes, human.scores)
            if score >= self.config.sam_prompt_confidence
        ]
        result = SAMVerification(ran=bool(prompts), prompted_boxes=len(prompts))
        if not prompts:
            return result
        if self.predictor is None:
            self.load()

        boxes = np.asarray([box for box, _ in prompts], dtype=np.float32)
        self.predictor.set_image(image.convert("RGB"))
        masks, scores, _ = self.predictor.predict(box=boxes, multimask_output=False)
        masks = np.asarray(masks, dtype=bool)
        height, width = alpha.shape
        masks = masks.reshape((-1, height, width))
        scores = np.asarray(scores, dtype=float).reshape(-1)
        candidate = alpha >= self.config.sam_alpha_threshold
        recalls: list[float] = []
        ious: list[float] = []

        for index, ((box, _), sam_mask) in enumerate(zip(prompts, masks)):
            sam_score = float(scores[index]) if index < len(scores) else 0.0
            if sam_score < self.config.sam_mask_confidence:
                continue
            x1, y1, x2, y2 = box
            local_sam = sam_mask[y1:y2, x1:x2].astype(np.uint8)
            local_candidate = candidate[y1:y2, x1:x2]
            if not local_sam.size or not np.any(local_sam):
                continue

            radius = max(1, round(min(width, height) * self.config.sam_boundary_tolerance_ratio))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
            core = cv2.erode(local_sam, kernel, iterations=1).astype(bool)
            if not np.any(core):
                core = local_sam.astype(bool)
            intersection = np.logical_and(core, local_candidate).sum()
            union = np.logical_or(core, local_candidate).sum()
            recall = float(intersection / max(1, core.sum()))
            iou = float(intersection / max(1, union))
            missing_box_ratio = float(np.logical_and(core, ~local_candidate).sum() / max(1, core.size))
            recalls.append(recall)
            ious.append(iou)
            result.checked_people += 1
            if (
                recall < self.config.sam_min_person_recall
                and missing_box_ratio >= self.config.sam_min_missing_box_ratio
            ):
                result.missing_count += 1
                result.details.append(
                    f"person {index + 1}: SAM recall={recall:.3f}, missing box ratio={missing_box_ratio:.3f}"
                )

        result.min_recall = min(recalls) if recalls else None
        result.min_iou = min(ious) if ious else None
        if result.missing_count:
            result.reasons.extend(["missing_body_part", "human_mask_disagreement"])
            if human.person_count > 1:
                result.reasons.append("multiple_people_uncertain")
        return result

    def clear_cache(self):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
