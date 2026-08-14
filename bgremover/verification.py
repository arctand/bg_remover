from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .config import QCConfig


@dataclass
class HumanVerification:
    person_count: int = 0
    missing_count: int = 0
    uncertain_count: int = 0
    coverages: list[float] = field(default_factory=list)
    center_coverages: list[float] = field(default_factory=list)
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def missing_body_part(self): return self.missing_count > 0
    @property
    def multiple_people_uncertain(self): return self.person_count > 1 and self.uncertain_count > 0


class TorchvisionPersonVerifier:
    """Independent, inexpensive COCO person detector used for QC, not cutout creation."""
    def __init__(self, qc: QCConfig, device: str = "cuda"):
        self.qc, self.device_name, self.model = qc, device, None

    def load(self):
        import torch
        from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.model = ssdlite320_mobilenet_v3_large(weights=weights).eval().to(self.device_name)
        return self

    def verify(self, image: Image.Image, alpha: np.ndarray) -> HumanVerification:
        import torch
        if self.model is None: self.load()
        tensor = self.transforms(image).to(self.device_name)
        with torch.inference_mode(): prediction = self.model([tensor])[0]
        keep = (prediction["labels"] == 1) & (prediction["scores"] >= self.qc.person_confidence)
        boxes = prediction["boxes"][keep].round().int().cpu().tolist()
        h, w = alpha.shape; fg = alpha >= self.qc.foreground_threshold
        result = HumanVerification(person_count=len(boxes))
        for raw in boxes:
            x1, y1, x2, y2 = max(0,raw[0]), max(0,raw[1]), min(w,raw[2]), min(h,raw[3])
            if x2 <= x1 or y2 <= y1: continue
            region = fg[y1:y2, x1:x2]
            # Detector boxes contain background; use a narrower torso/body prior too.
            mx, my = int((x2-x1)*.22), int((y2-y1)*.10)
            center = fg[y1+my:y2-my, x1+mx:x2-mx]
            coverage, center_coverage = float(region.mean()), float(center.mean()) if center.size else 0.0
            result.boxes.append((x1,y1,x2,y2)); result.coverages.append(coverage); result.center_coverages.append(center_coverage)
            if coverage < self.qc.min_person_mask_coverage or center_coverage < self.qc.min_person_center_coverage:
                result.missing_count += 1
            elif coverage < self.qc.min_person_mask_coverage * 1.35:
                result.uncertain_count += 1
        return result

    def clear_cache(self):
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception: pass
