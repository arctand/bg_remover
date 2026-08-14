from __future__ import annotations

import gc
import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .config import ModelSpec


@dataclass
class TimedResult:
    value: object
    time_s: float
    peak_vram_gb: float


def _sync_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_gb() -> float:
    import torch

    return float(torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0


def release_cuda(*objects: object) -> None:
    for obj in objects:
        del obj
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def expand_box(box: list[int], size: tuple[int, int], margin_ratio: float) -> list[int]:
    width, height = size
    x1, y1, x2, y2 = box
    margin_x = round((x2 - x1) * margin_ratio)
    margin_y = round((y2 - y1) * margin_ratio)
    return [max(0, x1 - margin_x), max(0, y1 - margin_y), min(width, x2 + margin_x), min(height, y2 + margin_y)]


def paste_crop_mask(canvas_size: tuple[int, int], box: list[int], crop_mask: Image.Image) -> Image.Image:
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop box: {box}")
    canvas = Image.new("L", canvas_size, 0)
    resized = crop_mask.convert("L").resize((x2 - x1, y2 - y1), Image.Resampling.BILINEAR)
    canvas.paste(resized, (x1, y1))
    return canvas


class ResearchBiRefNet:
    """Explicit-revision BiRefNet runner isolated from the production backend."""

    def __init__(self, spec: ModelSpec, device: str = "cuda", precision: str = "fp16"):
        self.spec = spec
        self.device_name = device
        self.precision = precision
        self.model = None
        self.load_time_s = 0.0

    def load(self) -> "ResearchBiRefNet":
        import torch
        from transformers import AutoModelForImageSegmentation

        if self.device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by the benchmark configuration")
        started = time.perf_counter()
        self.device = torch.device(self.device_name)
        self.dtype = torch.float16 if self.device.type == "cuda" and self.precision == "fp16" else torch.float32
        self.model = AutoModelForImageSegmentation.from_pretrained(
            self.spec.model_id,
            revision=self.spec.revision,
            trust_remote_code=True,
        ).eval().to(self.device)
        if self.dtype == torch.float16:
            self.model.half()
        _sync_cuda()
        self.load_time_s = time.perf_counter() - started
        return self

    def _predict_crop(self, image: Image.Image) -> Image.Image:
        import torch
        from torchvision.transforms import functional as functional

        tensor = functional.pil_to_tensor(
            image.resize((self.spec.image_size, self.spec.image_size), Image.Resampling.LANCZOS)
        ).float() / 255.0
        tensor = functional.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0)
        tensor = tensor.to(self.device, dtype=self.dtype)
        amp = torch.autocast("cuda", dtype=torch.float16) if self.dtype == torch.float16 else nullcontext()
        with torch.inference_mode(), amp:
            output = self.model(tensor)[-1]
            prediction = output.sigmoid()[0].squeeze().float().cpu().numpy()
        del tensor, output
        return Image.fromarray(np.clip(prediction * 255.0, 0, 255).astype(np.uint8), "L")

    def predict(self, image: Image.Image) -> TimedResult:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        _reset_peak()
        _sync_cuda()
        started = time.perf_counter()
        mask = self._predict_crop(image).resize(image.size, Image.Resampling.LANCZOS)
        _sync_cuda()
        return TimedResult(mask, time.perf_counter() - started, _peak_gb())

    def predict_guided(self, image: Image.Image, boxes: list[list[int]], margin_ratio: float) -> TimedResult:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        if not boxes:
            raise RuntimeError("No person boxes for guided prediction")
        _reset_peak()
        _sync_cuda()
        started = time.perf_counter()
        combined = np.zeros((image.height, image.width), dtype=np.uint8)
        for raw_box in boxes:
            box = expand_box(raw_box, image.size, margin_ratio)
            crop = image.crop(tuple(box))
            crop_mask = self._predict_crop(crop)
            placed = np.asarray(paste_crop_mask(image.size, box, crop_mask), dtype=np.uint8)
            combined = np.maximum(combined, placed)
        _sync_cuda()
        return TimedResult(Image.fromarray(combined, "L"), time.perf_counter() - started, _peak_gb())

    def unload(self) -> None:
        model = self.model
        self.model = None
        release_cuda(model)


class SSDLiteDetector:
    def __init__(self, confidence: float, device: str = "cuda"):
        self.confidence = confidence
        self.device_name = device
        self.model = None
        self.load_time_s = 0.0

    def load(self) -> "SSDLiteDetector":
        from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large

        started = time.perf_counter()
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.model = ssdlite320_mobilenet_v3_large(weights=weights).eval().to(self.device_name)
        _sync_cuda()
        self.load_time_s = time.perf_counter() - started
        return self

    def detect(self, image: Image.Image) -> TimedResult:
        import torch

        _reset_peak()
        tensor = self.transforms(image).to(self.device_name)
        _sync_cuda()
        started = time.perf_counter()
        with torch.inference_mode():
            prediction = self.model([tensor])[0]
        _sync_cuda()
        keep = (prediction["labels"] == 1) & (prediction["scores"] >= self.confidence)
        boxes = prediction["boxes"][keep].round().int().cpu().tolist()
        scores = prediction["scores"][keep].float().cpu().tolist()
        return TimedResult({"boxes": boxes, "scores": scores}, time.perf_counter() - started, _peak_gb())

    def unload(self) -> None:
        model = self.model
        self.model = None
        release_cuda(model)


class SAM21Verifier:
    def __init__(self, model_id: str, revision: str):
        self.model_id = model_id
        self.revision = revision
        self.predictor = None
        self.load_time_s = 0.0

    def load(self) -> "SAM21Verifier":
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        started = time.perf_counter()
        self.predictor = SAM2ImagePredictor.from_pretrained(self.model_id, revision=self.revision)
        _sync_cuda()
        self.load_time_s = time.perf_counter() - started
        return self

    def verify(self, image: Image.Image, boxes: list[list[int]]) -> TimedResult:
        if not boxes:
            raise RuntimeError("No SSDLite person boxes available as SAM 2.1 prompts")
        _reset_peak()
        _sync_cuda()
        started = time.perf_counter()
        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(
            box=np.asarray(boxes, dtype=np.float32),
            multimask_output=False,
        )
        _sync_cuda()
        masks = np.asarray(masks)
        if masks.ndim == 2:
            union = masks
        else:
            union = np.any(masks.reshape((-1, masks.shape[-2], masks.shape[-1])), axis=0)
        result = {
            "mask": Image.fromarray(union.astype(np.uint8) * 255, "L"),
            "scores": np.asarray(scores).reshape(-1).astype(float).tolist(),
        }
        return TimedResult(result, time.perf_counter() - started, _peak_gb())

    def unload(self) -> None:
        predictor = self.predictor
        self.predictor = None
        release_cuda(predictor)
