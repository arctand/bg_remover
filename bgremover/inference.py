from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .config import ModelConfig


@dataclass
class DeviceInfo:
    available: bool
    device: str
    name: str
    vram_gb: float
    torch_version: str
    cuda_runtime: str
    precision: str
    error: str = ""


def detect_device(precision: str = "fp16") -> DeviceInfo:
    try:
        import torch
        available = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if available else "CPU"
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3 if available else 0.0
        return DeviceInfo(available, "cuda" if available else "cpu", name, vram,
                          torch.__version__, str(torch.version.cuda or "нет"), precision if available else "fp32")
    except Exception as exc:
        return DeviceInfo(False, "cpu", "CPU", 0.0, "не установлен", "нет", "fp32", str(exc))


class BiRefNetBackend:
    def __init__(self, config: ModelConfig, model_id: str | None = None, allow_cpu: bool = False):
        self.config = config
        self.model_id = model_id or config.primary
        self.revision = config.secondary_revision if self.model_id == config.secondary else config.primary_revision
        self.allow_cpu = allow_cpu
        self.model = None
        self.info = detect_device(config.precision)

    def load(self):
        import torch
        from transformers import AutoModelForImageSegmentation
        if not self.info.available and not self.allow_cpu:
            raise RuntimeError("CUDA недоступна. CPU fallback требует явного подтверждения.")
        self.device = torch.device("cuda" if self.info.available else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" and self.config.precision == "fp16" else torch.float32
        self.model = AutoModelForImageSegmentation.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=True,
        ).eval().to(self.device)
        if self.dtype == torch.float16:
            self.model.half()
        return self

    def predict(self, image: Image.Image) -> Image.Image:
        import torch
        from torchvision.transforms import functional as F
        if self.model is None:
            self.load()
        tensor = F.pil_to_tensor(image.resize((self.config.image_size,) * 2, Image.Resampling.LANCZOS)).float() / 255
        tensor = F.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0)
        tensor = tensor.to(self.device, dtype=self.dtype)
        amp = torch.autocast("cuda", dtype=torch.float16) if self.dtype == torch.float16 else nullcontext()
        with torch.inference_mode(), amp:
            output = self.model(tensor)[-1]
            prediction = output.sigmoid()[0].squeeze().float().cpu().numpy()
        del tensor, output
        mask = Image.fromarray(np.clip(prediction * 255, 0, 255).astype(np.uint8), "L")
        return mask.resize(image.size, Image.Resampling.LANCZOS)

    def clear_cache(self):
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception:
            pass

    def smoke_test(self) -> dict:
        image = Image.new("RGB", (256, 256), "white")
        mask = self.predict(image)
        return {"passed": mask.mode == "L" and mask.size == image.size,
                "size": mask.size, "device": self.info.device, "gpu": self.info.name}
