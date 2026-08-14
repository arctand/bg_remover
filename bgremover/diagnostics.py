from __future__ import annotations

import platform
import sys

from .config import AppConfig
from .inference import detect_device


def diagnostics_text(config: AppConfig) -> str:
    info = detect_device(config.model.precision)
    values = {
        "Windows": platform.platform(), "Python": sys.version.split()[0], "PyTorch": info.torch_version,
        "CUDA runtime": info.cuda_runtime, "CUDA доступна": "да" if info.available else "нет",
        "GPU": info.name, "VRAM": f"{info.vram_gb:.1f} GB", "Модель": config.model.primary,
        "Precision": info.precision, "Ошибка": info.error or "нет",
    }
    return "\n".join(f"{key}: {value}" for key, value in values.items())
