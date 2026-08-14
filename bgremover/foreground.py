from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .config import ForegroundConfig


@dataclass(frozen=True)
class ForegroundResult:
    rgb: Image.Image
    alpha: Image.Image


class PyMattingForegroundRefiner:
    """Recover foreground RGB while preserving the model alpha byte-for-byte."""

    def __init__(self, config: ForegroundConfig):
        self.config = config

    def refine(self, image: Image.Image, alpha: Image.Image) -> ForegroundResult:
        source = image.convert("RGB")
        matte = alpha.convert("L")
        if source.size != matte.size:
            raise ValueError("Foreground RGB and alpha must have identical dimensions")
        if not self.config.enabled:
            return ForegroundResult(source.copy(), matte.copy())
        if self.config.method != "pymatting_ml":
            raise ValueError(f"Unsupported foreground refinement method: {self.config.method}")

        from pymatting import estimate_foreground_ml

        alpha_bytes = np.asarray(matte, dtype=np.uint8).copy()
        rgb = np.asarray(source, dtype=np.float64) / 255.0
        alpha_float = alpha_bytes.astype(np.float64) / 255.0
        foreground = estimate_foreground_ml(rgb, alpha_float)
        output = Image.fromarray(np.clip(np.rint(foreground * 255.0), 0, 255).astype(np.uint8))
        # Return the original 8-bit alpha values. Refinement is RGB-only.
        unchanged_alpha = Image.fromarray(alpha_bytes)
        return ForegroundResult(output, unchanged_alpha)

    def warm_up(self) -> None:
        if not self.config.enabled or self.config.method != "pymatting_ml":
            return
        from pymatting import estimate_foreground_ml

        image = np.zeros((32, 32, 3), dtype=np.float64)
        alpha = np.zeros((32, 32), dtype=np.float64)
        alpha[8:24, 8:24] = 1.0
        estimate_foreground_ml(image, alpha, n_small_iterations=1, n_big_iterations=1)
