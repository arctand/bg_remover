from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .config import EdgeConfig


@dataclass
class EdgeMetrics:
    correction_mean: float = 0.0
    correction_p95: float = 0.0
    clipped_ratio: float = 0.0
    edge_pixels: int = 0


def decontaminate_rgb(rgb: Image.Image, alpha: Image.Image, cfg: EdgeConfig) -> tuple[Image.Image, EdgeMetrics]:
    """Estimate foreground RGB in semi-transparent pixels using C=aF+(1-a)B.

    B is estimated by propagating known background colors underneath foreground via
    Telea inpainting. Alpha is never eroded or binarized.
    """
    if not cfg.enabled:
        return rgb.copy(), EdgeMetrics()
    image = np.asarray(rgb.convert("RGB"), dtype=np.uint8)
    matte = np.asarray(alpha.convert("L"), dtype=np.uint8)
    edge = (matte >= cfg.alpha_low) & (matte <= cfg.alpha_high)
    if not np.any(edge):
        return rgb.copy(), EdgeMetrics()
    unknown_foreground = (matte > cfg.alpha_low).astype(np.uint8) * 255
    background = cv2.inpaint(image, unknown_foreground, cfg.background_inpaint_radius, cv2.INPAINT_TELEA).astype(np.float32)
    source = image.astype(np.float32) / 255.0
    bg = background / 255.0
    a = matte.astype(np.float32) / 255.0
    safe_a = np.maximum(a, cfg.low_alpha_protection)[..., None]
    estimated = (source - (1.0 - a[..., None]) * bg) / safe_a
    clipped = np.any((estimated < 0.0) | (estimated > 1.0), axis=2) & edge
    estimated = np.clip(estimated, 0.0, 1.0)
    # Correction fades to zero at fully opaque pixels and is protected at tiny alpha.
    strength = cfg.correction_strength * (1.0 - a) * np.clip(a / cfg.low_alpha_protection, 0.0, 1.0)
    corrected = source * (1.0 - strength[..., None]) + estimated * strength[..., None]
    corrected[~edge] = source[~edge]
    magnitude = np.linalg.norm(corrected - source, axis=2) / np.sqrt(3.0)
    values = magnitude[edge]
    metrics = EdgeMetrics(float(values.mean()), float(np.percentile(values, 95)),
                          float(clipped.sum() / max(1, edge.sum())), int(edge.sum()))
    return Image.fromarray(np.rint(corrected * 255).astype(np.uint8)), metrics
