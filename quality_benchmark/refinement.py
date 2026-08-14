from __future__ import annotations

import time

import cv2
import numpy as np
from PIL import Image

from bgremover.config import EdgeConfig
from bgremover.edge import decontaminate_rgb

from .models import TimedResult, _peak_gb, _reset_peak, _sync_cuda


def _official_mean_blur(tensor, kernel_size: int):
    import torch

    if kernel_size % 2 == 0:
        pad_left = pad_top = kernel_size // 2 - 1
        pad_right = pad_bottom = kernel_size // 2
    else:
        pad_left = pad_right = pad_top = pad_bottom = kernel_size // 2
    padded = torch.nn.functional.pad(
        tensor,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="replicate",
    )
    return torch.nn.functional.avg_pool2d(
        padded,
        kernel_size=(kernel_size, kernel_size),
        stride=1,
        count_include_pad=False,
    )


def _official_fusion(image, foreground, background, alpha, radius: int):
    import torch

    image = image.float()
    foreground = foreground.float()
    background = background.float()
    alpha = alpha.float()
    blurred_alpha = _official_mean_blur(alpha, radius)
    blurred_foreground = _official_mean_blur(foreground * alpha, radius) / (blurred_alpha + 1e-5)
    blurred_background = _official_mean_blur(background * (1 - alpha), radius) / (1 - blurred_alpha + 1e-5)
    result = blurred_foreground + alpha * (
        image - alpha * blurred_foreground - (1 - alpha) * blurred_background
    )
    return torch.clamp(result, 0, 1), blurred_background


def refine_birefnet_official(image: Image.Image, alpha: Image.Image, radius: int = 90) -> TimedResult:
    """GPU port from ZhengPeng7/BiRefNet image_proc.py (MIT), commit observed 2026-08-14."""
    import torch
    from torchvision.transforms import functional

    _reset_peak()
    image_tensor = functional.to_tensor(image.convert("RGB")).float().cuda().unsqueeze(0)
    alpha_tensor = functional.to_tensor(alpha.convert("L").resize(image.size)).float().cuda().unsqueeze(0)
    _sync_cuda()
    started = time.perf_counter()
    foreground, blurred_background = _official_fusion(
        image_tensor, image_tensor, image_tensor, alpha_tensor, radius
    )
    foreground, _ = _official_fusion(
        image_tensor, foreground, blurred_background, alpha_tensor, 6
    )
    _sync_cuda()
    array = (
        foreground.squeeze(0).mul(255.0).to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()
    )
    elapsed = time.perf_counter() - started
    return TimedResult(Image.fromarray(array, "RGB"), elapsed, _peak_gb())


def refine_current_edge(image: Image.Image, alpha: Image.Image, config: EdgeConfig) -> TimedResult:
    started = time.perf_counter()
    foreground, metrics = decontaminate_rgb(image, alpha, config)
    return TimedResult(
        {"foreground": foreground, "edge_metrics": metrics.__dict__},
        time.perf_counter() - started,
        0.0,
    )


def refine_pymatting(image: Image.Image, alpha: Image.Image) -> TimedResult:
    from pymatting import estimate_foreground_ml

    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    matte = np.asarray(alpha.convert("L").resize(image.size), dtype=np.float64) / 255.0
    started = time.perf_counter()
    foreground = estimate_foreground_ml(rgb, matte)
    elapsed = time.perf_counter() - started
    output = Image.fromarray(np.clip(np.rint(foreground * 255.0), 0, 255).astype(np.uint8), "RGB")
    return TimedResult(output, elapsed, 0.0)


def warm_up_pymatting() -> None:
    from pymatting import estimate_foreground_ml

    image = np.zeros((32, 32, 3), dtype=np.float64)
    alpha = np.zeros((32, 32), dtype=np.float64)
    alpha[8:24, 8:24] = 1.0
    estimate_foreground_ml(image, alpha, n_small_iterations=1, n_big_iterations=1)


def raw_foreground(image: Image.Image) -> TimedResult:
    return TimedResult(image.convert("RGB").copy(), 0.0, 0.0)
