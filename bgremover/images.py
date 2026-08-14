from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageOps


def discover_images(root: Path, extensions: tuple[str, ...]):
    for path in sorted(root.rglob("*"), key=lambda p: str(p).casefold()):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        source.load()
        return ImageOps.exif_transpose(source).convert("RGB")


def rgba_from_mask(rgb: Image.Image, alpha: Image.Image) -> Image.Image:
    if alpha.size != rgb.size:
        alpha = alpha.resize(rgb.size, Image.Resampling.LANCZOS)
    out = rgb.copy().convert("RGBA")
    out.putalpha(alpha.convert("L"))
    return out


def atomic_save_png(image: Image.Image, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.stem}.{uuid4().hex}.tmp.png")
    try:
        image.save(temp, format="PNG", compress_level=6)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def mask_array(mask: Image.Image) -> np.ndarray:
    return np.asarray(mask.convert("L"), dtype=np.uint8)
