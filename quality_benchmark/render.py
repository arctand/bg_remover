from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import PreviewSpec


def _fit(image: Image.Image, size: tuple[int, int], background=(32, 34, 38, 255)) -> Image.Image:
    copy = image.convert("RGBA")
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", size, background)
    canvas.alpha_composite(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return canvas


def _contrast_background(size: tuple[int, int]) -> Image.Image:
    width, height = size
    cell = max(16, min(width, height) // 10)
    array = np.zeros((height, width, 4), dtype=np.uint8)
    colors = np.asarray([[0, 220, 255, 255], [255, 0, 170, 255]], dtype=np.uint8)
    yy, xx = np.indices((height, width))
    array[:] = colors[((xx // cell + yy // cell) % 2)]
    return Image.fromarray(array)


def composite(foreground: Image.Image, alpha: Image.Image, background: Image.Image) -> Image.Image:
    rgba = foreground.convert("RGBA")
    rgba.putalpha(alpha.convert("L"))
    return Image.alpha_composite(background.convert("RGBA"), rgba)


def make_variant_preview(
    original: Image.Image,
    alpha: Image.Image,
    foreground: Image.Image,
    title: str,
    subtitle: str,
    target: Path,
    config: PreviewSpec,
) -> Path:
    panel_size = (config.panel_width, config.panel_height)
    backgrounds = [
        Image.new("RGBA", original.size, "white"),
        Image.new("RGBA", original.size, (128, 128, 128, 255)),
        Image.new("RGBA", original.size, "black"),
        _contrast_background(original.size),
    ]
    panels = [original.convert("RGBA"), alpha.convert("RGBA")]
    panels.extend(composite(foreground, alpha, background) for background in backgrounds)
    header = 58
    canvas = Image.new("RGB", (panel_size[0] * len(panels), panel_size[1] + header), "#17191c")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 7), title, fill="white", font=ImageFont.load_default())
    draw.text((10, 29), subtitle, fill="#b7c0cc", font=ImageFont.load_default())
    labels = ["original", "alpha", "white", "gray", "black", "contrast"]
    for index, (label, panel) in enumerate(zip(labels, panels)):
        fitted = _fit(panel, panel_size).convert("RGB")
        x = index * panel_size[0]
        canvas.paste(fitted, (x, header))
        draw.text((x + 6, header + 6), label, fill="white", stroke_width=2, stroke_fill="black")
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=90)
    return target


def make_contact_sheet(previews: list[Path], target: Path, columns: int = 2, thumb_width: int = 900) -> Path:
    if not previews:
        raise ValueError("No previews for contact sheet")
    opened = [Image.open(path).convert("RGB") for path in previews]
    ratio = opened[0].height / opened[0].width
    thumb_height = max(1, round(thumb_width * ratio))
    rows = math.ceil(len(opened) / columns)
    canvas = Image.new("RGB", (columns * thumb_width, rows * thumb_height), "#101114")
    for index, image in enumerate(opened):
        image.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        canvas.paste(image, ((index % columns) * thumb_width, (index // columns) * thumb_height))
    for image in opened:
        image.close()
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=88)
    return target


def make_side_by_side(case_name: str, previews: list[tuple[str, Path]], target: Path) -> Path:
    images: list[tuple[str, Image.Image]] = []
    for name, path in previews:
        images.append((name, Image.open(path).convert("RGB")))
    width = 760
    header = 34
    thumb_height = round(width * images[0][1].height / images[0][1].width)
    canvas = Image.new("RGB", (width, len(images) * (thumb_height + header)), "#101114")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for name, image in images:
        draw.text((8, y + 8), f"{case_name} | {name}", fill="white")
        image.thumbnail((width, thumb_height), Image.Resampling.LANCZOS)
        canvas.paste(image, (0, y + header))
        y += thumb_height + header
        image.close()
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=88)
    return target
