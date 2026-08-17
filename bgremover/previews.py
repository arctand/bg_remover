from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .config import PreviewConfig
from .images import atomic_save_png

try:
    from .artifacts import ArtifactResult
except ImportError:  # pragma: no cover - keeps the preview module independently usable
    ArtifactResult = object  # type: ignore[misc,assignment]


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.copy(); copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", size, (238, 238, 238, 255))
    canvas.alpha_composite(copy.convert("RGBA"), ((size[0]-copy.width)//2, (size[1]-copy.height)//2))
    return canvas


def _composite(rgba: Image.Image, color) -> Image.Image:
    bg = Image.new("RGBA", rgba.size, color)
    return Image.alpha_composite(bg, rgba).convert("RGB")


def _artifact_overlay(original: Image.Image, artifact: ArtifactResult | None) -> Image.Image:
    base = original.convert("RGB")
    heatmap = getattr(artifact, "heatmap", None) if artifact is not None else None
    if heatmap is None:
        return base
    heat = Image.fromarray((heatmap * 255).clip(0, 255).astype("uint8"))
    heat = heat.resize(base.size, Image.Resampling.BILINEAR)
    color = Image.new("RGB", base.size, "#ff2b59")
    return Image.composite(color, base, heat.point(lambda value: round(value * 0.72)))


def make_preview(original: Image.Image, rgba: Image.Image, path: Path, cfg: PreviewConfig,
                 artifact: ArtifactResult | None = None):
    panel_count = 6 if artifact is not None else 5
    panel_w, panel_h = cfg.width // panel_count, cfg.panel_height
    contrast = Image.new("RGB", rgba.size, "#ff00a8")
    block = max(16, min(rgba.size) // 12)
    draw_contrast = ImageDraw.Draw(contrast)
    for y in range(0, rgba.height, block):
        for x in range(0, rgba.width, block):
            if (x // block + y // block) % 2:
                draw_contrast.rectangle((x, y, x + block, y + block), fill="#00d6ff")
    panels = [
        original.convert("RGB"),
        _composite(rgba, "white"),
        _composite(rgba, "#888888"),
        _composite(rgba, "black"),
        Image.alpha_composite(contrast.convert("RGBA"), rgba).convert("RGB"),
    ]
    labels = ["Original", "White", "Gray", "Black", "Contrast"]
    if artifact is not None:
        panels.append(_artifact_overlay(original, artifact))
        labels.append(f"Artifact: {artifact.severity}")
    canvas = Image.new("RGB", (panel_w * panel_count, panel_h + 32), "#202124")
    draw = ImageDraw.Draw(canvas)
    for i, (panel, label) in enumerate(zip(panels, labels)):
        fitted = _fit(panel, (panel_w, panel_h)).convert("RGB")
        canvas.paste(fitted, (i * panel_w, 32)); draw.text((i * panel_w + 8, 8), label, fill="white")
    atomic_save_png(canvas.convert("RGBA"), path)


def make_contact_sheet(items: list[tuple[Path, str, str]], target: Path, cfg: PreviewConfig):
    if not items: return
    cols, cell_w, cell_h = cfg.contact_columns, cfg.contact_thumb_width, int(cfg.contact_thumb_width * .8)
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "#17191c")
    draw = ImageDraw.Draw(sheet)
    colors = {"ready": "#47c97a", "review": "#ffb84d", "failed": "#ff6464"}
    for index, (preview, name, status) in enumerate(items):
        x, y = (index % cols) * cell_w, (index // cols) * cell_h
        try:
            with Image.open(preview) as im: thumb = _fit(im.convert("RGB"), (cell_w - 12, cell_h - 48)).convert("RGB")
            sheet.paste(thumb, (x + 6, y + 6))
        except Exception: pass
        draw.text((x + 8, y + cell_h - 38), name[:38], fill="white")
        draw.text((x + 8, y + cell_h - 20), status.upper(), fill=colors.get(status, "white"))
    atomic_save_png(sheet.convert("RGBA"), target)
