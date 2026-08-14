from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    primary: str = "ZhengPeng7/BiRefNet_HR-matting"
    secondary: str = "ZhengPeng7/BiRefNet_dynamic-matting"
    primary_revision: str = "5d6b6f8adcb5b417c871b1d84ceaae9871355b7f"
    secondary_revision: str = "074df545be87034e74a96bf71566ecbbc4c15f0a"
    image_size: int = 2048
    precision: str = "fp16"
    secondary_enabled: bool = False
    secondary_review_only: bool = True
    disagreement_iou: float = 0.86
    disagreement_alpha: float = 0.12


@dataclass
class QCConfig:
    foreground_threshold: int = 24
    solid_threshold: int = 224
    min_foreground_ratio: float = 0.015
    max_foreground_ratio: float = 0.94
    max_components: int = 5
    min_component_ratio: float = 0.002
    max_hole_ratio: float = 0.08
    max_translucent_ratio: float = 0.42
    edge_band_ratio: float = 0.018
    edge_min_span_ratio: float = 0.08
    edge_min_area_ratio: float = 0.0015
    halo_correction_p95: float = 0.32
    halo_clipped_ratio: float = 0.10
    min_person_mask_coverage: float = 0.18
    min_person_center_coverage: float = 0.30
    person_confidence: float = 0.55
    cropped_side_span_ratio: float = 0.18
    cropped_top_span_ratio: float = 0.12
    cropped_bottom_span_ratio: float = 0.28


@dataclass
class EdgeConfig:
    enabled: bool = True
    alpha_low: int = 8
    alpha_high: int = 247
    background_inpaint_radius: int = 7
    correction_strength: float = 0.85
    low_alpha_protection: float = 0.12


@dataclass
class VerificationConfig:
    enabled: bool = True
    model: str = "ssdlite320_mobilenet_v3_large"
    run_on_all: bool = True


@dataclass
class PreviewConfig:
    width: int = 1000
    panel_height: int = 320
    contact_columns: int = 4
    contact_thumb_width: int = 300


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")


def _merge(obj: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if hasattr(obj, key):
            setattr(obj, key, tuple(value) if key == "extensions" else value)


def load_config(path: str | Path | None = None) -> AppConfig:
    cfg = AppConfig()
    candidate = Path(path) if path else Path(__file__).resolve().parent.parent / "config.yaml"
    if not candidate.exists():
        return cfg
    raw = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    _merge(cfg.model, raw.get("model", {}))
    _merge(cfg.qc, raw.get("qc", {}))
    _merge(cfg.preview, raw.get("preview", {}))
    _merge(cfg.edge, raw.get("edge", {}))
    _merge(cfg.verification, raw.get("verification", {}))
    if "extensions" in raw:
        cfg.extensions = tuple(e.lower() for e in raw["extensions"])
    return cfg
