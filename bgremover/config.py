from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    primary: str = "ZhengPeng7/BiRefNet-portrait"
    fallback: str = "ZhengPeng7/BiRefNet_HR-matting"
    primary_revision: str = "ecdeb6240ef23557dbd48ff27c59c1a88cbcb755"
    fallback_revision: str = "5d6b6f8adcb5b417c871b1d84ceaae9871355b7f"
    image_size: int = 1024
    fallback_image_size: int = 2048
    precision: str = "fp16"


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
    # Legacy diagnostic refiner. Production uses ForegroundConfig below.
    enabled: bool = False
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
    sam_enabled: bool = True
    sam_model: str = "facebook/sam2.1-hiera-small"
    sam_revision: str = "ee5bba1d82bb8749febdf90f45e84b687142ba03"
    sam_prompt_confidence: float = 0.80
    sam_mask_confidence: float = 0.80
    sam_alpha_threshold: int = 24
    sam_trigger_box_coverage: float = 0.45
    sam_trigger_multiple_coverage: float = 0.60
    sam_min_person_recall: float = 0.82
    sam_min_missing_box_ratio: float = 0.02
    sam_boundary_tolerance_ratio: float = 0.006


@dataclass
class ForegroundConfig:
    method: str = "pymatting_ml"
    enabled: bool = True


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
    foreground: ForegroundConfig = field(default_factory=ForegroundConfig)
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
    _merge(cfg.foreground, raw.get("foreground", {}))
    _merge(cfg.verification, raw.get("verification", {}))
    if "extensions" in raw:
        cfg.extensions = tuple(e.lower() for e in raw["extensions"])
    return cfg
