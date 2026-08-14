from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str
    revision: str
    image_size: int


@dataclass(frozen=True)
class PreviewSpec:
    panel_width: int = 300
    panel_height: int = 300
    contact_columns: int = 2


@dataclass(frozen=True)
class BenchmarkConfig:
    config_path: Path
    source: Path
    output: Path
    device: str
    precision: str
    count: int
    extensions: tuple[str, ...]
    models: tuple[ModelSpec, ...]
    guidance_enabled: bool
    guidance_base_model: str
    guidance_detector: str
    guidance_margin_ratio: float
    ssdlite_confidence: float
    sam_model_id: str
    sam_revision: str
    refinement_mask_variant: str
    refinement_radius: int
    refinement_methods: tuple[str, ...]
    preview: PreviewSpec
    regressions: dict[str, str] = field(default_factory=dict)


def _resolved(base: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def load_benchmark_config(path: str | Path = "benchmark_quality.yaml") -> BenchmarkConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base = config_path.parent
    runtime = raw["runtime"]
    guidance = raw["person_guidance"]
    verifiers = raw["verifiers"]
    sam = verifiers["sam21"]
    refinement = raw["refinement"]
    preview = raw.get("preview", {})
    models = tuple(
        ModelSpec(name=name, model_id=item["model_id"], revision=item["revision"], image_size=int(item["image_size"]))
        for name, item in raw["models"].items()
    )
    return BenchmarkConfig(
        config_path=config_path,
        source=_resolved(base, raw["paths"]["source"]),
        output=_resolved(base, raw["paths"]["output"]),
        device=str(runtime.get("device", "cuda")),
        precision=str(runtime.get("precision", "fp16")),
        count=int(runtime.get("count", 20)),
        extensions=tuple(str(value).lower() for value in runtime.get("extensions", [])),
        models=models,
        guidance_enabled=bool(guidance.get("enabled", True)),
        guidance_base_model=str(guidance["base_model"]),
        guidance_detector=str(guidance["detector"]),
        guidance_margin_ratio=float(guidance.get("box_margin_ratio", 0.0)),
        ssdlite_confidence=float(verifiers.get("ssdlite_confidence", 0.55)),
        sam_model_id=str(sam["model_id"]),
        sam_revision=str(sam["revision"]),
        refinement_mask_variant=str(refinement["mask_variant"]),
        refinement_radius=int(refinement.get("official_radius", 90)),
        refinement_methods=tuple(refinement["methods"]),
        preview=PreviewSpec(
            panel_width=int(preview.get("panel_width", 300)),
            panel_height=int(preview.get("panel_height", 300)),
            contact_columns=int(preview.get("contact_columns", 2)),
        ),
        regressions={str(key): str(value) for key, value in raw.get("regressions", {}).items()},
    )
