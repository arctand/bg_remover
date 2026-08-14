from __future__ import annotations

import csv
import json
import platform
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from bgremover.config import load_config
from bgremover.qc import analyze_mask

from .config import BenchmarkConfig, ModelSpec
from .metrics import (
    binary_overlap,
    box_coverage,
    foreground_reconstruction_metrics,
    mask_statistics,
    summarize_rows,
)
from .models import ResearchBiRefNet, SAM21Verifier, SSDLiteDetector
from .refinement import (
    raw_foreground,
    refine_birefnet_official,
    refine_current_edge,
    refine_pymatting,
    warm_up_pymatting,
)
from .render import make_contact_sheet, make_side_by_side, make_variant_preview


Progress = Callable[[str], None]


def _discover(config: BenchmarkConfig) -> list[Path]:
    if not config.source.is_dir():
        raise FileNotFoundError(f"Benchmark source does not exist: {config.source}")
    files = sorted(
        path for path in config.source.iterdir()
        if path.is_file() and path.suffix.lower() in config.extensions
    )
    if not files:
        raise RuntimeError(f"No supported images in {config.source}")
    return files[:config.count]


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _mask_path(output: Path, variant: str, case: str) -> Path:
    return output / "masks" / variant / f"{case}.png"


def _row(case: str, stage: str, variant: str, status: str = "ok", **values) -> dict:
    return {"case": case, "stage": stage, "variant": variant, "status": status, **values}


def _environment() -> dict:
    import torch

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3)
        if torch.cuda.is_available() else 0.0,
    }


def _serializable_config(config: BenchmarkConfig) -> dict:
    return {
        "source": str(config.source),
        "output": str(config.output),
        "device": config.device,
        "precision": config.precision,
        "count": config.count,
        "models": [asdict(spec) for spec in config.models],
        "guidance": {
            "enabled": config.guidance_enabled,
            "base_model": config.guidance_base_model,
            "detector": config.guidance_detector,
            "margin_ratio": config.guidance_margin_ratio,
        },
        "sam21": {"model_id": config.sam_model_id, "revision": config.sam_revision},
        "refinement": {
            "mask_variant": config.refinement_mask_variant,
            "radius": config.refinement_radius,
            "methods": list(config.refinement_methods),
        },
        "regressions": config.regressions,
    }


def _write_metrics(config: BenchmarkConfig, rows: list[dict], loads: dict, cases: list[str]) -> None:
    config.output.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    preferred = ["case", "stage", "variant", "status", "time_s", "peak_vram_gb", "error"]
    ordered = [name for name in preferred if name in fields] + [name for name in fields if name not in preferred]
    with (config.output / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "environment": _environment(),
        "config": _serializable_config(config),
        "cases": cases,
        "model_load_time_s": loads,
        "summary": summarize_rows(rows),
        "rows": rows,
    }
    (config.output / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _qc_values(mask: Image.Image) -> dict:
    production = load_config()
    qc = analyze_mask(np.asarray(mask.convert("L"), dtype=np.uint8), production.qc)
    return {
        "qc_needs_review": qc.needs_review,
        "qc_reasons": "|".join(qc.review_reasons),
        "cropped_source": "cropped_source" in qc.review_reasons,
    }


def run_benchmark(config: BenchmarkConfig, progress: Progress = print) -> dict:
    import torch

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("RTX/CUDA is required for this benchmark")

    images = _discover(config)
    cases = [path.stem for path in images]
    config.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    indexed_rows: dict[tuple[str, str], dict] = {}
    detections: dict[str, list[list[int]]] = {}
    loads: dict[str, float] = {}

    progress(f"Dataset: {len(images)} images from {config.source}")
    progress("Stage 1/5: SSDLite person detection")
    detector = SSDLiteDetector(config.ssdlite_confidence, config.device)
    try:
        detector.load()
        loads["ssdlite_detector"] = detector.load_time_s
        for index, path in enumerate(images, 1):
            case = path.stem
            try:
                result = detector.detect(_load_rgb(path))
                detections[case] = result.value["boxes"]
                rows.append(_row(
                    case, "detector", "ssdlite320_mobilenet_v3_large",
                    time_s=result.time_s,
                    peak_vram_gb=result.peak_vram_gb,
                    person_count=len(result.value["boxes"]),
                    scores="|".join(f"{score:.6f}" for score in result.value["scores"]),
                ))
            except Exception as exc:
                detections[case] = []
                rows.append(_row(case, "detector", "ssdlite320_mobilenet_v3_large", "error", error=str(exc)))
            progress(f"  detector {index}/{len(images)} {case}: {len(detections[case])} person boxes")
    finally:
        detector.unload()
    _write_metrics(config, rows, loads, cases)

    progress("Stage 2/5: BiRefNet mask variants (one model resident at a time)")
    for spec in config.models:
        backend = ResearchBiRefNet(spec, config.device, config.precision)
        try:
            progress(f"  loading {spec.name}: {spec.model_id}@{spec.revision[:8]}")
            backend.load()
            loads[spec.name] = backend.load_time_s
            for index, path in enumerate(images, 1):
                case = path.stem
                image = _load_rgb(path)
                try:
                    result = backend.predict(image)
                    mask = result.value
                    target = _mask_path(config.output, spec.name, case)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    mask.save(target)
                    values = {
                        "model_id": spec.model_id,
                        "revision": spec.revision,
                        "input_size": spec.image_size,
                        "time_s": result.time_s,
                        "peak_vram_gb": result.peak_vram_gb,
                        **mask_statistics(mask),
                        **box_coverage(mask, detections.get(case, [])),
                        **_qc_values(mask),
                    }
                    row = _row(case, "mask", spec.name, **values)
                    rows.append(row)
                    indexed_rows[(spec.name, case)] = row
                except Exception as exc:
                    row = _row(case, "mask", spec.name, "error", model_id=spec.model_id, error=str(exc))
                    rows.append(row)
                    indexed_rows[(spec.name, case)] = row

                if config.guidance_enabled and spec.name == config.guidance_base_model:
                    guided_name = f"{spec.name}_person_guided"
                    try:
                        result = backend.predict_guided(
                            image, detections.get(case, []), config.guidance_margin_ratio
                        )
                        mask = result.value
                        target = _mask_path(config.output, guided_name, case)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        mask.save(target)
                        values = {
                            "model_id": spec.model_id,
                            "revision": spec.revision,
                            "input_size": spec.image_size,
                            "guidance": config.guidance_detector,
                            "guidance_margin_ratio": config.guidance_margin_ratio,
                            "time_s": result.time_s,
                            "peak_vram_gb": result.peak_vram_gb,
                            **mask_statistics(mask),
                            **box_coverage(mask, detections.get(case, [])),
                            **_qc_values(mask),
                        }
                        row = _row(case, "mask", guided_name, **values)
                        rows.append(row)
                        indexed_rows[(guided_name, case)] = row
                    except Exception as exc:
                        row = _row(case, "mask", guided_name, "error", model_id=spec.model_id, error=str(exc))
                        rows.append(row)
                        indexed_rows[(guided_name, case)] = row
                progress(f"  {spec.name} {index}/{len(images)} {case}")
        except Exception as exc:
            progress(f"  FAILED TO LOAD {spec.name}: {exc}")
            for path in images:
                row = _row(path.stem, "mask", spec.name, "error", model_id=spec.model_id, error=f"load: {exc}")
                rows.append(row)
                indexed_rows[(spec.name, path.stem)] = row
        finally:
            backend.unload()
            _write_metrics(config, rows, loads, cases)

    mask_variants = [spec.name for spec in config.models]
    if config.guidance_enabled:
        mask_variants.insert(1, f"{config.guidance_base_model}_person_guided")

    progress("Stage 3/5: SAM 2.1 prompted-mask verification")
    sam = SAM21Verifier(config.sam_model_id, config.sam_revision)
    try:
        sam.load()
        loads["sam21_verifier"] = sam.load_time_s
        for index, path in enumerate(images, 1):
            case = path.stem
            try:
                result = sam.verify(_load_rgb(path), detections.get(case, []))
                sam_mask = result.value["mask"]
                target = config.output / "verifiers" / "sam21" / f"{case}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                sam_mask.save(target)
                rows.append(_row(
                    case, "verifier", "sam2.1_hiera_small",
                    model_id=config.sam_model_id,
                    revision=config.sam_revision,
                    time_s=result.time_s,
                    peak_vram_gb=result.peak_vram_gb,
                    scores="|".join(f"{score:.6f}" for score in result.value["scores"]),
                    prompted_boxes=len(detections.get(case, [])),
                ))
                for variant in mask_variants:
                    path_mask = _mask_path(config.output, variant, case)
                    if path_mask.exists() and (variant, case) in indexed_rows:
                        with Image.open(path_mask) as model_mask:
                            overlap = binary_overlap(model_mask, sam_mask)
                        indexed_rows[(variant, case)].update({
                            "sam_iou": overlap["iou"],
                            "sam_precision": overlap["precision"],
                            "sam_recall": overlap["recall"],
                        })
            except Exception as exc:
                rows.append(_row(case, "verifier", "sam2.1_hiera_small", "error", error=str(exc)))
            progress(f"  SAM2.1 {index}/{len(images)} {case}")
    except Exception as exc:
        progress(f"  FAILED TO LOAD SAM2.1: {exc}")
        for path in images:
            rows.append(_row(path.stem, "verifier", "sam2.1_hiera_small", "error", error=f"load: {exc}"))
    finally:
        sam.unload()
        _write_metrics(config, rows, loads, cases)

    progress("Stage 4/5: foreground RGB refinement on the HR mask")
    production_config = load_config()
    if "pymatting_ml" in config.refinement_methods:
        warm_up_pymatting()
    refinement_previews: dict[str, list[Path]] = {method: [] for method in config.refinement_methods}
    model_previews: dict[str, list[Path]] = {variant: [] for variant in mask_variants}
    case_previews: dict[str, list[tuple[str, Path]]] = {case: [] for case in cases}

    for index, path in enumerate(images, 1):
        case = path.stem
        image = _load_rgb(path)
        base_mask_path = _mask_path(config.output, config.refinement_mask_variant, case)
        if not base_mask_path.exists():
            for method in config.refinement_methods:
                rows.append(_row(case, "refinement", method, "error", error="base mask missing"))
            continue
        with Image.open(base_mask_path) as stored_mask:
            base_mask = stored_mask.convert("L")
        for method in config.refinement_methods:
            try:
                if method == "raw_rgb":
                    result = raw_foreground(image)
                    foreground = result.value
                    extra = {}
                elif method == "current_edge":
                    result = refine_current_edge(image, base_mask, production_config.edge)
                    foreground = result.value["foreground"]
                    extra = {f"current_{key}": value for key, value in result.value["edge_metrics"].items()}
                elif method == "birefnet_official":
                    result = refine_birefnet_official(image, base_mask, config.refinement_radius)
                    foreground = result.value
                    extra = {"official_radius": config.refinement_radius}
                elif method == "pymatting_ml":
                    result = refine_pymatting(image, base_mask)
                    foreground = result.value
                    extra = {"pymatting_version": "1.1.15"}
                else:
                    raise ValueError(f"Unknown refinement method: {method}")
                quality = foreground_reconstruction_metrics(image, base_mask, foreground)
                row = _row(
                    case, "refinement", method,
                    mask_variant=config.refinement_mask_variant,
                    time_s=result.time_s,
                    peak_vram_gb=result.peak_vram_gb,
                    **quality,
                    **extra,
                )
                rows.append(row)
                preview_path = config.output / "previews" / "refinements" / method / f"{case}.jpg"
                make_variant_preview(
                    image, base_mask, foreground,
                    f"{case} | {method}",
                    f"time={result.time_s:.3f}s recon_MAE={quality['reconstruction_mae']:.4f}",
                    preview_path, config.preview,
                )
                refinement_previews[method].append(preview_path)
                case_previews[case].append((f"refine:{method}", preview_path))
            except Exception as exc:
                rows.append(_row(
                    case, "refinement", method, "error",
                    mask_variant=config.refinement_mask_variant,
                    error=f"{exc}\n{traceback.format_exc(limit=2)}",
                ))
        progress(f"  refinement {index}/{len(images)} {case}")
        _write_metrics(config, rows, loads, cases)

    progress("Stage 5/5: previews, contact sheets, and side-by-side comparisons")
    for variant in mask_variants:
        for path in images:
            case = path.stem
            stored_path = _mask_path(config.output, variant, case)
            if not stored_path.exists():
                continue
            image = _load_rgb(path)
            with Image.open(stored_path) as stored_mask:
                mask = stored_mask.convert("L")
            metric_row = indexed_rows.get((variant, case), {})
            subtitle = (
                f"time={float(metric_row.get('time_s', 0)):.3f}s "
                f"VRAM={float(metric_row.get('peak_vram_gb', 0)):.2f}GB "
                f"SAM-IoU={float(metric_row.get('sam_iou', 0)):.3f} "
                f"QC={metric_row.get('qc_reasons', '') or 'READY'}"
            )
            preview_path = config.output / "previews" / "masks" / variant / f"{case}.jpg"
            make_variant_preview(image, mask, image, f"{case} | {variant}", subtitle, preview_path, config.preview)
            model_previews[variant].append(preview_path)
            case_previews[case].insert(0, (f"mask:{variant}", preview_path))

    for variant, previews in model_previews.items():
        if previews:
            make_contact_sheet(
                previews,
                config.output / "contact_sheets" / f"masks_{variant}.jpg",
                config.preview.contact_columns,
            )
    for method, previews in refinement_previews.items():
        if previews:
            make_contact_sheet(
                previews,
                config.output / "contact_sheets" / f"refinement_{method}.jpg",
                config.preview.contact_columns,
            )
    for case, previews in case_previews.items():
        if previews:
            make_side_by_side(case, previews, config.output / "side_by_side" / f"{case}.jpg")

    regression_paths = [
        config.output / "side_by_side" / f"{case}.jpg"
        for case in config.regressions
        if (config.output / "side_by_side" / f"{case}.jpg").exists()
    ]
    if regression_paths:
        make_contact_sheet(
            regression_paths,
            config.output / "contact_sheets" / "regression_cases.jpg",
            2,
            760,
        )

    _write_metrics(config, rows, loads, cases)
    result = {
        "output": str(config.output),
        "metrics_csv": str(config.output / "metrics.csv"),
        "metrics_json": str(config.output / "metrics.json"),
        "summary": summarize_rows(rows),
    }
    progress(json.dumps(result, ensure_ascii=False, indent=2))
    return result
