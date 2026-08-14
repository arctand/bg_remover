from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from quality_benchmark.config import load_benchmark_config
from quality_benchmark.metrics import binary_overlap, mask_statistics, summarize_rows
from quality_benchmark.models import expand_box, paste_crop_mask
from quality_benchmark.render import make_variant_preview


def test_config_is_reproducibly_pinned():
    config = load_benchmark_config()
    assert config.count == 20
    assert config.output.name == "benchmark_output"
    assert {spec.name for spec in config.models} == {"hr_matting", "portrait", "dynamic_matting"}
    assert all(len(spec.revision) == 40 for spec in config.models)
    assert len(config.sam_revision) == 40


def test_guided_mask_is_pasted_only_inside_expanded_box():
    box = expand_box([20, 20, 40, 50], (100, 80), 0.1)
    assert box == [18, 17, 42, 53]
    crop = Image.new("L", (8, 8), 255)
    canvas = np.asarray(paste_crop_mask((100, 80), box, crop))
    assert canvas[17:53, 18:42].min() == 255
    assert canvas[:17].max() == 0
    assert canvas[:, :18].max() == 0


def test_overlap_and_summary_metrics():
    first = Image.fromarray(np.array([[255, 255], [0, 0]], dtype=np.uint8))
    second = Image.fromarray(np.array([[255, 0], [255, 0]], dtype=np.uint8))
    overlap = binary_overlap(first, second)
    assert overlap == {"iou": 1 / 3, "precision": 0.5, "recall": 0.5}
    assert mask_statistics(first)["foreground_ratio"] == 0.5
    summary = summarize_rows([
        {"stage": "mask", "variant": "x", "status": "ok", "time_s": 1.0, "peak_vram_gb": 2.0},
        {"stage": "mask", "variant": "x", "status": "ok", "time_s": 3.0, "peak_vram_gb": 2.5},
        {"stage": "mask", "variant": "x", "status": "error", "error": "boom"},
    ])["mask:x"]
    assert summary["average_time_s"] == 2.0
    assert summary["median_time_s"] == 2.0
    assert summary["errors"] == 1
    assert summary["peak_vram_gb"] == 2.5


def test_preview_contains_all_background_panels(tmp_path: Path):
    original = Image.new("RGB", (40, 30), "red")
    alpha = Image.new("L", (40, 30), 128)
    config = load_benchmark_config().preview
    target = tmp_path / "preview.jpg"
    make_variant_preview(original, alpha, original, "case", "metrics", target, config)
    with Image.open(target) as preview:
        assert preview.width == config.panel_width * 6
        assert preview.height == config.panel_height + 58
    assert json.dumps({"path": str(target)})
