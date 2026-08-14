from __future__ import annotations

import math
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image


def mask_statistics(mask: Image.Image) -> dict[str, float | int]:
    alpha = np.asarray(mask.convert("L"), dtype=np.uint8)
    return {
        "foreground_ratio": float((alpha >= 24).mean()),
        "solid_ratio": float((alpha >= 224).mean()),
        "translucent_ratio": float(((alpha > 8) & (alpha < 247)).mean()),
        "edge_pixels": int(((alpha > 8) & (alpha < 247)).sum()),
    }


def binary_overlap(mask: Image.Image, reference: Image.Image, threshold: int = 127) -> dict[str, float]:
    predicted = np.asarray(mask.convert("L"), dtype=np.uint8) >= threshold
    expected = np.asarray(reference.convert("L"), dtype=np.uint8) >= threshold
    intersection = int(np.logical_and(predicted, expected).sum())
    union = int(np.logical_or(predicted, expected).sum())
    predicted_count = int(predicted.sum())
    expected_count = int(expected.sum())
    return {
        "iou": float(intersection / max(1, union)),
        "precision": float(intersection / max(1, predicted_count)),
        "recall": float(intersection / max(1, expected_count)),
    }


def box_coverage(mask: Image.Image, boxes: list[list[int]], threshold: int = 24) -> dict[str, float | int]:
    alpha = np.asarray(mask.convert("L"), dtype=np.uint8)
    foreground = alpha >= threshold
    h, w = foreground.shape
    coverages: list[float] = []
    centers: list[float] = []
    for raw in boxes:
        x1, y1, x2, y2 = max(0, raw[0]), max(0, raw[1]), min(w, raw[2]), min(h, raw[3])
        if x2 <= x1 or y2 <= y1:
            continue
        region = foreground[y1:y2, x1:x2]
        mx, my = int((x2 - x1) * 0.22), int((y2 - y1) * 0.10)
        center = foreground[y1 + my:y2 - my, x1 + mx:x2 - mx]
        coverages.append(float(region.mean()))
        centers.append(float(center.mean()) if center.size else 0.0)
    return {
        "person_count": len(coverages),
        "box_coverage_min": min(coverages, default=0.0),
        "box_coverage_mean": float(np.mean(coverages)) if coverages else 0.0,
        "center_coverage_min": min(centers, default=0.0),
    }


def foreground_reconstruction_metrics(image: Image.Image, alpha: Image.Image, foreground: Image.Image) -> dict[str, float | int]:
    source = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    matte_u8 = np.asarray(alpha.convert("L"), dtype=np.uint8)
    matte = matte_u8.astype(np.float32) / 255.0
    estimated = np.asarray(foreground.convert("RGB"), dtype=np.float32) / 255.0
    edge = (matte_u8 > 8) & (matte_u8 < 247)
    if not np.any(edge):
        return {"edge_pixels": 0, "reconstruction_mae": 0.0, "correction_mean": 0.0, "correction_p95": 0.0}
    unknown = (matte_u8 > 8).astype(np.uint8) * 255
    background = cv2.inpaint((source * 255).astype(np.uint8), unknown, 7, cv2.INPAINT_TELEA).astype(np.float32) / 255.0
    reconstructed = matte[..., None] * estimated + (1.0 - matte[..., None]) * background
    reconstruction_error = np.abs(reconstructed - source).mean(axis=2)[edge]
    correction = np.linalg.norm(estimated - source, axis=2)[edge] / math.sqrt(3.0)
    return {
        "edge_pixels": int(edge.sum()),
        "reconstruction_mae": float(reconstruction_error.mean()),
        "correction_mean": float(correction.mean()),
        "correction_p95": float(np.percentile(correction, 95)),
    }


def summarize_rows(rows: list[dict]) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[f"{row.get('stage', '')}:{row.get('variant', '')}"].append(row)
    summary: dict[str, dict[str, float | int]] = {}
    for key, items in sorted(groups.items()):
        times = np.asarray([float(item["time_s"]) for item in items if item.get("status") == "ok" and item.get("time_s") not in (None, "")])
        peaks = [float(item.get("peak_vram_gb", 0.0) or 0.0) for item in items if item.get("status") == "ok"]
        summary[key] = {
            "count": len(items),
            "errors": sum(item.get("status") != "ok" for item in items),
            "average_time_s": float(times.mean()) if times.size else 0.0,
            "median_time_s": float(np.median(times)) if times.size else 0.0,
            "p95_time_s": float(np.percentile(times, 95)) if times.size else 0.0,
            "peak_vram_gb": max(peaks, default=0.0),
        }
    return summary
