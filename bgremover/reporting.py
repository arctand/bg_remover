from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from uuid import uuid4


REPORT_FIELDS = [
    "source_file", "filename", "output_file", "width", "height", "status", "processing_time",
    "device", "precision", "foreground_ratio", "touch_top", "touch_bottom",
    "touch_left", "touch_right", "review_reason", "review_details", "person_count",
    "missing_person_count", "edge_correction_p95", "edge_clipped_ratio",
    "secondary_model_used", "mask_iou", "alpha_difference", "error",
]


def read_completed(path: Path) -> dict[str, dict]:
    if not path.exists(): return {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return {row["source_file"]: row for row in csv.DictReader(fh) if row.get("source_file")}


def write_csv_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, REPORT_FIELDS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
