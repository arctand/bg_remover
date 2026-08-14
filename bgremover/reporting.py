from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from uuid import uuid4


REPORT_FIELDS = [
    "source_file", "filename", "output_file", "width", "height", "status", "processing_time",
    "device", "precision", "foreground_ratio", "touch_top", "touch_bottom",
    "touch_left", "touch_right", "cropped_source_signal", "review_reason", "review_details",
    "person_count", "person_detector_zero", "person_box_coverage_min", "sam_requested",
    "sam_ran", "sam_prompted_boxes", "sam_checked_people", "sam_min_recall", "sam_min_iou",
    "sam_error", "foreground_refinement", "pipeline_fingerprint", "error",
]


def read_completed(path: Path, expected_fingerprint: str | None = None) -> dict[str, dict]:
    if not path.exists(): return {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return {
            row["source_file"]: row
            for row in csv.DictReader(fh)
            if row.get("source_file")
            and (
                expected_fingerprint is None
                or row.get("pipeline_fingerprint") == expected_fingerprint
            )
        }


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
