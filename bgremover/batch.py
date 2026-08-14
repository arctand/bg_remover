from __future__ import annotations

import random
import statistics
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .config import AppConfig
from .foreground import PyMattingForegroundRefiner
from .images import atomic_save_png, discover_images, load_rgb, mask_array, rgba_from_mask
from .previews import make_contact_sheet, make_preview
from .qc import analyze_mask
from .reporting import read_completed, write_csv_atomic, write_json_atomic
from .verification import HumanVerification, SAM21PersonVerifier, should_run_sam


@dataclass
class Progress:
    processed: int
    total: int
    current: str
    counts: dict
    average: float


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


class BatchProcessor:
    def __init__(
        self,
        config: AppConfig,
        primary,
        verifier=None,
        foreground_refiner=None,
        strong_verifier=None,
    ):
        self.config = config
        self.primary = primary
        self.verifier = verifier
        self.foreground_refiner = foreground_refiner or PyMattingForegroundRefiner(config.foreground)
        self.strong_verifier = strong_verifier
        if self.strong_verifier is None and verifier and config.verification.sam_enabled:
            self.strong_verifier = SAM21PersonVerifier(config.verification)
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def _paths(self, root: Path, test: bool):
        base = root / "debug_output" if test else root
        return base, base / "report.csv", base / "summary.json"

    def run(
        self,
        source: Path,
        destination: Path,
        test: bool = False,
        sample_size: int = 25,
        resume: bool = True,
        callback: Callable[[Progress], None] | None = None,
    ):
        source, destination = source.resolve(), destination.resolve()
        files = list(discover_images(source, self.config.extensions))
        if test and len(files) > sample_size:
            files = random.Random(42).sample(files, sample_size)
        base, report_path, summary_path = self._paths(destination, test)
        for folder in ("ready", "review", "failed"):
            (base / folder).mkdir(parents=True, exist_ok=True)
        if test:
            (base / "previews").mkdir(parents=True, exist_ok=True)
        completed = read_completed(report_path) if resume else {}
        rows = list(completed.values())
        done = set(completed)
        counts = Counter(row.get("status", "") for row in rows)
        timings = [float(row.get("processing_time", 0) or 0) for row in rows]
        sam_runs = sum(_truthy(row.get("sam_ran")) for row in rows)
        sam_errors = sum(bool(row.get("sam_error")) for row in rows)
        preview_items = []
        self.stop_event.clear()

        # Always-on models and CPU refinement warm-up are excluded from per-photo timing.
        if hasattr(self.primary, "load") and getattr(self.primary, "model", None) is None:
            self.primary.load()
        if self.verifier and hasattr(self.verifier, "load") and getattr(self.verifier, "model", None) is None:
            self.verifier.load()
        if hasattr(self.foreground_refiner, "warm_up"):
            self.foreground_refiner.warm_up()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        for path in files:
            relative = path.relative_to(source)
            key = relative.as_posix()
            if key in done:
                continue
            started = time.perf_counter()
            row = {"source_file": key}
            status, output = "FAILED", ""
            try:
                rgb = load_rgb(path)
                try:
                    mask = self.primary.predict(rgb)
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    self.primary.clear_cache()
                    mask = self.primary.predict(rgb)
                alpha = mask_array(mask)
                human = self.verifier.verify(rgb, alpha) if self.verifier else HumanVerification()

                refined = self.foreground_refiner.refine(rgb, mask)
                if refined.rgb.size != rgb.size or refined.alpha.size != rgb.size:
                    raise RuntimeError("Foreground refinement changed output dimensions")
                if not np.array_equal(mask_array(refined.alpha), alpha):
                    raise RuntimeError("Foreground refinement changed portrait alpha")

                qc = analyze_mask(alpha, self.config.qc)
                sam_requested = should_run_sam(qc.review_reasons, human, self.config.verification)
                sam_result = None
                sam_error = ""
                if sam_requested and self.strong_verifier:
                    try:
                        sam_result = self.strong_verifier.verify(rgb, alpha, human)
                        if sam_result.ran:
                            sam_runs += 1
                        if not sam_result.ran:
                            qc.review_reasons.append("low_confidence")
                            qc.review_details.append("strong verification was requested but no high-confidence prompt was available")
                        qc.review_reasons.extend(sam_result.reasons)
                        qc.review_details.extend(sam_result.details)
                    except Exception as exc:
                        sam_errors += 1
                        sam_error = f"{type(exc).__name__}: {exc}"
                        qc.review_reasons.append("low_confidence")
                        qc.review_details.append("strong verifier failed on a suspicious result")
                qc.review_reasons = list(dict.fromkeys(qc.review_reasons))

                status = "REVIEW" if qc.needs_review else "READY"
                target = base / status.lower() / relative.with_suffix(".png")
                rgba = rgba_from_mask(refined.rgb, refined.alpha)
                atomic_save_png(rgba, target)
                output = target.relative_to(base).as_posix()
                if test:
                    preview = base / "previews" / relative.with_suffix(".png")
                    make_preview(rgb, rgba, preview, self.config.preview)
                    preview_items.append((preview, key, status.lower()))

                row.update(qc.as_dict())
                row.update(
                    width=rgb.width,
                    height=rgb.height,
                    person_count=human.person_count,
                    person_detector_zero=human.person_detector_zero,
                    person_box_coverage_min=(f"{min(human.coverages):.4f}" if human.coverages else ""),
                    sam_requested=sam_requested,
                    sam_ran=bool(sam_result and sam_result.ran),
                    sam_prompted_boxes=(sam_result.prompted_boxes if sam_result else 0),
                    sam_checked_people=(sam_result.checked_people if sam_result else 0),
                    sam_min_recall=(f"{sam_result.min_recall:.4f}" if sam_result and sam_result.min_recall is not None else ""),
                    sam_min_iou=(f"{sam_result.min_iou:.4f}" if sam_result and sam_result.min_iou is not None else ""),
                    sam_error=sam_error,
                    foreground_refinement=self.config.foreground.method,
                )
            except Exception as exc:
                row.update(
                    error=f"{type(exc).__name__}: {exc}",
                    review_reason="",
                    review_details="",
                    width="",
                    height="",
                    foreground_ratio="",
                    touch_top="",
                    touch_bottom="",
                    touch_left="",
                    touch_right="",
                )

            elapsed = time.perf_counter() - started
            timings.append(elapsed)
            counts[status] += 1
            info = getattr(self.primary, "info", None)
            row.update(
                filename=relative.with_suffix(".png").as_posix(),
                output_file=output,
                status=status,
                processing_time=f"{elapsed:.4f}",
                device=getattr(info, "name", "test"),
                precision=getattr(info, "precision", "test"),
                error=row.get("error", ""),
            )
            rows.append(row)
            write_csv_atomic(report_path, rows)
            if callback:
                callback(Progress(len(rows), len(files), key, dict(counts), statistics.mean(timings)))
            if self.stop_event.is_set():
                break

        if test:
            make_contact_sheet(preview_items, base / "contact_sheet" / "contact_sheet.png", self.config.preview)
        try:
            import torch

            peak_vram = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        except Exception:
            peak_vram = 0
        summary = {
            "total": len(rows),
            "ready": counts["READY"],
            "review": counts["REVIEW"],
            "failed": counts["FAILED"],
            "stopped": self.stop_event.is_set(),
            "total_processing_time": sum(timings),
            "average_processing_time": statistics.mean(timings) if timings else 0,
            "median_processing_time": statistics.median(timings) if timings else 0,
            "peak_vram_gb": peak_vram,
            "p95_processing_time": float(np.percentile(timings, 95)) if timings else 0,
            "sam_runs": sam_runs,
            "sam_run_percent": (100.0 * sam_runs / len(rows)) if rows else 0.0,
            "sam_errors": sam_errors,
            "model": self.config.model.primary,
            "model_revision": self.config.model.primary_revision,
            "foreground_refinement": self.config.foreground.method,
            "device": getattr(getattr(self.primary, "info", None), "name", "test"),
            "precision": getattr(getattr(self.primary, "info", None), "precision", "test"),
        }
        write_json_atomic(summary_path, summary)
        return summary
