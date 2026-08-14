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
from .edge import decontaminate_rgb
from .images import atomic_save_png, discover_images, load_rgb, mask_array, rgba_from_mask
from .previews import make_contact_sheet, make_preview
from .qc import analyze_mask, mask_similarity
from .reporting import read_completed, write_csv_atomic, write_json_atomic


@dataclass
class Progress:
    processed: int; total: int; current: str; counts: dict; average: float


class BatchProcessor:
    def __init__(self, config: AppConfig, primary, secondary=None, verifier=None):
        self.config, self.primary, self.secondary, self.verifier = config, primary, secondary, verifier
        self.stop_event = threading.Event()

    def stop(self): self.stop_event.set()

    def _paths(self, root: Path, test: bool):
        base = root / "debug_output" if test else root
        return base, base / "report.csv", base / "summary.json"

    def run(self, source: Path, destination: Path, test=False, sample_size=25,
            resume=True, callback: Callable[[Progress], None] | None = None):
        source, destination = source.resolve(), destination.resolve()
        files = list(discover_images(source, self.config.extensions))
        if test and len(files) > sample_size:
            files = random.Random(42).sample(files, sample_size)
        base, report_path, summary_path = self._paths(destination, test)
        for folder in ("ready", "review", "failed"):
            (base / folder).mkdir(parents=True, exist_ok=True)
        if test: (base / "previews").mkdir(parents=True, exist_ok=True)
        completed = read_completed(report_path) if resume else {}
        rows = list(completed.values())
        done = set(completed)
        counts = Counter(row.get("status", "") for row in rows)
        timings = [float(r.get("processing_time", 0) or 0) for r in rows]
        preview_items = []
        self.stop_event.clear()
        # Model loading is one-time setup, not per-image processing time.
        if hasattr(self.primary, "load") and getattr(self.primary, "model", None) is None: self.primary.load()
        if self.verifier and hasattr(self.verifier, "load") and getattr(self.verifier, "model", None) is None: self.verifier.load()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        except Exception: pass
        for path in files:
            relative = path.relative_to(source)
            key = relative.as_posix()
            if key in done: continue
            started = time.perf_counter(); row = {"source_file": key}
            status, output = "FAILED", ""
            try:
                rgb = load_rgb(path)
                try:
                    mask = self.primary.predict(rgb)
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower(): raise
                    self.primary.clear_cache(); mask = self.primary.predict(rgb)
                alpha = mask_array(mask)
                human = self.verifier.verify(rgb, alpha) if self.verifier else None
                corrected_rgb, edge_metrics = decontaminate_rgb(rgb, mask, self.config.edge)
                qc = analyze_mask(alpha, self.config.qc, edge_metrics, human)
                secondary_used, iou, difference = False, "", ""
                if qc.needs_review and self.secondary and self.config.model.secondary_enabled:
                    secondary_used = True
                    second = mask_array(self.secondary.predict(rgb))
                    iou, difference = mask_similarity(alpha, second)
                    if iou < self.config.model.disagreement_iou or difference > self.config.model.disagreement_alpha:
                        qc.review_reasons.append("secondary_disagreement"); qc.review_details.append(f"secondary IoU={iou:.3f}, alpha difference={difference:.3f}")
                status = "REVIEW" if qc.needs_review else "READY"
                target = base / status.lower() / relative.with_suffix(".png")
                rgba = rgba_from_mask(corrected_rgb, mask); atomic_save_png(rgba, target); output = target.relative_to(base).as_posix()
                if test:
                    preview = base / "previews" / relative.with_suffix(".png")
                    make_preview(rgb, rgba, preview, self.config.preview)
                    preview_items.append((preview, key, status.lower()))
                row.update(qc.as_dict()); row.update(width=rgb.width, height=rgb.height,
                    person_count=human.person_count if human else "", missing_person_count=human.missing_count if human else "",
                    edge_correction_p95=f"{edge_metrics.correction_p95:.4f}", edge_clipped_ratio=f"{edge_metrics.clipped_ratio:.4f}",
                    secondary_model_used=secondary_used, mask_iou=iou, alpha_difference=difference)
            except Exception as exc:
                row.update(error=f"{type(exc).__name__}: {exc}", review_reason="", review_details="",
                           width="", height="", foreground_ratio="", touch_top="", touch_bottom="", touch_left="", touch_right="")
            elapsed = time.perf_counter() - started; timings.append(elapsed); counts[status] += 1
            info = getattr(self.primary, "info", None)
            row.update(filename=relative.with_suffix(".png").as_posix(), output_file=output, status=status, processing_time=f"{elapsed:.4f}",
                       device=getattr(info, "name", "test"), precision=getattr(info, "precision", "test"), error=row.get("error", ""))
            rows.append(row); write_csv_atomic(report_path, rows)
            if callback: callback(Progress(len(rows), len(files), key, dict(counts), statistics.mean(timings)))
            if self.stop_event.is_set(): break
        if test:
            make_contact_sheet(preview_items, base / "contact_sheet" / "contact_sheet.png", self.config.preview)
        total_time = sum(timings)
        try:
            import torch
            peak_vram = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        except Exception: peak_vram = 0
        summary = {"total": len(rows), "ready": counts["READY"], "review": counts["REVIEW"],
            "failed": counts["FAILED"], "stopped": self.stop_event.is_set(),
            "total_processing_time": total_time, "average_processing_time": statistics.mean(timings) if timings else 0,
            "median_processing_time": statistics.median(timings) if timings else 0, "peak_vram_gb": peak_vram,
            "p95_processing_time": float(np.percentile(timings, 95)) if timings else 0,
            "model": self.config.model.primary, "device": getattr(getattr(self.primary, "info", None), "name", "test"),
            "precision": getattr(getattr(self.primary, "info", None), "precision", "test")}
        write_json_atomic(summary_path, summary)
        return summary
