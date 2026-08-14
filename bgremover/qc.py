from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .config import QCConfig
from .edge import EdgeMetrics
from .verification import HumanVerification


@dataclass
class QCResult:
    width: int
    height: int
    foreground_area: int
    foreground_ratio: float
    components: int
    touch_top: bool
    touch_bottom: bool
    touch_left: bool
    touch_right: bool
    edge_top_ratio: float
    edge_bottom_ratio: float
    edge_left_ratio: float
    edge_right_ratio: float
    translucent_ratio: float
    hole_ratio: float
    cropped_source_signal: bool = False
    hard_reasons: list[str] = field(default_factory=list)
    hard_details: list[str] = field(default_factory=list)
    verification_triggers: list[str] = field(default_factory=list)
    trigger_details: list[str] = field(default_factory=list)
    telemetry_signals: list[str] = field(default_factory=list)

    def as_dict(self):
        data = asdict(self)
        data["fast_qc_hard_reasons"] = ";".join(self.hard_reasons)
        data["fast_qc_hard_details"] = "; ".join(self.hard_details)
        data["verification_triggers"] = ";".join(self.verification_triggers)
        data["verification_trigger_details"] = "; ".join(self.trigger_details)
        data["telemetry_signals"] = ";".join(self.telemetry_signals)
        # Legacy aliases remain for consumers that read fast-QC results directly.
        data["review_reason"] = ";".join(self.hard_reasons)
        data["review_details"] = "; ".join(self.hard_details)
        for key in (
            "hard_reasons", "hard_details", "verification_triggers",
            "trigger_details", "telemetry_signals",
        ):
            data.pop(key)
        return data

    @property
    def review_reasons(self):
        """Compatibility view: only independently decisive fast-QC reasons."""
        return self.hard_reasons

    @property
    def review_details(self):
        return self.hard_details

    @property
    def needs_review(self): return bool(self.hard_reasons)


def _edge_metrics(fg: np.ndarray, band: int, min_span: float):
    h, w = fg.shape
    strips = (fg[:band, :], fg[-band:, :], fg[:, :band], fg[:, -band:])
    areas = tuple(float(s.mean()) for s in strips)
    spans = (
        float(np.any(strips[0], axis=0).mean()), float(np.any(strips[1], axis=0).mean()),
        float(np.any(strips[2], axis=1).mean()), float(np.any(strips[3], axis=1).mean()),
    )
    touches = tuple(span >= min_span for span in spans)
    return areas, touches


def analyze_mask(alpha: np.ndarray, cfg: QCConfig, edge: EdgeMetrics | None = None,
                 human: HumanVerification | None = None) -> QCResult:
    if alpha.ndim != 2:
        raise ValueError("Alpha mask must be a 2D array")
    h, w = alpha.shape
    fg = alpha >= cfg.foreground_threshold
    area = int(fg.sum())
    ratio = area / max(1, h * w)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
    meaningful = sum(1 for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] / (h * w) >= cfg.min_component_ratio)
    band = max(2, int(min(h, w) * cfg.edge_band_ratio))
    edge_areas, touches = _edge_metrics(fg, band, cfg.edge_min_span_ratio)
    solid = alpha >= cfg.solid_threshold
    contours, hierarchy = cv2.findContours(solid.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0.0
    if hierarchy is not None:
        for contour, item in zip(contours, hierarchy[0]):
            if item[3] >= 0:
                holes += abs(cv2.contourArea(contour))
    hole_ratio = holes / max(1, area)
    translucent = ((alpha > cfg.foreground_threshold) & (alpha < cfg.solid_threshold)).sum() / max(1, area)
    hard_reasons: list[str] = []
    hard_details: list[str] = []
    triggers: list[str] = []
    trigger_details: list[str] = []
    if ratio < cfg.min_foreground_ratio:
        hard_reasons.append("mask_issue")
        hard_details.append("almost empty foreground mask")
    if ratio > cfg.max_foreground_ratio:
        hard_reasons.append("mask_issue")
        hard_details.append("foreground covers almost the entire image")
    if meaningful > cfg.max_components:
        hard_reasons.append("mask_issue")
        hard_details.append(f"{meaningful} significant disconnected components")
    if hole_ratio > cfg.max_hole_ratio:
        triggers.append("large_internal_holes")
        trigger_details.append("large internal holes in foreground")
    if translucent > cfg.max_translucent_ratio:
        triggers.append("high_translucency")
        trigger_details.append("unusually high translucent pixel ratio")
    # Source-crop heuristic: ignore wide bottom contact typical for waist/bust portraits.
    top_crop = touches[0] and edge_areas[0] >= cfg.edge_min_area_ratio and spans_for_crop(fg[:band, :], 0) >= cfg.cropped_top_span_ratio
    bottom_span = spans_for_crop(fg[-band:, :], 0)
    bottom_crop = touches[1] and edge_areas[1] >= cfg.edge_min_area_ratio and cfg.edge_min_span_ratio <= bottom_span <= cfg.cropped_bottom_span_ratio
    left_crop = touches[2] and edge_areas[2] >= cfg.edge_min_area_ratio and spans_for_crop(fg[:, :band], 1) >= cfg.cropped_side_span_ratio
    right_crop = touches[3] and edge_areas[3] >= cfg.edge_min_area_ratio and spans_for_crop(fg[:, -band:], 1) >= cfg.cropped_side_span_ratio
    cropped_source_signal = top_crop or bottom_crop or left_crop or right_crop
    # Frame contact and RGB correction magnitude are telemetry, not independent
    # evidence of a bad cutout. Semantic escalation is handled after fast QC.
    # Stable order without duplicate reason codes.
    hard_reasons = list(dict.fromkeys(hard_reasons))
    triggers = list(dict.fromkeys(triggers))
    telemetry = ["cropped_source_signal"] if cropped_source_signal else []
    return QCResult(w, h, area, ratio, meaningful, *touches, *edge_areas,
                    float(translucent), float(hole_ratio), cropped_source_signal,
                    hard_reasons, hard_details, triggers, trigger_details, telemetry)


def spans_for_crop(strip: np.ndarray, collapse_axis: int) -> float:
    return float(np.any(strip, axis=collapse_axis).mean())


def mask_similarity(a: np.ndarray, b: np.ndarray, threshold: int = 127) -> tuple[float, float]:
    af, bf = a >= threshold, b >= threshold
    union = np.logical_or(af, bf).sum()
    iou = float(np.logical_and(af, bf).sum() / max(1, union))
    difference = float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean() / 255.0)
    return iou, difference
