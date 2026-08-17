from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from .config import ArtifactConfig


@dataclass
class ArtifactResult:
    residual_background_score: float = 0.0
    chromatic_background_score: float = 0.0
    neutral_background_score: float = 0.0
    loose_edge_score: float = 0.0
    compact_damage_score: float = 0.0
    weak_hair_edge_score: float = 0.0
    local_alpha_instability_score: float = 0.0
    semantic_edge_anomaly_score: float = 0.0
    local_agreement_score: float = 0.0
    total_score: float = 0.0
    edge_pixels: int = 0
    severity: str = "none"
    flags: list[str] = field(default_factory=list)
    hard_reasons: list[str] = field(default_factory=list)
    hard_details: list[str] = field(default_factory=list)
    weak_triggers: list[str] = field(default_factory=list)
    telemetry_signals: list[str] = field(default_factory=list)
    heatmap: np.ndarray | None = field(default=None, repr=False, compare=False)

    def as_dict(self) -> dict[str, object]:
        return dict(
            artifact_residual_background_score=f"{self.residual_background_score:.4f}",
            artifact_chromatic_background_score=f"{self.chromatic_background_score:.4f}",
            artifact_neutral_background_score=f"{self.neutral_background_score:.4f}",
            artifact_loose_edge_score=f"{self.loose_edge_score:.4f}",
            artifact_compact_damage_score=f"{self.compact_damage_score:.4f}",
            artifact_weak_hair_edge_score=f"{self.weak_hair_edge_score:.4f}",
            artifact_local_alpha_instability_score=f"{self.local_alpha_instability_score:.4f}",
            artifact_semantic_edge_anomaly_score=f"{self.semantic_edge_anomaly_score:.4f}",
            artifact_local_agreement_score=f"{self.local_agreement_score:.4f}",
            artifact_total_score=f"{self.total_score:.4f}",
            artifact_edge_pixels=self.edge_pixels,
            artifact_severity=self.severity,
            artifact_flags=";".join(self.flags),
            artifact_hard_reasons=";".join(self.hard_reasons),
            artifact_hard_details="; ".join(self.hard_details),
            artifact_weak_triggers=";".join(self.weak_triggers),
            artifact_telemetry=";".join(self.telemetry_signals),
        )


def _resize_for_analysis(original: Image.Image, rgba: Image.Image, maximum: int):
    width, height = original.size
    scale = min(1.0, maximum / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if size != original.size:
        original = original.resize(size, Image.Resampling.LANCZOS)
        rgba = rgba.resize(size, Image.Resampling.LANCZOS)
    return (
        np.asarray(original.convert("RGB"), dtype=np.float32) / 255.0,
        np.asarray(rgba.convert("RGBA"), dtype=np.float32) / 255.0,
    )


def _normal_references(rgb: np.ndarray, foreground: np.ndarray, alpha: np.ndarray, radius: int):
    gx = cv2.Sobel(alpha, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(alpha, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    nx = gx / np.maximum(magnitude, 1e-6)
    ny = gy / np.maximum(magnitude, 1e-6)
    transition = (alpha > 0.035) & (alpha < 0.965) & (magnitude > 0.015)
    yy, xx = np.indices(alpha.shape)

    inner = foreground.copy()
    outer = rgb.copy()
    far_outer = rgb.copy()
    inner_distance = np.zeros(alpha.shape, np.float32)
    outer_distance = np.zeros(alpha.shape, np.float32)
    found_inner = np.zeros(alpha.shape, bool)
    found_outer = np.zeros(alpha.shape, bool)

    for step in range(1, radius + 1):
        ix = np.clip(np.rint(xx + nx * step).astype(np.int32), 0, alpha.shape[1] - 1)
        iy = np.clip(np.rint(yy + ny * step).astype(np.int32), 0, alpha.shape[0] - 1)
        ox = np.clip(np.rint(xx - nx * step).astype(np.int32), 0, alpha.shape[1] - 1)
        oy = np.clip(np.rint(yy - ny * step).astype(np.int32), 0, alpha.shape[0] - 1)
        take_inner = transition & ~found_inner & (alpha[iy, ix] >= 0.96)
        take_outer = transition & ~found_outer & (alpha[oy, ox] <= 0.03)
        inner[take_inner] = foreground[iy[take_inner], ix[take_inner]]
        outer[take_outer] = rgb[oy[take_outer], ox[take_outer]]
        inner_distance[take_inner] = step
        outer_distance[take_outer] = step
        found_inner |= take_inner
        found_outer |= take_outer

    far_step = radius + max(3, radius // 3)
    fx = np.clip(np.rint(xx - nx * far_step).astype(np.int32), 0, alpha.shape[1] - 1)
    fy = np.clip(np.rint(yy - ny * far_step).astype(np.int32), 0, alpha.shape[0] - 1)
    far_outer[transition] = rgb[fy[transition], fx[transition]]
    valid = transition & found_inner & found_outer
    return valid, inner, outer, far_outer, inner_distance, outer_distance, magnitude


def _local_texture(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mean = cv2.boxFilter(gray, cv2.CV_32F, (5, 5))
    mean_sq = cv2.boxFilter(gray * gray, cv2.CV_32F, (5, 5))
    return np.sqrt(np.maximum(0.0, mean_sq - mean * mean))


def _loose_edge_score(alpha: np.ndarray, valid: np.ndarray, inner_distance: np.ndarray,
                      outer_distance: np.ndarray) -> tuple[float, np.ndarray]:
    """Return a robust local transition-width anomaly at the 640px analysis scale."""

    stable = valid & (alpha >= 0.08) & (alpha <= 0.88)
    widths = inner_distance + outer_distance
    if not np.any(stable):
        return 0.0, np.zeros_like(alpha, np.float32)
    scale = min(1.0, max(alpha.shape) / 640.0)
    normal_width = max(3.0, 7.0 * scale)
    anomaly_span = max(6.0, 12.0 * scale)
    # At 640px, seven pixels cover antialiasing and fine natural strands while
    # a twelve-pixel excess spans the calibrated weak-to-severe range.
    p90 = float(np.percentile(widths[stable], 90))
    score = float(np.clip((p90 - normal_width) / anomaly_span, 0.0, 1.0))
    heat = np.where(stable, np.clip((widths - normal_width) / anomaly_span, 0.0, 1.0), 0.0)
    return score, heat.astype(np.float32)


def _alpha_instability(alpha: np.ndarray, valid: np.ndarray) -> tuple[float, np.ndarray]:
    low = ((alpha > 0.02) & (alpha < 0.35)).astype(np.uint8)
    core_near = cv2.dilate((alpha >= 0.5).astype(np.uint8), np.ones((5, 5), np.uint8))
    isolated = low & (core_near == 0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(isolated, 8)
    speckles = np.zeros_like(isolated, bool)
    max_component = max(4, round(alpha.size * 0.00025))
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if 1 <= area <= max_component:
            speckles |= labels == index
    ratio = float(speckles.sum() / max(1, valid.sum()))
    return float(min(1.0, ratio / 0.20)), speckles.astype(np.float32)


def analyze_artifacts(
    original: Image.Image,
    rgba: Image.Image,
    cfg: ArtifactConfig,
    *,
    semantic_relevant: bool | None = None,
) -> ArtifactResult:
    """Measure visible cutout defects without changing foreground RGB or alpha.

    The normal-direction comparison is background-colour agnostic.  Its colour
    residual is also the difference exposed by white, gray, black, and contrast
    composites: the canvas term cancels, while alpha scales visibility.
    """

    if not cfg.enabled:
        return ArtifactResult()
    if original.size != rgba.size:
        raise ValueError("Artifact analysis requires original and RGBA at the same size")

    rgb, cutout = _resize_for_analysis(original, rgba, cfg.analysis_max_size)
    foreground, alpha = cutout[..., :3], cutout[..., 3]
    valid, inner, outer, far_outer, inner_d, outer_d, magnitude = _normal_references(
        rgb, foreground, alpha, cfg.normal_search_radius
    )
    edge_pixels = int(valid.sum())
    result = ArtifactResult(edge_pixels=edge_pixels)
    if edge_pixels < cfg.min_edge_pixels:
        result.telemetry_signals.append("artifact_edge_sample_too_small")
        return result

    delta = foreground - inner
    background_direction = outer - inner
    delta_chroma = delta - delta.mean(axis=2, keepdims=True)
    background_chroma = background_direction - background_direction.mean(axis=2, keepdims=True)
    bg_norm = np.linalg.norm(background_chroma, axis=2)
    delta_norm = np.linalg.norm(delta_chroma, axis=2)
    projection = np.maximum(
        0.0, np.sum(delta_chroma * background_chroma, axis=2) / np.maximum(bg_norm, 1e-5)
    )
    cosine = np.sum(delta_chroma * background_chroma, axis=2) / np.maximum(
        delta_norm * bg_norm, 1e-5
    )
    hsv = cv2.cvtColor(np.clip(outer, 0, 1), cv2.COLOR_RGB2HSV)
    composite_visibility = projection * alpha
    stable = valid & (alpha >= 0.08) & (alpha <= 0.88)
    suspicious = stable & (hsv[..., 1] > 0.18) & (cosine > 0.45) & (composite_visibility > 0.012)
    saturated_ratio = float(suspicious.sum() / max(1, stable.sum()))
    chromatic_score = min(1.0, saturated_ratio / 0.08)
    edge_luma = foreground.mean(axis=2)
    inner_luma = inner.mean(axis=2)
    outer_luma = outer.mean(axis=2)
    luma_delta = edge_luma - inner_luma
    luma_direction = outer_luma - inner_luma
    neutral_suspicious = (
        stable & (hsv[..., 1] <= 0.18)
        & (luma_delta * luma_direction > 0.0)
        & (np.abs(luma_delta) * alpha > 0.045)
        & (np.abs(luma_direction) > 0.08)
    )
    neutral_score = min(1.0, float(neutral_suspicious.sum() / max(1, stable.sum())) / 0.08)
    residual_score = max(chromatic_score, neutral_score)

    texture = _local_texture(rgb)
    loose_score, loose_heat = _loose_edge_score(alpha, valid, inner_d, outer_d)
    instability_score, instability_heat = _alpha_instability(alpha, valid)

    # A low-alpha pixel just outside a compact edge is suspicious when it still
    # resembles the inward foreground much more than the farther background.
    outer_to_inner = np.linalg.norm(outer - inner, axis=2)
    outer_to_far = np.linalg.norm(outer - far_outer, axis=2)
    omitted_evidence = np.clip((outer_to_far - outer_to_inner - 0.025) / 0.20, 0.0, 1.0)
    coherent = np.clip((0.12 - texture) / 0.09, 0.0, 1.0)
    compact_values = omitted_evidence[valid] * coherent[valid]
    hair_values = omitted_evidence[valid] * np.clip(texture[valid] / 0.12, 0.0, 1.0)
    compact_score = float(np.percentile(compact_values, 97)) if compact_values.size else 0.0
    hair_score = float(np.percentile(hair_values, 97)) if hair_values.size else 0.0

    semantic_score = float(np.sqrt(chromatic_score * loose_score))
    local_width_threshold = max(6.0, 12.0 * min(1.0, max(alpha.shape) / 640.0))
    locally_loose = stable & ((inner_d + outer_d) >= local_width_threshold)
    compact_joint = suspicious & locally_loose & (texture < 0.045)
    compact_joint_ratio = float(compact_joint.sum() / max(1, suspicious.sum()))
    local_agreement_score = min(1.0, compact_joint_ratio / 0.02)
    total_score = float(max(
        semantic_score,
        residual_score * 0.72,
        loose_score * 0.72,
        compact_score * 0.60,
        hair_score * 0.55,
        instability_score * 0.60,
    ))
    result.residual_background_score = residual_score
    result.chromatic_background_score = chromatic_score
    result.neutral_background_score = neutral_score
    result.loose_edge_score = loose_score
    result.compact_damage_score = compact_score
    result.weak_hair_edge_score = hair_score
    result.local_alpha_instability_score = instability_score
    result.semantic_edge_anomaly_score = semantic_score
    result.local_agreement_score = local_agreement_score
    result.total_score = total_score

    thresholds = (
        ("residual_background_color", residual_score, cfg.residual_weak_score),
        ("loose_edge_noise", loose_score, cfg.loose_edge_weak_score),
        ("overcut_compact_structure", compact_score, cfg.local_signal_weak_score),
        ("weak_hair_edge", hair_score, cfg.local_signal_weak_score),
        ("local_alpha_instability", instability_score, cfg.local_signal_weak_score),
    )
    result.flags = [name for name, score, threshold in thresholds if score >= threshold]
    if residual_score >= cfg.residual_weak_score and loose_score >= cfg.loose_edge_weak_score * 0.8:
        result.flags.append("semantic_edge_anomaly")
    result.flags = list(dict.fromkeys(result.flags))

    localized_hard = (
        semantic_relevant is not False
        and chromatic_score >= cfg.hard_residual_score
        and chromatic_score <= cfg.hard_residual_max_score
        and loose_score >= cfg.hard_loose_edge_score
        and loose_score <= cfg.hard_loose_edge_max_score
        and semantic_score >= cfg.hard_combined_score
        and semantic_score <= cfg.hard_combined_max_score
        and local_agreement_score >= cfg.hard_local_agreement_score
        and local_agreement_score <= cfg.hard_local_agreement_max_score
    )
    structural_hard = (
        semantic_relevant is not False
        and compact_score >= cfg.hard_structural_score
        and instability_score >= cfg.hard_structural_score
    )
    if localized_hard or structural_hard:
        result.severity = "hard"
        result.hard_reasons.append("edge_artifact")
        if localized_hard:
            result.hard_details.append(
                "background-colour residual and locally loose edge agree "
                f"(residual={chromatic_score:.3f}, loose={loose_score:.3f})"
            )
        else:
            result.hard_details.append(
                "compact edge damage and local alpha instability agree "
                f"(compact={compact_score:.3f}, alpha={instability_score:.3f})"
            )
    elif any(flag in {
        "loose_edge_noise", "local_alpha_instability", "semantic_edge_anomaly",
    } for flag in result.flags):
        result.severity = "weak"
        result.weak_triggers.extend(
            flag for flag in result.flags
            if flag in {"loose_edge_noise", "local_alpha_instability", "semantic_edge_anomaly"}
        )
    elif result.flags or total_score >= 0.15:
        result.severity = "telemetry"
        result.telemetry_signals.append("artifact_signal_observed")

    if semantic_relevant is False and result.flags:
        result.telemetry_signals.append("artifact_semantic_suppression")
    neutral_visibility = np.abs(luma_delta) * alpha
    residual_heat = np.where(
        valid, np.clip(np.maximum(composite_visibility, neutral_visibility) / 0.12, 0.0, 1.0), 0.0
    )
    result.heatmap = np.maximum.reduce((residual_heat, loose_heat, instability_heat)).astype(np.float32)
    return result
