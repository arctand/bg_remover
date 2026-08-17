from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bgremover.artifacts import _local_texture, _normal_references, _resize_for_analysis, analyze_artifacts
from bgremover.config import ArtifactConfig
from bgremover.config import PreviewConfig
from bgremover.previews import make_preview


def main() -> None:
    parser = argparse.ArgumentParser(description="Score saved RC2 cutouts with the artifact layer")
    parser.add_argument("originals", type=Path)
    parser.add_argument("run", type=Path, help="Run root containing report.csv and ready/review")
    parser.add_argument("--summary", action="store_true", help="Print only severity counts and hard cases")
    parser.add_argument("--previews", type=Path, help="Write six-panel previews for hard artifact cases")
    args = parser.parse_args()
    with (args.run / "report.csv").open(encoding="utf-8-sig", newline="") as handle:
        report = {row["source_file"]: row for row in csv.DictReader(handle)}
    fields = (
        "case", "residual", "loose", "compact", "hair", "alpha", "combined", "total",
        "severity", "flags", "local_agreement", "person_count", "sat_ratio", "chroma_ratio", "visibility_p99",
        "normal_width_p90", "joint_ratio", "joint_of_residual", "compact_joint",
    )
    writer = csv.writer(sys.stdout, lineterminator="\n")
    if not args.summary:
        writer.writerow(fields)
    counts: Counter[str] = Counter()
    hard_cases: list[str] = []
    hard_metrics: list[str] = []
    for original_path in sorted(args.originals.glob("*")):
        if original_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        row = report.get(original_path.name, {})
        cutout_path = next(
            (args.run / folder / f"{original_path.stem}.png" for folder in ("ready", "review")
            if (args.run / folder / f"{original_path.stem}.png").exists()),
            None,
        )
        if cutout_path is None:
            continue
        with Image.open(original_path) as original, Image.open(cutout_path) as cutout:
            analysis_rgb, analysis_cutout = _resize_for_analysis(original.convert("RGB"), cutout.convert("RGBA"), 640)
            analysis_fg, analysis_alpha = analysis_cutout[..., :3], analysis_cutout[..., 3]
            valid, inner, outer, _, inner_d, outer_d, _ = _normal_references(
                analysis_rgb, analysis_fg, analysis_alpha, 16
            )
            result = analyze_artifacts(
                original.convert("RGB"), cutout.convert("RGBA"), ArtifactConfig(),
                semantic_relevant=int(row.get("person_count") or 0) > 0,
            )
        output_row = (
            original_path.stem,
            f"{result.residual_background_score:.3f}",
            f"{result.loose_edge_score:.3f}",
            f"{result.compact_damage_score:.3f}",
            f"{result.weak_hair_edge_score:.3f}",
            f"{result.local_alpha_instability_score:.3f}",
            f"{result.semantic_edge_anomaly_score:.3f}",
            f"{result.total_score:.3f}", result.severity, "|".join(result.flags),
            f"{result.local_agreement_score:.3f}",
            row.get("person_count", ""),
            *_diagnostics(analysis_rgb, analysis_fg, analysis_alpha, valid, inner, outer, inner_d, outer_d),
        )
        counts[result.severity] += 1
        if result.severity == "hard":
            hard_cases.append(original_path.name)
            hard_metrics.append(
                f"{original_path.stem}:r={result.residual_background_score:.2f},"
                f"l={result.loose_edge_score:.2f},a={result.local_agreement_score:.2f}"
            )
            if args.previews:
                with Image.open(original_path) as original, Image.open(cutout_path) as cutout:
                    make_preview(
                        original.convert("RGB"), cutout.convert("RGBA"),
                        args.previews / f"{original_path.stem}.png", PreviewConfig(), result,
                    )
        if not args.summary:
            writer.writerow(output_row)
    if args.summary:
        writer.writerow(("total", sum(counts.values())))
        for severity in ("hard", "weak", "telemetry", "none"):
            writer.writerow((severity, counts[severity]))
        writer.writerow(("hard_cases", "|".join(hard_cases)))
        writer.writerow(("hard_metrics", "|".join(hard_metrics)))


def _diagnostics(rgb, foreground, alpha, valid, inner, outer, inner_d, outer_d):
    import cv2
    import numpy as np

    delta = foreground - inner
    direction = outer - inner
    direction_norm = np.linalg.norm(direction, axis=2)
    delta_norm = np.linalg.norm(delta, axis=2)
    projection = np.maximum(0.0, np.sum(delta * direction, axis=2) / np.maximum(direction_norm, 1e-5))
    cosine = np.sum(delta * direction, axis=2) / np.maximum(delta_norm * direction_norm, 1e-5)
    saturation = cv2.cvtColor(np.clip(outer, 0, 1), cv2.COLOR_RGB2HSV)[..., 1]
    visibility = projection * alpha
    stable = valid & (alpha >= 0.08) & (alpha <= 0.88)
    suspicious = stable & (saturation > 0.18) & (cosine > 0.45) & (visibility > 0.012)
    delta_chroma = delta - delta.mean(axis=2, keepdims=True)
    direction_chroma = direction - direction.mean(axis=2, keepdims=True)
    chroma_direction_norm = np.linalg.norm(direction_chroma, axis=2)
    chroma_delta_norm = np.linalg.norm(delta_chroma, axis=2)
    chroma_projection = np.maximum(
        0.0, np.sum(delta_chroma * direction_chroma, axis=2) / np.maximum(chroma_direction_norm, 1e-5)
    )
    chroma_cosine = np.sum(delta_chroma * direction_chroma, axis=2) / np.maximum(
        chroma_delta_norm * chroma_direction_norm, 1e-5
    )
    chroma_suspicious = stable & (saturation > 0.18) & (chroma_cosine > 0.45) & (chroma_projection * alpha > 0.012)
    locally_loose = stable & ((inner_d + outer_d) >= 12)
    joint = chroma_suspicious & locally_loose
    compact_joint = joint & (_local_texture(rgb) < 0.045)
    return (
        f"{suspicious.sum() / max(1, stable.sum()):.4f}",
        f"{chroma_suspicious.sum() / max(1, stable.sum()):.4f}",
        f"{np.percentile(visibility[stable], 99) if stable.any() else 0:.4f}",
        f"{np.percentile((inner_d + outer_d)[stable], 90) if stable.any() else 0:.3f}",
        f"{joint.sum() / max(1, stable.sum()):.4f}",
        f"{joint.sum() / max(1, chroma_suspicious.sum()):.4f}",
        f"{compact_joint.sum() / max(1, chroma_suspicious.sum()):.4f}",
    )


if __name__ == "__main__":
    main()
