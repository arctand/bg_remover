from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from bgremover.artifacts import ArtifactResult, analyze_artifacts
from bgremover.batch import BatchProcessor
from bgremover.config import AppConfig, ArtifactConfig
from bgremover.foreground import ForegroundResult
from bgremover.reporting import read_completed


def _cutout(*, contaminated: bool, blur: float,
            background=(24, 76, 224), subject=(184, 154, 116)) -> tuple[Image.Image, Image.Image]:
    height = width = 192
    background = np.array(background, np.float32)
    subject = np.array(subject, np.float32)
    core = np.zeros((height, width), np.uint8)
    cv2.ellipse(core, (96, 101), (54, 72), 0, 0, 360, 255, -1)
    original = np.broadcast_to(background, (height, width, 3)).copy()
    original[core > 0] = subject
    alpha = cv2.GaussianBlur(core, (0, 0), blur) if blur else core
    foreground = np.broadcast_to(subject, original.shape).copy()
    if contaminated:
        amount = np.clip(1.0 - alpha[..., None] / 255.0, 0.0, 1.0) ** 0.45
        foreground = foreground * (1.0 - amount) + background * amount
    rgba = np.dstack((foreground.clip(0, 255).astype(np.uint8), alpha))
    return Image.fromarray(original.astype(np.uint8)), Image.fromarray(rgba)


def test_residual_background_colour_is_detected_without_red_specific_rule():
    original, rgba = _cutout(contaminated=True, blur=4.0)
    result = analyze_artifacts(original, rgba, ArtifactConfig())
    assert result.residual_background_score >= 0.35
    assert "residual_background_color" in result.flags


def test_neutral_background_residual_has_separate_telemetry_score():
    original, rgba = _cutout(
        contaminated=True, blur=4.0, background=(210, 210, 210), subject=(65, 65, 65)
    )
    result = analyze_artifacts(original, rgba, ArtifactConfig())
    assert result.neutral_background_score >= 0.35
    assert "residual_background_color" in result.flags


def test_clean_solid_edge_does_not_raise_artifact_flag():
    original, rgba = _cutout(contaminated=False, blur=0.55)
    result = analyze_artifacts(original, rgba, ArtifactConfig())
    assert result.severity in {"none", "telemetry"}
    assert not result.flags and not result.hard_reasons


def test_loose_edge_noise_is_distinct_from_clean_solid_edge():
    original, rgba = _cutout(contaminated=False, blur=5.0)
    result = analyze_artifacts(original, rgba, ArtifactConfig())
    assert result.loose_edge_score >= ArtifactConfig().loose_edge_weak_score
    assert "loose_edge_noise" in result.flags


def test_semantic_safety_suppresses_hard_decision_but_keeps_evidence():
    original, rgba = _cutout(contaminated=True, blur=4.0)
    cfg = ArtifactConfig(hard_loose_edge_score=0.0, hard_combined_score=0.0)
    result = analyze_artifacts(original, rgba, cfg, semantic_relevant=False)
    assert result.flags
    assert result.severity != "hard" and not result.hard_reasons
    assert "artifact_semantic_suppression" in result.telemetry_signals


class _Backend:
    info = type("Info", (), {"name": "test", "precision": "fp32"})()

    def predict(self, image):
        alpha = np.zeros((image.height, image.width), np.uint8)
        alpha[16:-16, 16:-16] = 255
        return Image.fromarray(alpha)


class _Refiner:
    def warm_up(self):
        pass

    def refine(self, image, alpha):
        return ForegroundResult(image.convert("RGB"), alpha.convert("L"))


def _run_with_artifact(monkeypatch, tmp_path, artifact: ArtifactResult):
    monkeypatch.setattr("bgremover.batch.analyze_artifacts", lambda *args, **kwargs: artifact)
    source = tmp_path / "input"
    source.mkdir()
    Image.new("RGB", (96, 96), "gray").save(source / "case.jpg")
    output = tmp_path / "output"
    summary = BatchProcessor(AppConfig(), _Backend(), foreground_refiner=_Refiner()).run(source, output)
    return summary, read_completed(output / "report.csv")["case.jpg"]


def test_hard_artifact_routes_to_review(monkeypatch, tmp_path):
    artifact = ArtifactResult(
        severity="hard", flags=["semantic_edge_anomaly"], hard_reasons=["edge_artifact"],
        hard_details=["synthetic hard artifact"], total_score=0.9,
    )
    summary, row = _run_with_artifact(monkeypatch, tmp_path, artifact)
    assert summary["review"] == 1 and row["status"] == "REVIEW"
    assert "edge_artifact" in row["final_review_reasons"]


def test_weak_artifact_is_reported_without_forcing_review(monkeypatch, tmp_path):
    artifact = ArtifactResult(
        severity="weak", flags=["loose_edge_noise"], weak_triggers=["loose_edge_noise"],
        total_score=0.5,
    )
    summary, row = _run_with_artifact(monkeypatch, tmp_path, artifact)
    assert summary["ready"] == 1 and row["status"] == "READY"
    assert row["artifact_weak_triggers"] == "loose_edge_noise"
    assert "artifact:loose_edge_noise" in row["telemetry_signals"]
