from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from bgremover.batch import BatchProcessor
from bgremover.config import AppConfig
from bgremover.foreground import ForegroundResult
from bgremover.qc import analyze_mask
from bgremover.reporting import read_completed
from bgremover.verification import HumanVerification, SAMVerification


class FakeBackend:
    info = type("Info", (), {"name": "test", "precision": "fp32"})()
    def predict(self, image):
        a = np.zeros((image.height, image.width), np.uint8)
        a[image.height//4:3*image.height//4, image.width//4:3*image.width//4] = 255
        return Image.fromarray(a)
    def clear_cache(self): pass


class FakeRefiner:
    def warm_up(self): pass
    def refine(self, image, alpha):
        return ForegroundResult(image.convert("RGB").copy(), alpha.convert("L").copy())


class FakePersonVerifier:
    def __init__(self, result): self.result, self.model = result, object()
    def verify(self, image, alpha): return self.result


class FakeStrongVerifier:
    def __init__(self, result=None):
        self.result = result or SAMVerification(ran=True, prompted_boxes=1, checked_people=1)
        self.calls = 0
    def verify(self, image, alpha, human):
        self.calls += 1
        return self.result


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def test_naming_rgba_resolution_source_safety_and_unicode(tmp_path):
    source = tmp_path / "Фото клиента"; nested = source / "Событие 1"; nested.mkdir(parents=True)
    original = nested / "фото 01.jpg"; Image.new("RGB", (73, 51), "red").save(original); before = digest(original)
    summary = BatchProcessor(AppConfig(), FakeBackend()).run(source, tmp_path / "Результат")
    output = tmp_path / "Результат" / "ready" / "Событие 1" / "фото 01.png"
    assert output.exists() and digest(original) == before and summary["ready"] == 1
    with Image.open(output) as result: assert result.mode == "RGBA" and result.size == (73, 51)


def test_broken_file_does_not_stop_and_resume_skips(tmp_path):
    source = tmp_path / "in"; source.mkdir(); Image.new("RGB", (40, 40)).save(source / "ok.jpg"); (source / "bad.jpg").write_bytes(b"broken")
    out = tmp_path / "out"; backend = FakeBackend(); first = BatchProcessor(AppConfig(), backend).run(source, out)
    assert first["total"] == 2 and first["failed"] == 1
    class FailIfCalled(FakeBackend):
        def predict(self, image): raise AssertionError("resume repeated a completed image")
    second = BatchProcessor(AppConfig(), FailIfCalled()).run(source, out)
    assert second["total"] == 2


def test_duplicate_names_keep_relative_structure(tmp_path):
    source = tmp_path / "in"
    for folder in ("a", "b"):
        (source/folder).mkdir(parents=True); Image.new("RGB", (20, 20)).save(source/folder/"same.jpg")
    BatchProcessor(AppConfig(), FakeBackend()).run(source, tmp_path/"out")
    assert (tmp_path/"out/ready/a/same.png").exists()
    assert (tmp_path/"out/ready/b/same.png").exists()


def test_bad_mask_routes_to_review(tmp_path):
    class BadEdge(FakeBackend):
        def predict(self, image):
            a=np.full((image.height,image.width),255,np.uint8); return Image.fromarray(a)
    source=tmp_path/"in"; source.mkdir(); Image.new("RGB",(100,100)).save(source/"x.jpg")
    BatchProcessor(AppConfig(),BadEdge()).run(source,tmp_path/"out")
    assert (tmp_path/"out/review/x.png").exists()


def test_test_mode_creates_preview_contact_and_report(tmp_path):
    source=tmp_path/"in"; source.mkdir()
    for i in range(3): Image.new("RGB",(30,30)).save(source/f"{i}.jpg")
    BatchProcessor(AppConfig(),FakeBackend()).run(source,tmp_path/"out",test=True,sample_size=2)
    base=tmp_path/"out/debug_output"
    assert len(list((base/"previews").glob("*.png"))) == 2
    assert (base/"contact_sheet/contact_sheet.png").exists() and (base/"report.csv").exists()


def test_graceful_stop_then_resume(tmp_path):
    source=tmp_path/"in"; source.mkdir()
    for i in range(4): Image.new("RGB",(30,30)).save(source/f"{i}.jpg")
    out=tmp_path/"out"; processor=BatchProcessor(AppConfig(),FakeBackend())
    first=processor.run(source,out,callback=lambda p: processor.stop() if p.processed == 1 else None)
    assert first["stopped"] and first["total"] == 1
    second=BatchProcessor(AppConfig(),FakeBackend()).run(source,out)
    assert not second["stopped"] and second["total"] == 4


def test_cropped_source_telemetry_alone_routes_ready(tmp_path):
    class SideContact(FakeBackend):
        def predict(self, image):
            alpha = np.zeros((image.height, image.width), np.uint8)
            alpha[20:80, :40] = 255
            return Image.fromarray(alpha)

    source = tmp_path / "in"; source.mkdir(); Image.new("RGB", (100, 100)).save(source / "x.jpg")
    out = tmp_path / "out"
    summary = BatchProcessor(AppConfig(), SideContact(), foreground_refiner=FakeRefiner()).run(source, out)
    row = read_completed(out / "report.csv")["x.jpg"]
    assert summary["ready"] == 1 and row["cropped_source_signal"] == "True"
    assert row["review_reason"] == ""


def test_zero_person_detections_not_failed_and_sam_not_called(tmp_path):
    source = tmp_path / "in"; source.mkdir(); Image.new("RGB", (80, 80)).save(source / "x.jpg")
    strong = FakeStrongVerifier()
    summary = BatchProcessor(
        AppConfig(), FakeBackend(), verifier=FakePersonVerifier(HumanVerification()),
        foreground_refiner=FakeRefiner(), strong_verifier=strong,
    ).run(source, tmp_path / "out")
    assert summary["failed"] == 0 and summary["ready"] == 1 and strong.calls == 0


def test_clean_case_does_not_call_sam(tmp_path):
    source = tmp_path / "in"; source.mkdir(); Image.new("RGB", (80, 80)).save(source / "x.jpg")
    human = HumanVerification(person_count=1, boxes=[(10, 10, 70, 70)], scores=[0.99], coverages=[0.75], center_coverages=[0.9])
    strong = FakeStrongVerifier()
    BatchProcessor(
        AppConfig(), FakeBackend(), verifier=FakePersonVerifier(human),
        foreground_refiner=FakeRefiner(), strong_verifier=strong,
    ).run(source, tmp_path / "out")
    assert strong.calls == 0


def test_strong_semantic_disagreement_calls_sam_and_routes_review(tmp_path):
    source = tmp_path / "in"; source.mkdir(); Image.new("RGB", (80, 80)).save(source / "x.jpg")
    human = HumanVerification(person_count=1, boxes=[(5, 5, 75, 75)], scores=[0.99], coverages=[0.1], center_coverages=[0.2])
    semantic = SAMVerification(
        ran=True, prompted_boxes=1, checked_people=1, min_recall=0.4, min_iou=0.3,
        missing_count=1, reasons=["missing_body_part", "human_mask_disagreement"],
    )
    strong = FakeStrongVerifier(semantic)
    out = tmp_path / "out"
    summary = BatchProcessor(
        AppConfig(), FakeBackend(), verifier=FakePersonVerifier(human),
        foreground_refiner=FakeRefiner(), strong_verifier=strong,
    ).run(source, out)
    row = read_completed(out / "report.csv")["x.jpg"]
    assert strong.calls == 1 and summary["review"] == 1
    assert "human_mask_disagreement" in row["review_reason"]


def test_multiple_people_disagreement_routes_review(tmp_path):
    source = tmp_path / "in"; source.mkdir(); Image.new("RGB", (80, 80)).save(source / "x.jpg")
    human = HumanVerification(
        person_count=2, boxes=[(5, 5, 35, 75), (40, 5, 75, 75)], scores=[0.99, 0.95],
        coverages=[0.58, 0.8], center_coverages=[0.7, 0.9],
    )
    semantic = SAMVerification(
        ran=True, prompted_boxes=2, checked_people=2, missing_count=1,
        reasons=["missing_body_part", "human_mask_disagreement", "multiple_people_uncertain"],
    )
    out = tmp_path / "out"
    BatchProcessor(
        AppConfig(), FakeBackend(), verifier=FakePersonVerifier(human),
        foreground_refiner=FakeRefiner(), strong_verifier=FakeStrongVerifier(semantic),
    ).run(source, out)
    row = read_completed(out / "report.csv")["x.jpg"]
    assert "multiple_people_uncertain" in row["review_reason"]
