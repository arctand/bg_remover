from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from bgremover.batch import BatchProcessor
from bgremover.config import AppConfig
from bgremover.qc import analyze_mask


class FakeBackend:
    info = type("Info", (), {"name": "test", "precision": "fp32"})()
    def predict(self, image):
        a = np.zeros((image.height, image.width), np.uint8)
        a[image.height//4:3*image.height//4, image.width//4:3*image.width//4] = 255
        return Image.fromarray(a)
    def clear_cache(self): pass


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
