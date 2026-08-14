from __future__ import annotations

import argparse
import json
from pathlib import Path

from bgremover.batch import BatchProcessor
from bgremover.config import load_config
from bgremover.inference import BiRefNetBackend
from bgremover.inference import detect_device
from bgremover.verification import TorchvisionPersonVerifier

parser = argparse.ArgumentParser(description="BiRefNet technical batch benchmark")
parser.add_argument("source", type=Path)
parser.add_argument("destination", type=Path)
parser.add_argument("--count", type=int, default=100)
args = parser.parse_args()
cfg = load_config()
info = detect_device(cfg.model.precision)
verifier = TorchvisionPersonVerifier(cfg.qc, "cuda" if info.available else "cpu") if cfg.verification.enabled else None
summary = BatchProcessor(cfg, BiRefNetBackend(cfg.model), verifier=verifier).run(
    args.source, args.destination, test=True, sample_size=args.count, resume=False,
    callback=lambda p: print(f"{p.processed}/{p.total}: {p.current}", flush=True),
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
