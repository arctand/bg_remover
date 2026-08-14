from __future__ import annotations

import argparse
from pathlib import Path

from quality_benchmark.config import load_benchmark_config
from quality_benchmark.runner import run_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated BG Remover quality research benchmark")
    parser.add_argument("--config", type=Path, default=Path("benchmark_quality.yaml"))
    parser.add_argument("--source", type=Path, help="Override local source photos; never copied to the repository")
    parser.add_argument("--output", type=Path, help="Override gitignored benchmark output directory")
    args = parser.parse_args()

    config = load_benchmark_config(args.config)
    if args.source or args.output:
        from dataclasses import replace

        config = replace(
            config,
            source=args.source.resolve() if args.source else config.source,
            output=args.output.resolve() if args.output else config.output,
        )
    run_benchmark(config, progress=lambda message: print(message, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
