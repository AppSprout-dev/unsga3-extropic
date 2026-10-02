#!/usr/bin/env python3
"""Append one benchmark JSONL record.

    python benchmarks/append_run.py --stub --out benchmarks/records/runs.jsonl
    python benchmarks/append_run.py --smoke --out benchmarks/records/runs.jsonl

``--smoke`` samples the in-repo Potts chain with THRML (``profile=smoke``).
``--stub`` only checks that a record can be appended. The multi-seed deep
budget is ``benchmarks/run_deep.py``, not a flag on this script.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unsga3_extropic.results import (  # noqa: E402
    append_run_record,
    measure_potts_chain,
    stub_run_record,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append a benchmark run record.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="Sample PottsChainProblem (THRML) and record metrics.",
    )
    mode.add_argument(
        "--stub",
        action="store_true",
        help="Append a non-measurement record. Does not sample.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmarks" / "records" / "runs.jsonl",
        help="JSONL path to append.",
    )
    parser.add_argument(
        "--front",
        type=Path,
        default=None,
        help="Front .npz path. Default: artifacts/ next to --out.",
    )
    return parser.parse_args(argv)


def _default_front(out: Path, stem: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return out.parent / "artifacts" / f"{stem}-{stamp}.npz"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.stub:
        front = args.front or _default_front(args.out, "stub")
        record = stub_run_record(front_path=front)
    else:
        front = args.front or _default_front(args.out, "potts_chain")
        record = measure_potts_chain(smoke=True, front_path=front)
    append_run_record(args.out, record)
    print(f"appended {record['problem']} / {record['backend']} -> {args.out}")
    print(f"  nd_count: {record['metrics']['nd_count']}")
    print(f"  front: {record['artifacts']['front']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
