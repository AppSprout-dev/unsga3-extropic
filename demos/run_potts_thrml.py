#!/usr/bin/env python3
"""Potts-chain weight sweep sampled with THRML categorical factors.

From a checkout, after ``pip install -e ".[dev,cpu]"``::

    python demos/run_potts_thrml.py
    python demos/run_potts_thrml.py --smoke

This is the in-repo unary-plus-pairwise chain, not the codon walkthrough
and not a domain-wall Ising model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from unsga3_extropic import __version__
from unsga3_extropic.fidelity import load_front
from unsga3_extropic.results import append_run_record, measure_potts_chain


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Potts chain weight sweep via THRML CategoricalEBMFactor."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Short anneal for CI (fewer weights, betas, and samples).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = ROOT / "demos" / "_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    front_path = out_dir / "potts_front.npz"

    t0 = time.perf_counter()
    record = measure_potts_chain(smoke=bool(args.smoke), front_path=front_path)
    elapsed = time.perf_counter() - t0
    record["notes"] = record["notes"] + f" elapsed_s={elapsed:.3f}. version={__version__}."
    append_run_record(out_dir / "potts_runs.jsonl", record)
    append_run_record(ROOT / "benchmarks" / "records" / "runs.jsonl", record)
    front = load_front(front_path, n_obj=2)

    summary = {
        "elapsed_s": round(elapsed, 3),
        "version": __version__,
        "backend": record["backend"],
        "problem": record["problem"],
        "smoke": bool(args.smoke),
        "eval_budget": record["eval_budget"],
        "metrics": record["metrics"],
        "git_sha": record["git_sha"],
    }
    print("=== unsga3-extropic Potts THRML demo ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"  ND front (first 8):\n{front[:8]}")

    (out_dir / "potts_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  wrote {out_dir / 'potts_summary.json'}")
    print(f"  appended {out_dir / 'potts_runs.jsonl'}")
    print(f"  appended {ROOT / 'benchmarks' / 'records' / 'runs.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
