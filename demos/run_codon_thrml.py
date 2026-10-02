#!/usr/bin/env python3
"""Codon-style Ising weight sweep sampled with THRML.

From a checkout, after ``pip install -e ".[dev,cpu]"``::

    python demos/run_codon_thrml.py
    python demos/run_codon_thrml.py --smoke

``--smoke`` is ``profile=smoke`` (CI). No flag is ``profile=default``.
The multi-seed deep budget is ``benchmarks/run_deep.py``, not this script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from unsga3_extropic import __version__
from unsga3_extropic.fidelity import load_front
from unsga3_extropic.problems import CodonIsingProblem
from unsga3_extropic.results import append_run_record, measure_codon_ising
from unsga3_extropic.schedules import codon_problem_kwargs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Codon-style Ising weight sweep via THRML IsingEBM."
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
    front_path = out_dir / "codon_front.npz"

    t0 = time.perf_counter()
    record = measure_codon_ising(smoke=bool(args.smoke), front_path=front_path)
    elapsed = time.perf_counter() - t0
    record["notes"] = record["notes"] + f" elapsed_s={elapsed:.3f}. version={__version__}."
    append_run_record(out_dir / "codon_runs.jsonl", record)
    append_run_record(ROOT / "benchmarks" / "records" / "runs.jsonl", record)
    front = load_front(front_path, n_obj=2)

    problem = CodonIsingProblem(**codon_problem_kwargs())
    _, f_exact = problem.enumerate_front()
    found = set(map(tuple, np.round(front, 5)))
    exact = set(map(tuple, np.round(f_exact, 5)))
    summary = {
        "elapsed_s": round(elapsed, 3),
        "version": __version__,
        "n_spins": problem.n_spins,
        "backend": record["backend"],
        "problem": record["problem"],
        "smoke": bool(args.smoke),
        "profile": "smoke" if args.smoke else "default",
        "eval_budget": record["eval_budget"],
        "metrics": record["metrics"],
        "exact_nd_size": int(len(f_exact)),
        "exact_front_coverage": float(len(found & exact) / len(exact)) if exact else 0.0,
        "git_sha": record["git_sha"],
        "betas": record["schedule"]["betas"],
        "n_warmup": record["schedule"]["n_warmup"],
        "n_samples": record["schedule"]["n_samples"],
        "steps_per_sample": record["schedule"]["steps_per_sample"],
    }
    print("=== unsga3-extropic codon THRML demo ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"  ND front (first 8):\n{front[:8]}")

    (out_dir / "codon_summary.json").write_text(json.dumps(summary, indent=2))
    np.save(out_dir / "codon_front.npy", front)
    np.save(out_dir / "codon_exact_front.npy", f_exact)
    print(f"  wrote {out_dir / 'codon_summary.json'}")
    print(f"  appended {out_dir / 'codon_runs.jsonl'}")
    print(f"  appended {ROOT / 'benchmarks' / 'records' / 'runs.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
