#!/usr/bin/env python3
"""Codon-style Ising weight sweep sampled with THRML.

From a checkout, after ``pip install -e ".[dev,cpu]"``::

    python demos/run_codon_thrml.py
    python demos/run_codon_thrml.py --smoke
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

from unsga3_extropic import AnnealConfig, WeightSweepLoop, __version__
from unsga3_extropic.problems import CodonIsingProblem
from unsga3_extropic.results import append_run_record, build_measured_record


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

    problem = CodonIsingProblem(n_spins=12, J=1.0, seed=0, global_bias=0.05)
    backend = problem.make_backend()
    if args.smoke:
        weights = np.array([[0.5, 0.5], [0.8, 0.2]], dtype=np.float64)
        anneal = AnnealConfig(
            betas=(1.0, 4.0),
            n_warmup=4,
            n_samples=6,
            steps_per_sample=1,
        )
    else:
        weights = np.array(
            [
                [0.9, 0.1],
                [0.7, 0.3],
                [0.5, 0.5],
                [0.3, 0.7],
                [0.1, 0.9],
            ],
            dtype=np.float64,
        )
        anneal = AnnealConfig(
            betas=(0.5, 1.0, 2.0, 4.0),
            n_warmup=20,
            n_samples=24,
            steps_per_sample=2,
        )
    loop = WeightSweepLoop(
        backend=backend,
        anneal=anneal,
        seed=7,
        weights=weights,
        n_obj=2,
    )

    t0 = time.perf_counter()
    result = loop.run()
    elapsed = time.perf_counter() - t0
    _x_nd, f_nd = result.nondominated_front()
    summary = result.summary()
    summary["elapsed_s"] = round(elapsed, 3)
    summary["version"] = __version__
    summary["n_spins"] = problem.n_spins
    summary["backend"] = "thrml_ising"
    summary["problem"] = "codon_ising"
    summary["smoke"] = bool(args.smoke)
    summary["betas"] = list(anneal.betas)
    summary["n_warmup"] = anneal.n_warmup
    summary["n_samples"] = anneal.n_samples
    summary["steps_per_sample"] = anneal.steps_per_sample

    _, f_exact = problem.enumerate_front()
    summary["exact_nd_size"] = int(len(f_exact))
    found = set(map(tuple, np.round(f_nd, 5)))
    exact = set(map(tuple, np.round(f_exact, 5)))
    summary["exact_front_coverage"] = (
        float(len(found & exact) / len(exact)) if exact else 0.0
    )

    record = build_measured_record(
        result,
        anneal=anneal,
        base_seed=7,
        issue="5",
        phase="2",
        problem="codon_ising",
        backend="thrml_ising",
        front_path=out_dir / "codon_front.npz",
        reference=f_exact,
        notes=(
            "GD and coverage compare the non-dominated archive to "
            "CodonIsingProblem.enumerate_front (minimization). "
            "eval_budget counts recorded samples. "
            "Candidates come from sample_weight only."
        ),
    )
    append_run_record(out_dir / "codon_runs.jsonl", record)
    append_run_record(ROOT / "benchmarks" / "records" / "runs.jsonl", record)

    print("=== unsga3-extropic codon THRML demo ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  ND front (first 8):\n{f_nd[:8]}")

    (out_dir / "codon_summary.json").write_text(json.dumps(summary, indent=2))
    np.save(out_dir / "codon_front.npy", f_nd)
    np.save(out_dir / "codon_exact_front.npy", f_exact)
    print(f"  wrote {out_dir}/codon_summary.json")
    print(f"  appended {out_dir / 'codon_runs.jsonl'}")
    print(f"  appended {ROOT / 'benchmarks' / 'records' / 'runs.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
