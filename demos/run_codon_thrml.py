#!/usr/bin/env python3
"""Smoke / demo: U-NSGA-III-shaped weight sweep on codon-style Ising via THRML.

Usage (from anywhere, using the workspace venv)::

  /workspace/extropic-first-job/.venv/bin/python \\
    /workspace/extropic-first-job/unsga3-extropic/demos/run_codon_thrml.py

Or after editable install::

  PYTHONPATH=src python demos/run_codon_thrml.py
"""

from __future__ import annotations

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


def main() -> int:
    out_dir = ROOT / "demos" / "_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    problem = CodonIsingProblem(n_spins=12, J=1.0, seed=0, global_bias=0.05)
    backend = problem.make_backend()
    # Prefer interior + near-extreme niches (avoid w=(1,0)/(0,1) pure singles
    # only — include them for coverage like reference directions)
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
    # Fast smoke anneal (still multi-beta)
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
    x_nd, f_nd = result.nondominated_front()
    summary = result.summary()
    summary["elapsed_s"] = round(elapsed, 3)
    summary["version"] = __version__
    summary["n_spins"] = problem.n_spins
    summary["backend"] = "thrml_ising"
    summary["problem"] = "codon_ising"

    # Exact front for fidelity of archive coverage (small n)
    _, f_exact = problem.enumerate_front()
    summary["exact_nd_size"] = int(len(f_exact))
    # fraction of exact front points found (rounded)
    found = set(map(tuple, np.round(f_nd, 5)))
    exact = set(map(tuple, np.round(f_exact, 5)))
    summary["exact_front_coverage"] = (
        float(len(found & exact) / len(exact)) if exact else 0.0
    )

    print("=== unsga3-extropic codon THRML demo ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  ND front (first 8):\n{f_nd[:8]}")

    (out_dir / "codon_summary.json").write_text(json.dumps(summary, indent=2))
    np.save(out_dir / "codon_front.npy", f_nd)
    np.save(out_dir / "codon_exact_front.npy", f_exact)
    print(f"  wrote {out_dir}/codon_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
