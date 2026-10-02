#!/usr/bin/env python3
"""Classical continuous U-NSGA-III on the Bend ZDT / DTLZ2 yardstick.

ZDT1 n=30, ZDT2 n=30, and DTLZ2 M=3 k=10, seeds 1–15. Budgets are
pop*gens from unsga3-bend docs/ORACLE-MULTISEED.md (ZDT1 52/100,
ZDT2 52/250, DTLZ2 92/150). The sampler is ``unsga3_extropic.classical``.
It is NumPy. It is not THRML and it does not call ``WeightSweepLoop``.

IGD is ``fidelity.inverted_generational_distance`` against the analytic
reference fronts described in ``benchmarks/ORACLE_RESULTS.md``. pymoo and
Bend are not imported.

From the repository root::

    python benchmarks/run_classical_unsga3.py

Writes non-dominated fronts under
``benchmarks/records/classical_unsga3/`` and rewrites
``benchmarks/ORACLE_UNSGA3_RESULTS.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Never

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmarks"))

import numpy as np

from classical_report import read_exactew_table, render_classical_results  # noqa: E402
from unsga3_extropic.classical.igd import score_front  # noqa: E402
from unsga3_extropic.classical.unsga3 import run_unsga3  # noqa: E402
from unsga3_extropic.problems.continuous import (  # noqa: E402
    ORACLE_SEEDS,
    OracleName,
    dtlz2,
    oracle_specs,
    zdt1,
    zdt2,
)
from unsga3_extropic.results import current_git_sha, utc_now_iso  # noqa: E402

FRONT_DIR = ROOT / "benchmarks" / "records" / "classical_unsga3"
DEFAULT_REPORT = ROOT / "benchmarks" / "ORACLE_UNSGA3_RESULTS.md"
EXACTEW_REPORT = ROOT / "benchmarks" / "ORACLE_RESULTS.md"


def _unreachable(name: Never) -> Never:
    raise AssertionError(f"unhandled oracle problem {name!r}")


def objective_for(problem: OracleName):
    match problem:
        case "zdt1":
            return zdt1
        case "zdt2":
            return zdt2
        case "dtlz2":
            return dtlz2
        case _ as other:
            _unreachable(other)


def igd_text(value: float) -> str:
    """Shortest round-trip decimal, matching a printed ``igd=`` float."""
    return format(value, ".16g")


def write_front(path: Path, front: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(format(float(value), ".17g") for value in row) for row in front]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def run_one(problem: OracleName, seed: int) -> dict:
    spec = oracle_specs()[problem]
    result = run_unsga3(
        objective_for(problem),
        n_var=spec.n_var,
        n_obj=spec.n_obj,
        pop_size=spec.pop,
        n_gen=spec.gens,
        partitions=spec.partitions,
        seed=seed,
        lower=spec.lower,
        upper=spec.upper,
    )
    if result.n_evals != spec.pop * spec.gens:
        raise AssertionError(
            f"{problem} seed {seed} spent {result.n_evals} evals, expected {spec.pop * spec.gens}"
        )
    igd, pf_rows, pf_source = score_front(
        problem, result.front, partitions=spec.partitions
    )
    front_path = FRONT_DIR / f"{problem}_seed{seed:02d}.csv"
    write_front(front_path, result.front)
    text = igd_text(igd)
    sidecar = front_path.with_suffix(".igd.txt")
    sidecar.write_text(
        "\n".join(
            [
                f"igd={text}",
                f"problem={problem}",
                f"front_rows={len(result.front)}",
                f"pf_rows={pf_rows}",
                f"pf_source={pf_source}",
                f"objective_evals={result.n_evals}",
                f"seed={seed}",
                "sampler=classical-unsga3",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        f"{problem} seed {seed}: igd={text} front_rows={len(result.front)} evals={result.n_evals}",
        flush=True,
    )
    return {
        "problem": problem,
        "seed": seed,
        "igd_text": text,
        "front_rows": len(result.front),
        "pf_rows": pf_rows,
        "pf_source": pf_source,
        "objective_evals": result.n_evals,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("zdt1", "zdt2", "dtlz2", "all"), default="all")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    problems: tuple[OracleName, ...]
    if args.problem == "all":
        problems = ("zdt1", "zdt2", "dtlz2")
    else:
        problems = (args.problem,)
    seeds = tuple(args.seeds) if args.seeds else ORACLE_SEEDS
    rows = [run_one(problem, seed) for problem in problems for seed in seeds]
    exactew = read_exactew_table(EXACTEW_REPORT)
    partial = set(problems) != {"zdt1", "zdt2", "dtlz2"} or set(seeds) != set(ORACLE_SEEDS)
    text = render_classical_results(
        rows,
        exactew=exactew,
        generated_at=utc_now_iso(),
        git_sha=current_git_sha(),
        partial=partial,
    )
    args.report.write_text(text, encoding="utf-8")
    print(f"wrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
