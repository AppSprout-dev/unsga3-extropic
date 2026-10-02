#!/usr/bin/env python3
"""NumPy ExactEw weight-sweep on the Bend continuous yardstick.

ZDT1 n=30, ZDT2 n=30, and DTLZ2 M=3 k=10, seeds 1–15. Objective-call
budgets track pop*gens from unsga3-bend docs/ORACLE-MULTISEED.md
(ZDT1 52/100, ZDT2 52/250, DTLZ2 92/150). The sampler is
ExactEwContinuousBackend. It is not THRML and not U-NSGA-III.

IGD is whatever ``ab/igd_vs_pymoo.py`` prints. This script does not
compute a substitute. Point ``UNSGA3_BEND_ROOT`` at a checkout of
AppSprout-dev/unsga3-bend, or clone it at ``/workspace/unsga3-bend``.

From the repository root::

    python benchmarks/run_oracle_continuous.py

Writes ND objective CSVs under ``benchmarks/records/oracle_fronts/``,
appends JSONL rows, and rewrites ``benchmarks/ORACLE_RESULTS.md`` when
seeds 1–15 of all three problems have an ``igd=`` line.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmarks"))

import numpy as np

from oracle_report import (  # noqa: E402
    OracleIgdRow,
    igd_command,
    parse_igd_stdout,
    render_oracle_results,
)
from unsga3_extropic.backends import ExactEwContinuousBackend  # noqa: E402
from unsga3_extropic.loop import AnnealConfig, WeightSweepLoop  # noqa: E402
from unsga3_extropic.problems.continuous import (  # noqa: E402
    ORACLE_SEEDS,
    OracleName,
    oracle_specs,
)
from unsga3_extropic.results import (  # noqa: E402
    append_run_record,
    build_measured_record,
    current_git_sha,
    read_run_records,
    utc_now_iso,
)
from unsga3_extropic.weights import simplex_weights  # noqa: E402

BACKEND = "exact_ew_continuous"
FRONT_DIR = ROOT / "benchmarks" / "records" / "oracle_fronts"
DEFAULT_LOG = ROOT / "benchmarks" / "records" / "runs.jsonl"
DEFAULT_REPORT = ROOT / "benchmarks" / "ORACLE_RESULTS.md"


def _unreachable(name: object) -> None:
    raise AssertionError(f"unhandled oracle problem {name!r}")


def find_igd_script() -> Path:
    """Locate Bend's IGD script. Does not vendor it."""
    candidates: list[Path] = []
    env = os.environ.get("UNSGA3_BEND_ROOT")
    if env:
        candidates.append(Path(env) / "ab" / "igd_vs_pymoo.py")
    candidates.append(Path("/workspace/unsga3-bend/ab/igd_vs_pymoo.py"))
    candidates.append(ROOT.parent / "unsga3-bend" / "ab" / "igd_vs_pymoo.py")
    for path in candidates:
        if path.is_file():
            return path
    joined = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "igd_vs_pymoo.py was not found. Clone AppSprout-dev/unsga3-bend "
        f"or set UNSGA3_BEND_ROOT. Looked at: {joined}"
    )


def write_objective_csv(path: Path, front: np.ndarray) -> None:
    """One objective vector per line. No header. ``#`` lines are not written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.asarray(front, dtype=np.float64)
    if rows.ndim != 2 or len(rows) == 0:
        raise ValueError(f"ND front must be a non-empty 2-D array, got {rows.shape}")
    lines = [
        ",".join(format(float(value), ".16g") for value in row) for row in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def require_yardstick_pf(problem: OracleName, parsed: dict[str, str]) -> None:
    """Refuse a run whose printed PF is not the Bend yardstick."""
    if "igd" not in parsed or parsed["igd"] == "":
        raise RuntimeError(f"{problem}: IGD script stdout has no igd= line: {parsed}")
    pf_rows = parsed.get("pf_rows", "")
    source = parsed.get("pf_source", "")
    if problem == "zdt1":
        ok = pf_rows == "500" and source.startswith("analytic-zdt1")
    elif problem == "zdt2":
        ok = pf_rows == "500" and source.startswith("analytic-zdt2")
    elif problem == "dtlz2":
        ok = pf_rows == "91" and "das-dennis" in source
    else:
        _unreachable(problem)
        ok = False
    if not ok:
        raise RuntimeError(
            f"{problem}: PF yardstick mismatch (wanted analytic 500 or "
            f"Das–Dennis 91). Script printed pf_rows={pf_rows!r} "
            f"pf_source={source!r}. Refusing to record an IGD."
        )


def score_csv(script: Path, csv_path: Path, problem: OracleName) -> tuple[dict[str, str], str]:
    command = igd_command(sys.executable, str(script), str(csv_path), problem)
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    parsed = parse_igd_stdout(completed.stdout)
    if "igd" not in parsed:
        raise RuntimeError(
            "igd_vs_pymoo.py did not print igd=.\n"
            f"command: {' '.join(command)}\n"
            f"exit: {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    require_yardstick_pf(problem, parsed)
    stdout_path = csv_path.with_suffix(".igd.txt")
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    return parsed, completed.stdout


def _row_matches(record: dict, problem: str, seed: int, n_weights: int, n_samples: int) -> bool:
    if record.get("problem") != problem or record.get("backend") != BACKEND:
        return False
    seeds = record.get("seeds") or []
    schedule = record.get("schedule") or {}
    if not seeds or seeds[0] != seed:
        return False
    if schedule.get("n_weights") != n_weights or schedule.get("n_samples") != n_samples:
        return False
    return "igd=" in str(record.get("notes", ""))


def already_logged(records: list[dict], problem: str, seed: int, n_weights: int, n_samples: int) -> bool:
    return any(_row_matches(record, problem, seed, n_weights, n_samples) for record in records)


def front_stem(problem: str, seed: int) -> str:
    return f"{problem}_seed{seed:02d}"


def run_seed(problem: OracleName, seed: int):
    spec = oracle_specs()[problem]
    weights = simplex_weights(spec.n_weights, spec.n_obj)
    if weights.shape != (spec.n_weights, spec.n_obj):
        raise RuntimeError(
            f"{problem}: expected {spec.n_weights} Das–Dennis weights, got {weights.shape}"
        )
    backend = ExactEwContinuousBackend(
        n_var=spec.n_var,
        objective_fn=spec.objective_fn(),
        lower=spec.lower,
        upper=spec.upper,
        step_scale=spec.step_scale,
    )
    anneal = AnnealConfig(
        betas=spec.betas,
        n_warmup=spec.n_warmup,
        n_samples=spec.n_samples,
        steps_per_sample=spec.steps_per_sample,
    )
    started = time.perf_counter()
    result = WeightSweepLoop(
        backend=backend,
        anneal=anneal,
        seed=seed,
        weights=weights,
        n_obj=spec.n_obj,
    ).run()
    elapsed = time.perf_counter() - started
    if result.total_evals != spec.objective_evals():
        raise RuntimeError(
            f"{problem} seed {seed}: objective evals {result.total_evals} "
            f"!= scheduled {spec.objective_evals()}"
        )
    recorded = int(sum(len(row.objectives) for row in result.per_weight))
    if recorded != spec.recorded_samples():
        raise RuntimeError(
            f"{problem} seed {seed}: recorded {recorded} != {spec.recorded_samples()}"
        )
    _decisions, front = result.nondominated_front()
    return result, front, anneal, elapsed


def parse_seed_list(text: str) -> tuple[int, ...]:
    seeds: list[int] = []
    for part in text.split(","):
        piece = part.strip()
        if not piece:
            continue
        if "-" in piece:
            lo_s, hi_s = piece.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"seed range {piece} is reversed")
            seeds.extend(range(lo, hi + 1))
        else:
            seeds.append(int(piece))
    if not seeds:
        raise ValueError("no seeds")
    if any(seed < 1 for seed in seeds):
        raise ValueError("seeds must be >= 1")
    return tuple(seeds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", default="zdt1,zdt2,dtlz2")
    parser.add_argument("--seeds", default="1-15")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    names = tuple(part.strip() for part in args.problems.split(",") if part.strip())
    for name in names:
        if name not in oracle_specs():
            raise SystemExit(f"unknown problem {name}")
    seeds = parse_seed_list(args.seeds)
    script = find_igd_script()
    specs = oracle_specs()
    existing = read_run_records(args.log) if args.log.is_file() else []
    scored: list[OracleIgdRow] = []

    for name in names:
        problem: OracleName = name  # type: ignore[assignment]
        spec = specs[problem]
        for seed in seeds:
            stem = front_stem(problem, seed)
            csv_path = FRONT_DIR / f"{stem}.csv"
            igd_path = FRONT_DIR / f"{stem}.igd.txt"
            npz_path = FRONT_DIR / f"{stem}.npz"
            logged = already_logged(
                existing, problem, seed, spec.n_weights, spec.n_samples
            )
            if logged and igd_path.is_file() and csv_path.is_file():
                parsed = parse_igd_stdout(igd_path.read_text(encoding="utf-8"))
                require_yardstick_pf(problem, parsed)
                print(f"resume {problem} seed {seed} igd={parsed['igd']}", flush=True)
            else:
                print(f"run {problem} seed {seed}", flush=True)
                result, front, anneal, elapsed = run_seed(problem, seed)
                write_objective_csv(csv_path, front)
                parsed, _stdout = score_csv(script, csv_path, problem)
                notes = (
                    "profile=oracle-continuous. "
                    "NumPy ExactEwContinuousBackend weight-sweep. "
                    "Not THRML-native. Not an Ising or Potts factor energy. "
                    "No surrogate was fit to force f into IsingEBM. "
                    "Not U-NSGA-III: no SBX, no polynomial mutation, "
                    "no generational population. "
                    "Proposal adds Normal(0, 0.1) to one coordinate, redrawn until "
                    "it stays in [0, 1], with the truncated-normal Hastings correction. "
                    f"instance=n_var={spec.n_var} box=[0,1] partitions={spec.partitions} "
                    f"pop={spec.pop} gens={spec.gens}. "
                    f"Yardstick objective calls pop*gens={spec.yardstick_evals()}. "
                    f"objective_evals={spec.objective_evals()} "
                    "(initial state plus every proposal). "
                    f"recorded_samples={spec.recorded_samples()}. "
                    "eval_budget counts recorded samples. "
                    "IGD is only the igd= line from unsga3-bend ab/igd_vs_pymoo.py. "
                    f"igd={parsed['igd']} "
                    f"front_rows={parsed.get('front_rows', '')} "
                    f"pf_rows={parsed.get('pf_rows', '')} "
                    f"pf_source={parsed.get('pf_source', '')}. "
                    "metrics.generational_distance is null; that field is not pymoo IGD. "
                    f"csv={csv_path.relative_to(ROOT)} "
                    f"elapsed_s={elapsed:.3f}."
                )
                if not logged:
                    record = build_measured_record(
                        result,
                        anneal=anneal,
                        base_seed=seed,
                        issue="none",
                        phase="oracle",
                        problem=problem,
                        backend=BACKEND,
                        front_path=npz_path,
                        notes=notes,
                        reference=None,
                        eval_budget=spec.recorded_samples(),
                    )
                    append_run_record(args.log, record)
                    existing.append(record)
                print(
                    f"scored {problem} seed {seed} igd={parsed['igd']} "
                    f"pf_rows={parsed.get('pf_rows')} elapsed_s={elapsed:.1f}",
                    flush=True,
                )
            scored.append(
                OracleIgdRow(
                    problem=problem,
                    seed=seed,
                    igd_text=parsed["igd"],
                    front_rows=parsed.get("front_rows", ""),
                    pf_rows=parsed.get("pf_rows", ""),
                    pf_source=parsed.get("pf_source", ""),
                    objective_evals=spec.objective_evals(),
                    recorded_samples=spec.recorded_samples(),
                    nd_count=int(parsed.get("front_rows", "0") or "0"),
                )
            )

    full = set(names) == {"zdt1", "zdt2", "dtlz2"} and set(seeds) == set(ORACLE_SEEDS)
    report = render_oracle_results(
        scored,
        generated_at=utc_now_iso(),
        git_sha=current_git_sha(),
        partial=not full,
    )
    args.report.write_text(report, encoding="utf-8")
    print(f"wrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
