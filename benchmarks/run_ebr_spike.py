#!/usr/bin/env python3
"""EBR spike: lowest-energy selection and a plateau stop on the deep codon and Potts tasks.

From the repository root, after ``pip install -e ".[dev,cpu]"``::

    export JAX_PLATFORMS=cpu
    python benchmarks/run_ebr_spike.py            # ExactEw + THRML Ising
    python benchmarks/run_ebr_spike.py --skip-thrml

Tasks: the two deep instances in ``schedules`` (``DEEP_SCHEDULE``: 11
weights, 6 betas, 24 warmup and 24 recorded samples per beta, seeds 7,
11, 19). ``codon`` is the 12-spin chain (4096 states, exact front of 3
points). ``potts`` is the 6-site, K=3 chain (729 states, exact front of
5 points). Exact fronts come from each problem's ``enumerate_front``. The
exact per-weight minimum of ``E_w`` comes from enumerating every state.
On ``potts`` the ExactEw chain is the same Metropolis loop with a
categorical proposal (one site moves to one of the other K-1 categories,
a symmetric move).

Variants (all opt-in, the default path is ``baseline``):

* ``baseline``: ``ExactEwMetropolisBackend`` as shipped.
* ``select``: the same chain through ``LowestEnergySelection`` (one row per weight).
* ``plateau``: ``ExactEwMetropolisBackend(plateau=PlateauStop(window, tol))``.
* ``plateau+select``: both.
* ``thrml``, ``thrml+select``: the shipped THRML backend (Ising block
  Gibbs for codon, categorical block Gibbs for Potts), and the
  argmin row of each of its weight results. THRML runs one program per
  beta stage, so the plateau rule is only evaluated after the fact here
  (``thrml_plateau_posthoc_samples``); it saves no THRML work.

The plateau window and tolerance are fixed before the run (window 8,
tol 0 on these integer-valued energies; window 4 as a sensitivity row).
They were not tuned on the results.

Writes ``benchmarks/records/ebr_spike.json``. Does not touch ``runs.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from unsga3_extropic.archive import hypervolume_2d  # noqa: E402
from unsga3_extropic.backends import ExactEwMetropolisBackend  # noqa: E402
from unsga3_extropic.ebr import (  # noqa: E402
    LowestEnergySelection,
    PlateauStop,
    select_lowest_energy,
)
from unsga3_extropic.fidelity import coverage, generational_distance  # noqa: E402
from unsga3_extropic.loop import LoopResult, WeightSweepLoop  # noqa: E402
from unsga3_extropic.problems.codon_ising import CodonIsingProblem  # noqa: E402
from unsga3_extropic.problems.potts_chain import PottsChainProblem  # noqa: E402
from unsga3_extropic.results import _codon_bit_objectives, current_git_sha  # noqa: E402
from unsga3_extropic.schedules import (  # noqa: E402
    DEEP_SCHEDULE,
    codon_problem_kwargs,
    potts_instance,
)

PLATEAU_MAIN = PlateauStop(window=8, tol=0.0)
PLATEAU_SENS = PlateauStop(window=4, tol=0.0)
OUT = ROOT / "benchmarks" / "records" / "ebr_spike.json"


def _all_objectives(problem: CodonIsingProblem) -> np.ndarray:
    n = problem.n_spins
    k = np.arange(2**n, dtype=np.uint32)[:, None]
    shifts = np.arange(n - 1, -1, -1, dtype=np.uint32)
    bits = ((k >> shifts) & 1).astype(np.float64)
    return problem.energies_from_spins(2.0 * bits - 1.0)


def _hv_ref(exact: np.ndarray) -> tuple[float, float]:
    m0, m1 = float(exact[:, 0].max()), float(exact[:, 1].max())
    return (m0 + 0.1 * max(abs(m0), 1.0), m1 + 0.1 * max(abs(m1), 1.0))


def _score(
    result: LoopResult,
    *,
    exact: np.ndarray,
    ref: tuple[float, float],
    hv_exact: float,
    global_min: np.ndarray,
) -> dict:
    _x, front = result.nondominated_front()
    rows = int(sum(len(r.objectives) for r in result.per_weight))
    # Recompute w · f in float64: THRML Potts reports float32 energies (~3e-7 off).
    best = np.array(
        [float(np.min(r.objectives @ w)) for r, w in zip(result.per_weight, result.weights)]
    )
    hits = int(np.sum(np.abs(best - global_min) <= 1e-6))
    return {
        "recorded_rows": rows,
        "energy_evals": int(result.total_evals),
        "nd_count": int(len(front)),
        "hv_ratio": hypervolume_2d(front, ref=ref) / hv_exact,
        "gd": generational_distance(front, exact),
        "coverage": coverage(front, exact),
        "argmin_hits": hits,
        "n_weights": int(len(result.weights)),
    }


def _select_after(result: LoopResult) -> LoopResult:
    """Apply ``select_lowest_energy`` to each weight result (same as the wrapper)."""
    from unsga3_extropic.archive import Archive

    archive = Archive()
    per = []
    for i, r in enumerate(result.per_weight):
        s = select_lowest_energy(r)
        archive.add(s.decisions, s.objectives, weight_id=i)
        per.append(s)
    return LoopResult(
        archive=archive, weights=result.weights, per_weight=per, total_evals=result.total_evals
    )


def _posthoc_plateau_rows(result: LoopResult, stop: PlateauStop, n_betas: int) -> int:
    kept = 0
    for r in result.per_weight:
        stages = np.asarray(r.energies, dtype=np.float64).reshape(n_betas, -1)
        for trace in stages:
            k = stop.stop_index(trace)
            kept += len(trace) if k is None else k + 1
    return kept


def _loop(backend, seed: int) -> LoopResult:
    return WeightSweepLoop(
        backend=backend,
        anneal=DEEP_SCHEDULE.anneal(),
        seed=seed,
        weights=DEEP_SCHEDULE.weight_matrix(),
        n_obj=2,
    ).run()


def _codon_task() -> dict:
    problem = CodonIsingProblem(**codon_problem_kwargs())
    return {
        "name": "codon",
        "label": "codon_ising deep (n_spins=12, J=1, field_seed=0, global_bias=0.05)",
        "exact": problem.enumerate_front()[1],
        "all_f": _all_objectives(problem),
        "make_ew": lambda plateau: ExactEwMetropolisBackend(
            n_bits=problem.n_spins,
            objective_fn=_codon_bit_objectives(problem),
            plateau=plateau,
        ),
        "make_thrml": problem.make_backend,
        "thrml_backend": "thrml_ising",
    }


def _potts_task() -> dict:
    n_sites, k = potts_instance("deep")
    problem = PottsChainProblem(n_sites=n_sites, n_categories=k)

    def propose(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        y = x.copy()
        i = int(rng.integers(0, len(y)))
        y[i] = (int(y[i]) + int(rng.integers(1, k))) % k
        return y

    def init(rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, k, size=n_sites).astype(np.float64)

    def objective_fn(states: np.ndarray) -> np.ndarray:
        return problem.energies_from_states(np.asarray(states).astype(np.int64))

    return {
        "name": "potts",
        "label": f"potts_chain deep (n_sites={n_sites}, K={k})",
        "exact": problem.enumerate_front()[1],
        "all_f": np.asarray(problem.energies_from_states(problem.enumerate_states()), dtype=np.float64),
        "make_ew": lambda plateau: ExactEwMetropolisBackend(
            n_bits=n_sites,
            objective_fn=objective_fn,
            propose=propose,
            init_fn=init,
            plateau=plateau,
        ),
        "make_thrml": problem.make_backend,
        "thrml_backend": "thrml_potts",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-thrml", action="store_true")
    parser.add_argument("--task", action="append", choices=["codon", "potts"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    chosen = args.task or ["codon", "potts"]

    wmat = DEEP_SCHEDULE.weight_matrix()
    n_betas = len(DEEP_SCHEDULE.betas)
    rows: list[dict] = []
    tasks_meta: dict = {}
    for task in [t() for t in (_codon_task, _potts_task)]:
        if task["name"] not in chosen:
            continue
        exact = task["exact"]
        ref = _hv_ref(exact)
        hv_exact = hypervolume_2d(exact, ref=ref)
        global_min = (task["all_f"] @ wmat.T).min(axis=0)
        tasks_meta[task["name"]] = {
            "label": task["label"],
            "exact_front_size": int(len(exact)),
            "hv_ref": list(ref),
            "hv_exact": hv_exact,
        }
        score = lambda res: _score(  # noqa: E731
            res, exact=exact, ref=ref, hv_exact=hv_exact, global_min=global_min
        )
        ew = task["make_ew"]
        for seed in DEEP_SCHEDULE.base_seeds:
            variants = {
                "baseline": ew(None),
                "select": LowestEnergySelection(ew(None)),
                "plateau(w=8)": ew(PLATEAU_MAIN),
                "plateau(w=8)+select": LowestEnergySelection(ew(PLATEAU_MAIN)),
                "plateau(w=4)": ew(PLATEAU_SENS),
                "plateau(w=4)+select": LowestEnergySelection(ew(PLATEAU_SENS)),
            }
            for name, backend in variants.items():
                t0 = time.perf_counter()
                res = _loop(backend, seed)
                dt = time.perf_counter() - t0
                row = {"task": task["name"], "seed": seed, "variant": name,
                       "backend": "exact_ew", "wall_s": dt}
                row.update(score(res))
                rows.append(row)
                print(json.dumps(row), flush=True)
            if args.skip_thrml:
                continue
            t0 = time.perf_counter()
            res = _loop(task["make_thrml"](), seed)
            dt = time.perf_counter() - t0
            base = {"task": task["name"], "seed": seed, "backend": task["thrml_backend"], "wall_s": dt}
            r1 = dict(base, variant="thrml")
            r1.update(score(res))
            r1["thrml_plateau_posthoc_samples"] = _posthoc_plateau_rows(res, PLATEAU_MAIN, n_betas)
            rows.append(r1)
            print(json.dumps(r1), flush=True)
            r2 = dict(base, variant="thrml+select")
            r2.update(score(_select_after(res)))
            rows.append(r2)
            print(json.dumps(r2), flush=True)

    meta = {
        "git_sha": current_git_sha(),
        "tasks": tasks_meta,
        "schedule": {
            "betas": list(DEEP_SCHEDULE.betas),
            "n_warmup": DEEP_SCHEDULE.n_warmup,
            "n_samples": DEEP_SCHEDULE.n_samples,
            "steps_per_sample": DEEP_SCHEDULE.steps_per_sample,
            "n_weights": DEEP_SCHEDULE.n_weights(),
            "seeds": list(DEEP_SCHEDULE.base_seeds),
        },
        "plateau_main": {"window": PLATEAU_MAIN.window, "tol": PLATEAU_MAIN.tol},
        "plateau_sensitivity": {"window": PLATEAU_SENS.window, "tol": PLATEAU_SENS.tol},
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
