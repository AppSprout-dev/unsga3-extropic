"""Append-only JSONL records for benchmark runs.

The schema lives in ``benchmarks/run_record.schema.json``. A record names
the problem, backend, seeds, anneal schedule, and metrics. Front arrays
are files (``.npy`` / ``.npz``); the record stores their paths.

``measure_potts_chain`` is the Potts smoke used by ``demos/run_potts_thrml.py``
and ``benchmarks/append_run.py``. ``measure_domain_wall`` is the same chain
sampled through its domain-wall Ising image. Neither tunes betas.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from unsga3_extropic.fidelity import compare_fronts
from unsga3_extropic.loop import AnnealConfig, LoopResult, WeightSweepLoop
from unsga3_extropic.problems.potts_chain import PottsChainProblem

SCHEMA_VERSION = 1
# WeightSweepLoop uses seed + 1009 * weight_index.
_SEED_STRIDE = 1009

RUN_RECORD_KEYS = (
    "schema_version",
    "date",
    "git_sha",
    "issue",
    "phase",
    "problem",
    "backend",
    "seeds",
    "schedule",
    "eval_budget",
    "metrics",
    "artifacts",
    "notes",
)
SCHEDULE_KEYS = (
    "betas",
    "n_warmup",
    "n_samples",
    "steps_per_sample",
    "n_weights",
)
METRIC_KEYS = (
    "nd_count",
    "hypervolume_2d",
    "hv_ref",
    "generational_distance",
    "coverage",
)


def current_git_sha() -> str:
    """HEAD of the working tree, or ``unknown`` when git is unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    sha = out.strip()
    return sha if sha else "unknown"


def utc_now_iso() -> str:
    """UTC timestamp with a ``Z`` suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(value: str) -> datetime:
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(text)


def _require_keys(obj: dict, keys: tuple[str, ...], *, where: str) -> None:
    missing = [key for key in keys if key not in obj]
    if missing:
        raise ValueError(f"{where} missing keys {missing}")
    extra = [key for key in obj if key not in keys]
    if extra:
        raise ValueError(f"{where} has unknown keys {extra}")


def _as_optional_number(value, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number or null")
    number = float(value)
    if number < 0:
        raise ValueError(f"{name} must be >= 0")
    return number


def validate_run_record(record: dict) -> None:
    """Raise ``ValueError`` when ``record`` does not match schema version 1."""
    if not isinstance(record, dict):
        raise ValueError("run record must be a JSON object")
    _require_keys(record, RUN_RECORD_KEYS, where="run record")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION}, got {record['schema_version']!r}"
        )
    if not isinstance(record["date"], str):
        raise ValueError("date must be an ISO-8601 string")
    _parse_iso(record["date"])
    for key in ("git_sha", "issue", "phase", "problem", "backend"):
        if not isinstance(record[key], str) or record[key] == "":
            raise ValueError(f"{key} must be a non-empty string")
    if not isinstance(record["notes"], str):
        raise ValueError("notes must be a string")
    seeds = record["seeds"]
    if not isinstance(seeds, list) or len(seeds) < 1:
        raise ValueError("seeds must be a non-empty list of integers")
    if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("seeds must be a non-empty list of integers >= 0")

    schedule = record["schedule"]
    if not isinstance(schedule, dict):
        raise ValueError("schedule must be an object")
    _require_keys(schedule, SCHEDULE_KEYS, where="schedule")
    betas = schedule["betas"]
    if not isinstance(betas, list) or len(betas) < 1:
        raise ValueError("schedule.betas must be a non-empty list of numbers")
    if any(isinstance(b, bool) or not isinstance(b, (int, float)) for b in betas):
        raise ValueError("schedule.betas must be a non-empty list of numbers")
    for key, minimum in (
        ("n_warmup", 0),
        ("n_samples", 1),
        ("steps_per_sample", 1),
        ("n_weights", 1),
    ):
        value = schedule[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"schedule.{key} must be an integer >= {minimum}")

    budget = record["eval_budget"]
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("eval_budget must be an integer >= 0")

    metrics = record["metrics"]
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be an object")
    _require_keys(metrics, METRIC_KEYS, where="metrics")
    nd_count = metrics["nd_count"]
    if isinstance(nd_count, bool) or not isinstance(nd_count, int) or nd_count < 0:
        raise ValueError("metrics.nd_count must be an integer >= 0")
    _as_optional_number(metrics["hypervolume_2d"], name="metrics.hypervolume_2d")
    _as_optional_number(
        metrics["generational_distance"], name="metrics.generational_distance"
    )
    coverage = _as_optional_number(metrics["coverage"], name="metrics.coverage")
    if coverage is not None and coverage > 1.0:
        raise ValueError("metrics.coverage must lie in [0, 1]")
    hv_ref = metrics["hv_ref"]
    if hv_ref is not None:
        if (
            not isinstance(hv_ref, list)
            or len(hv_ref) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in hv_ref)
        ):
            raise ValueError("metrics.hv_ref must be null or [number, number]")

    artifacts = record["artifacts"]
    if not isinstance(artifacts, dict):
        raise ValueError("artifacts must be an object")
    _require_keys(artifacts, ("front",), where="artifacts")
    if not isinstance(artifacts["front"], str):
        raise ValueError("artifacts.front must be a string path")


def append_run_record(path: str | Path, record: dict) -> None:
    """Validate ``record`` and append one JSON line."""
    validate_run_record(record)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True)
    with dest.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def read_run_records(path: str | Path) -> list[dict]:
    """Read a JSONL file of run records."""
    records: list[dict] = []
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        try:
            validate_run_record(record)
        except ValueError as exc:
            raise ValueError(f"{file_path}:{line_no}: {exc}") from exc
        records.append(record)
    return records


def display_path(path: Path) -> str:
    """Path relative to the working directory when that stays inside it."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def stub_run_record(*, front_path: Path) -> dict:
    """Schema-valid record that does not claim a measurement."""
    front_path = Path(front_path)
    front_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(front_path, front=np.zeros((0, 2), dtype=np.float64))
    return {
        "schema_version": SCHEMA_VERSION,
        "date": utc_now_iso(),
        "git_sha": current_git_sha(),
        "issue": "none",
        "phase": "none",
        "problem": "stub",
        "backend": "none",
        "seeds": [0],
        "schedule": {
            "betas": [1.0],
            "n_warmup": 0,
            "n_samples": 1,
            "steps_per_sample": 1,
            "n_weights": 1,
        },
        "eval_budget": 0,
        "metrics": {
            "nd_count": 0,
            "hypervolume_2d": None,
            "hv_ref": None,
            "generational_distance": None,
            "coverage": None,
        },
        "artifacts": {"front": display_path(front_path)},
        "notes": "stub record; metrics not measured",
    }


def build_measured_record(
    result: LoopResult,
    *,
    anneal: AnnealConfig,
    base_seed: int,
    issue: str,
    phase: str,
    problem: str,
    backend: str,
    front_path: Path,
    notes: str,
    reference: np.ndarray | None = None,
) -> dict:
    """Validate a run record for one finished ``WeightSweepLoop``.

    ``metrics.nd_count`` is the non-dominated archive. The notes gain one
    sentence with the quota-1 niche survivor count. The front file is
    ``front_path`` (``.npz`` key ``front``).
    """
    _decisions, front = result.nondominated_front()
    scores = compare_fronts(front, reference)
    niche = result.niche_front()
    front_path = Path(front_path)
    front_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(front_path, front=np.asarray(front, dtype=np.float64))
    seeds = [base_seed + _SEED_STRIDE * i for i in range(len(result.weights))]
    niche_note = (
        f"Closer-in-niche survival kept {len(niche.objectives)} of {len(front)} "
        f"non-dominated rows across {len(result.weights)} directions (quota 1). "
        "metrics.nd_count is the non-dominated archive."
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "date": utc_now_iso(),
        "git_sha": current_git_sha(),
        "issue": issue,
        "phase": phase,
        "problem": problem,
        "backend": backend,
        "seeds": seeds,
        "schedule": {
            "betas": [float(beta) for beta in anneal.betas],
            "n_warmup": int(anneal.n_warmup),
            "n_samples": int(anneal.n_samples),
            "steps_per_sample": int(anneal.steps_per_sample),
            "n_weights": int(len(result.weights)),
        },
        "eval_budget": int(result.total_evals),
        "metrics": {
            "nd_count": int(scores["nd_count"]),
            "hypervolume_2d": scores["hypervolume_2d"],
            "hv_ref": scores["hv_ref"],
            "generational_distance": scores["generational_distance"],
            "coverage": scores["coverage"],
        },
        "artifacts": {"front": display_path(front_path)},
        "notes": notes.rstrip() + " " + niche_note,
    }
    validate_run_record(record)
    return record


def measure_potts_chain(*, smoke: bool, front_path: Path) -> dict:
    """Sample ``PottsChainProblem`` and return a validated run record.

    The front file is ``front_path`` (``.npz`` key ``front``). Metrics use
    ``enumerate_front`` as the reference. ``eval_budget`` counts recorded
    samples (``n_weights * len(betas) * n_samples``), not Gibbs micro-steps.
    """
    if smoke:
        problem = PottsChainProblem(n_sites=6, n_categories=3)
        weights = np.array([[0.5, 0.5], [0.8, 0.2]], dtype=np.float64)
        anneal = AnnealConfig(
            betas=(1.0, 4.0),
            n_warmup=2,
            n_samples=4,
            steps_per_sample=1,
        )
    else:
        problem = PottsChainProblem(n_sites=8, n_categories=3)
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
            n_warmup=8,
            n_samples=8,
            steps_per_sample=1,
        )
    base_seed = 7
    loop = WeightSweepLoop(
        backend=problem.make_backend(),
        anneal=anneal,
        seed=base_seed,
        weights=weights,
        n_obj=2,
    )
    result = loop.run()
    _exact_x, exact = problem.enumerate_front()
    return build_measured_record(
        result,
        anneal=anneal,
        base_seed=base_seed,
        issue="4",
        phase="1",
        problem="potts_chain",
        backend="thrml_potts",
        front_path=front_path,
        reference=exact,
        notes=(
            "GD and coverage compare the non-dominated archive to "
            "PottsChainProblem.enumerate_front (minimization). "
            "eval_budget counts recorded samples. "
            "Fidelity metrics are the in-repo harness (issue 6). "
            "This row is the categorical Potts sampler. "
            "The domain-wall Ising image is a separate run (issue 10)."
        ),
    )


def measure_domain_wall(*, smoke: bool, front_path: Path) -> dict:
    """Sample the Potts chain through its domain-wall Ising image.

    The reference front is ``PottsChainProblem.enumerate_front``. Recorded
    rows are decoded categorical states. Invalid thermometers are excluded
    from the archive and counted in ``notes``. The run is a THRML
    simulation, not a Z1 execution and not a ``codon_opt`` reproduction.
    """
    if smoke:
        problem = PottsChainProblem(n_sites=6, n_categories=3)
        weights = np.array([[0.5, 0.5], [0.8, 0.2]], dtype=np.float64)
        anneal = AnnealConfig(
            betas=(1.0, 4.0),
            n_warmup=2,
            n_samples=4,
            steps_per_sample=1,
        )
    else:
        problem = PottsChainProblem(n_sites=8, n_categories=3)
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
            n_warmup=8,
            n_samples=8,
            steps_per_sample=1,
        )
    base_seed = 11
    backend = problem.make_domain_wall_backend()
    kind = backend.program_kind(weights[0])
    loop = WeightSweepLoop(
        backend=backend,
        anneal=anneal,
        seed=base_seed,
        weights=weights,
        n_obj=2,
    )
    result = loop.run()
    _exact_x, exact = problem.enumerate_front()
    n_invalid = int(sum(row.n_invalid for row in result.per_weight))
    record = build_measured_record(
        result,
        anneal=anneal,
        base_seed=base_seed,
        issue="10",
        phase="optional",
        problem="potts_chain",
        backend="thrml_domain_wall",
        front_path=front_path,
        reference=exact,
        notes=(
            "Domain-wall Ising image of PottsChainProblem (THRML example 03). "
            f"Sampler program is {kind}. "
            "GD and coverage compare decoded feasible states to "
            "PottsChainProblem.enumerate_front (minimization). "
            f"Invalid thermometers excluded from the archive: {n_invalid}. "
            "eval_budget counts scored categorical rows. "
            "THRML simulation only; not a Z1 run and not codon_opt."
        ),
    )
    return record
