#!/usr/bin/env python3
"""Deep multi-seed benchmarks. Not the CI smoke.

From the repository root, after ``pip install -e ".[dev,cpu]"``
(and ``".[dev,cpu,torx]"`` if the PSWAP rate check should run)::

    export JAX_PLATFORMS=cpu
    python benchmarks/run_deep.py

Appends one JSONL row per backend and seed to ``benchmarks/records/runs.jsonl``
with ``profile=deep`` in ``notes``, and writes ``benchmarks/DEEP_RESULTS.md``.
The Torx block is a stay/swap rate at a larger ``n`` than the smoke. It is
not a minimization front, so it is stored in
``benchmarks/records/torx_pswap_deep.json`` instead of the JSONL log.

Budgets live in ``unsga3_extropic.schedules``. This script does not search
for a beta schedule. Each seed runs in its own process so JAX compilations
from the previous seed are released. A seed already logged with
``profile=deep`` and this schedule is skipped, which makes a re-run resume.
The samples are THRML simulations (and one NumPy exact-E_w comparator).
They are not Z1 measurements and they do not call Thermalizers.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# One OS process per seed. These must be set before JAX is imported.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmarks"))

from deep_report import render_deep_results  # noqa: E402
from unsga3_extropic.results import (  # noqa: E402
    append_run_record,
    current_git_sha,
    measure_codon_exact_ew,
    measure_codon_ising,
    measure_domain_wall,
    measure_potts_chain,
    read_run_records,
    utc_now_iso,
)
from unsga3_extropic.schedules import (  # noqa: E402
    DEEP_SCHEDULE,
    TORX_DEEP_ABS_TOLERANCE,
    TORX_DEEP_N_SAMPLES,
    TORX_DEEP_P_SWAP,
    TORX_DEEP_SEEDS,
)
from unsga3_extropic.torx_circuit import (  # noqa: E402
    DOCUMENTED_INITIAL_STATE,
    TorxPswapCircuit,
)

SEARCH_CHOICES = ("codon", "potts", "domain_wall", "exact_ew", "torx")
_KIND_BACKEND = {
    "codon": "thrml_ising",
    "potts": "thrml_potts",
    "domain_wall": "thrml_domain_wall",
    "exact_ew": "exact_ew",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the deep benchmark profile.")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmarks" / "records" / "runs.jsonl",
        help="JSONL path to append search records to.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "benchmarks" / "DEEP_RESULTS.md",
        help="Markdown summary to write.",
    )
    parser.add_argument(
        "--torx-out",
        type=Path,
        default=ROOT / "benchmarks" / "records" / "torx_pswap_deep.json",
        help="JSON path for the Torx rate check.",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=SEARCH_CHOICES,
        default=None,
        help="Run one backend. Repeat to run several. Default is all of them.",
    )
    parser.add_argument(
        "--skip-torx",
        action="store_true",
        help="Skip the PSWAP rate check.",
    )
    parser.add_argument("--worker", choices=SEARCH_CHOICES, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--front", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _selected(args: argparse.Namespace) -> list[str]:
    chosen = list(SEARCH_CHOICES if args.only is None else args.only)
    if args.skip_torx:
        chosen = [name for name in chosen if name != "torx"]
    # Preserve the documented order when --only is repeated out of order.
    return [name for name in SEARCH_CHOICES if name in chosen]


def _front_path(backend: str, seed: int) -> Path:
    directory = ROOT / "benchmarks" / "records" / "artifacts"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"deep-{backend}-seed{seed}.npz"


def _stamp(record: dict, elapsed_s: float) -> dict:
    record["notes"] = record["notes"].rstrip() + f" elapsed_s={elapsed_s:.3f}."
    return record


def recorded_deep_seed(records: list[dict], kind: str, seed: int) -> dict | None:
    """Return an existing deep row for this backend, seed, and schedule."""
    backend = _KIND_BACKEND[kind]
    marker = f"base_seed={seed}."
    betas = [float(beta) for beta in DEEP_SCHEDULE.betas]
    for record in records:
        if record.get("backend") != backend:
            continue
        notes = str(record.get("notes", ""))
        if not notes.startswith("profile=deep."):
            continue
        if marker not in notes:
            continue
        schedule = record.get("schedule") or {}
        if schedule.get("n_weights") != DEEP_SCHEDULE.n_weights():
            continue
        if schedule.get("n_samples") != DEEP_SCHEDULE.n_samples:
            continue
        if schedule.get("n_warmup") != DEEP_SCHEDULE.n_warmup:
            continue
        if schedule.get("steps_per_sample") != DEEP_SCHEDULE.steps_per_sample:
            continue
        if [float(beta) for beta in schedule.get("betas", [])] != betas:
            continue
        return record
    return None


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _run_seed(kind: str, seed: int, front: Path | None = None) -> dict:
    _log(f"=== deep {kind} base_seed={seed} ===")
    started = time.perf_counter()
    front_path = front if front is not None else _front_path(_KIND_BACKEND[kind], seed)
    if kind == "codon":
        record = measure_codon_ising(
            smoke=False,
            profile="deep",
            base_seed=seed,
            front_path=front_path,
        )
    elif kind == "potts":
        record = measure_potts_chain(
            smoke=False,
            profile="deep",
            base_seed=seed,
            front_path=front_path,
        )
    elif kind == "domain_wall":
        record = measure_domain_wall(
            smoke=False,
            profile="deep",
            base_seed=seed,
            front_path=front_path,
        )
    elif kind == "exact_ew":
        record = measure_codon_exact_ew(
            base_seed=seed,
            front_path=front_path,
        )
    else:
        raise AssertionError(f"unhandled search backend: {kind!r}")
    elapsed = time.perf_counter() - started
    _stamp(record, elapsed)
    metrics = record["metrics"]
    _log(
        "  "
        f"backend={record['backend']} eval_budget={record['eval_budget']} "
        f"nd={metrics['nd_count']} hv={metrics['hypervolume_2d']} "
        f"gd={metrics['generational_distance']} coverage={metrics['coverage']} "
        f"elapsed_s={elapsed:.3f}"
    )
    return record


def _worker_env() -> dict[str, str]:
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    return env


def _run_worker(kind: str, seed: int | None = None) -> dict:
    """Run one seed, or the Torx block, in a child process."""
    cmd = [sys.executable, str(ROOT / "benchmarks" / "run_deep.py"), "--worker", kind]
    if kind != "torx":
        if seed is None:
            raise AssertionError("search workers need a seed")
        front = _front_path(_KIND_BACKEND[kind], seed)
        cmd.extend(["--seed", str(seed), "--front", str(front)])
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=_worker_env(),
        capture_output=True,
        text=True,
    )
    if proc.stderr:
        sys.stderr.write(proc.stderr)
        if not proc.stderr.endswith("\n"):
            sys.stderr.write("\n")
    # Torx returns 1 when the rate misses the sanity bound, after printing JSON.
    if proc.returncode != 0 and not (kind == "torx" and proc.returncode == 1):
        sys.stderr.write(proc.stdout)
        raise SystemExit(proc.returncode)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if len(lines) != 1:
        raise RuntimeError(f"worker {kind} returned {len(lines)} JSON lines")
    return json.loads(lines[0])


def _run_search(kinds: list[str], out: Path, existing: list[dict]) -> list[dict]:
    records: list[dict] = []
    for kind in kinds:
        if kind == "torx":
            continue
        for seed in DEEP_SCHEDULE.base_seeds:
            recorded = recorded_deep_seed(existing, kind, seed)
            if recorded is not None:
                _log(f"resume {kind} base_seed={seed} from {out}")
                records.append(recorded)
                continue
            record = _run_worker(kind, seed)
            append_run_record(out, record)
            _log(f"appended {record['backend']} seed {record['seeds'][0]} -> {out}")
            records.append(record)
    return records


def _run_torx() -> tuple[dict, bool]:
    _log(
        f"=== deep torx PSWAP n={TORX_DEEP_N_SAMPLES} seeds={list(TORX_DEEP_SEEDS)} ==="
    )
    draws: list[dict] = []
    within = True
    circuit = TorxPswapCircuit()
    for seed in TORX_DEEP_SEEDS:
        started = time.perf_counter()
        draw = circuit.sample(
            p_swap=TORX_DEEP_P_SWAP,
            n_samples=TORX_DEEP_N_SAMPLES,
            seed=seed,
            initial_state=DOCUMENTED_INITIAL_STATE,
        )
        elapsed = time.perf_counter() - started
        error = abs(draw.swap_rate - TORX_DEEP_P_SWAP)
        ok = error < TORX_DEEP_ABS_TOLERANCE
        within = within and ok
        draws.append(
            {
                "seed": seed,
                "stay_rate": draw.stay_rate,
                "swap_rate": draw.swap_rate,
                "abs_swap_error": error,
                "elapsed_s": round(elapsed, 3),
                "within_bound": ok,
            }
        )
        _log(
            f"  seed={seed} stay={draw.stay_rate:.4f} swap={draw.swap_rate:.4f} "
            f"|swap-p|={error:.4f} elapsed_s={elapsed:.3f}"
        )
    payload = {
        "profile": "deep",
        "circuit": "TorxPswapCircuit",
        "p_swap": TORX_DEEP_P_SWAP,
        "n_samples": TORX_DEEP_N_SAMPLES,
        "seeds": list(TORX_DEEP_SEEDS),
        "initial_state": list(DOCUMENTED_INITIAL_STATE),
        "abs_tolerance": TORX_DEEP_ABS_TOLERANCE,
        "git_sha": current_git_sha(),
        "date": utc_now_iso(),
        "draws": draws,
        "notes": (
            "profile=deep. Torx PSWAP stay/swap rate check. "
            "Not a minimization front, so it is not a row in runs.jsonl. "
            "Not a THRML program, not a Z1 run, and not Thermalizers."
        ),
    }
    return payload, within


def _worker_main(args: argparse.Namespace) -> int:
    if args.worker == "torx":
        payload, ok = _run_torx()
        sys.stdout.write(json.dumps({"payload": payload, "ok": ok}) + "\n")
        return 0 if ok else 1
    if args.seed is None or args.front is None:
        print("a search worker needs --seed and --front", file=sys.stderr)
        return 2
    record = _run_seed(args.worker, args.seed, args.front)
    sys.stdout.write(json.dumps(record) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.worker is not None:
        return _worker_main(args)
    kinds = _selected(args)
    if not kinds:
        print("no backends selected", file=sys.stderr)
        return 2
    prior: list[dict] = []
    if args.out.is_file():
        prior = read_run_records(args.out)
    search_kinds = [kind for kind in kinds if kind != "torx"]
    records = _run_search(search_kinds, args.out, prior)
    torx_payload: dict | None = None
    torx_ok = True
    try:
        if "torx" in kinds:
            torx_result = _run_worker("torx")
            torx_payload = torx_result["payload"]
            torx_ok = bool(torx_result["ok"])
            args.torx_out.parent.mkdir(parents=True, exist_ok=True)
            args.torx_out.write_text(
                json.dumps(torx_payload, indent=2) + "\n",
                encoding="utf-8",
            )
            _log(f"wrote {args.torx_out}")
    finally:
        partial = None if args.only is None and not args.skip_torx else ",".join(kinds)
        summary = render_deep_results(
            deep_records=records,
            prior_records=prior,
            torx=torx_payload,
            generated_at=utc_now_iso(),
            partial=partial,
        )
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(summary, encoding="utf-8")
        print(f"wrote {args.summary}")
    if not torx_ok:
        print(
            f"Torx swap rate left the ±{TORX_DEEP_ABS_TOLERANCE} bound",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
