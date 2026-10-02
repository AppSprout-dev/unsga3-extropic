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
for a beta schedule. The samples are THRML simulations (and one NumPy
exact-E_w comparator). They are not Z1 measurements and they do not call
Thermalizers.
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


def _run_seed(kind: str, seed: int) -> dict:
    print(f"=== deep {kind} base_seed={seed} ===", flush=True)
    started = time.perf_counter()
    if kind == "codon":
        record = measure_codon_ising(
            smoke=False,
            profile="deep",
            base_seed=seed,
            front_path=_front_path("thrml_ising", seed),
        )
    elif kind == "potts":
        record = measure_potts_chain(
            smoke=False,
            profile="deep",
            base_seed=seed,
            front_path=_front_path("thrml_potts", seed),
        )
    elif kind == "domain_wall":
        record = measure_domain_wall(
            smoke=False,
            profile="deep",
            base_seed=seed,
            front_path=_front_path("thrml_domain_wall", seed),
        )
    elif kind == "exact_ew":
        record = measure_codon_exact_ew(
            base_seed=seed,
            front_path=_front_path("exact_ew", seed),
        )
    else:
        raise AssertionError(f"unhandled search backend: {kind!r}")
    elapsed = time.perf_counter() - started
    _stamp(record, elapsed)
    metrics = record["metrics"]
    print(
        "  "
        f"backend={record['backend']} eval_budget={record['eval_budget']} "
        f"nd={metrics['nd_count']} hv={metrics['hypervolume_2d']} "
        f"gd={metrics['generational_distance']} coverage={metrics['coverage']} "
        f"elapsed_s={elapsed:.3f}",
        flush=True,
    )
    return record


def _run_search(kinds: list[str], out: Path) -> list[dict]:
    records: list[dict] = []
    for kind in kinds:
        if kind == "torx":
            continue
        for seed in DEEP_SCHEDULE.base_seeds:
            record = _run_seed(kind, seed)
            append_run_record(out, record)
            print(f"appended {record['backend']} seed {record['seeds'][0]} -> {out}")
            records.append(record)
    return records


def _run_torx() -> tuple[dict, bool]:
    print(
        f"=== deep torx PSWAP n={TORX_DEEP_N_SAMPLES} seeds={list(TORX_DEEP_SEEDS)} ===",
        flush=True,
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
        print(
            f"  seed={seed} stay={draw.stay_rate:.4f} swap={draw.swap_rate:.4f} "
            f"|swap-p|={error:.4f} elapsed_s={elapsed:.3f}",
            flush=True,
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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    kinds = _selected(args)
    if not kinds:
        print("no backends selected", file=sys.stderr)
        return 2
    prior: list[dict] = []
    if args.out.is_file():
        prior = read_run_records(args.out)
    search_kinds = [kind for kind in kinds if kind != "torx"]
    records = _run_search(search_kinds, args.out)
    torx_payload: dict | None = None
    torx_ok = True
    try:
        if "torx" in kinds:
            torx_payload, torx_ok = _run_torx()
            args.torx_out.parent.mkdir(parents=True, exist_ok=True)
            args.torx_out.write_text(
                json.dumps(torx_payload, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"wrote {args.torx_out}")
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
