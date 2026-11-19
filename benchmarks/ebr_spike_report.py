#!/usr/bin/env python3
"""Render the tables in ``benchmarks/EBR_SPIKE_RESULTS.md`` from ``records/ebr_spike.json``.

    python benchmarks/ebr_spike_report.py            # prints markdown tables
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "benchmarks" / "records" / "ebr_spike.json"

ORDER = (
    "baseline",
    "select",
    "plateau(w=8)",
    "plateau(w=8)+select",
    "plateau(w=4)",
    "plateau(w=4)+select",
    "thrml",
    "thrml+select",
)


def _fmt(v: float | None, nd: int = 3) -> str:
    if v is None:
        return "–"
    if abs(v) < 1e-12:
        v = 0.0
    return f"{v:.{nd}f}"


def render(data: dict) -> str:
    rows = data["rows"]
    out: list[str] = []
    for task in data["meta"]["tasks"]:
        out.append(f"### {task}\n")
        out.append(
            "| variant | backend | seed | recorded rows | energy evals | ND | HV ratio | GD | coverage | argmin hits |"
        )
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        sub = [r for r in rows if r["task"] == task]
        sub.sort(key=lambda r: (ORDER.index(r["variant"]), r["seed"]))
        for r in sub:
            out.append(
                f"| {r['variant']} | {r['backend']} | {r['seed']} | {r['recorded_rows']} | "
                f"{r['energy_evals']} | {r['nd_count']} | {_fmt(r['hv_ratio'])} | "
                f"{_fmt(r['gd'])} | {_fmt(r['coverage'])} | {r['argmin_hits']}/{r['n_weights']} |"
            )
        out.append("")
        out.append("Means over seeds 7, 11, 19:\n")
        out.append("| variant | recorded rows | energy evals | HV ratio | GD | coverage | argmin hits |")
        out.append("|---|---|---|---|---|---|---|")
        for name in ORDER:
            g = [r for r in sub if r["variant"] == name]
            if not g:
                continue
            n = len(g)
            out.append(
                f"| {name} | {sum(r['recorded_rows'] for r in g) / n:.1f} | "
                f"{sum(r['energy_evals'] for r in g) / n:.1f} | "
                f"{_fmt(sum(r['hv_ratio'] for r in g) / n)} | "
                f"{_fmt(sum(r['gd'] for r in g) / n)} | "
                f"{_fmt(sum(r['coverage'] for r in g) / n)} | "
                f"{sum(r['argmin_hits'] for r in g)}/{sum(r['n_weights'] for r in g)} |"
            )
        post = [r for r in sub if "thrml_plateau_posthoc_samples" in r]
        if post:
            out.append("")
            out.append(
                "THRML plateau rule scored after the fact (window 8, tol 0; no THRML work saved): "
                + ", ".join(
                    f"seed {r['seed']}: {r['thrml_plateau_posthoc_samples']} of {r['recorded_rows']} rows"
                    for r in post
                )
                + "."
            )
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0]) if args else DEFAULT
    print(render(json.loads(path.read_text(encoding="utf-8"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
