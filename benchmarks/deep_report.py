"""Markdown summary of one deep-benchmark invocation.

The JSONL log stays append-only. This renderer turns the records just
measured, plus earlier unlabeled rows, into ``benchmarks/DEEP_RESULTS.md``.
It does not sample.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unsga3_extropic.schedules import (  # noqa: E402
    CODON_INSTANCE_LABEL,
    CODON_SMOKE,
    DEEP_SCHEDULE,
    POTTS_SMOKE,
    TORX_DEEP_ABS_TOLERANCE,
    TORX_DEEP_N_SAMPLES,
    TORX_DEEP_P_SWAP,
    TORX_DEEP_SEEDS,
    potts_instance_label,
)

_NICHE = re.compile(
    r"Closer-in-niche survival kept (\d+) of (\d+) non-dominated rows"
)
_ELAPSED = re.compile(r"elapsed_s=([0-9]+(?:\.[0-9]+)?)")
_INVALID = re.compile(r"Invalid thermometers excluded from the archive: (\d+)")
_EXACT = re.compile(r"Reference front size (\d+)")
_INSTANCE = re.compile(r"instance=([^.]+)\.")
_BASE_SEED = re.compile(r"base_seed=(\d+)")
_REPLICATE = re.compile(r"replicate=(\d+/\d+)")


def _num(value: object, *, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _first(pattern: re.Pattern[str], text: str, group: int = 1) -> str | None:
    match = pattern.search(text)
    if match is None:
        return None
    return match.group(group)


def _triple(values: list[float], *, digits: int) -> str:
    if not values:
        return "—"
    mean = sum(values) / len(values)
    fmt = f"{{:.{digits}f}}"
    return f"{fmt.format(min(values))} / {fmt.format(mean)} / {fmt.format(max(values))}"


def prior_label(record: dict) -> str:
    """Classify a JSONL row that is not part of the invocation being summarized."""
    notes = str(record.get("notes", ""))
    if "profile=deep" in notes:
        return "deep"
    if "profile=smoke" in notes:
        return "smoke"
    if "profile=default" in notes:
        return "default"
    schedule = record.get("schedule") or {}
    budget = record.get("eval_budget")
    n_weights = schedule.get("n_weights")
    if n_weights == 2 and budget in (CODON_SMOKE.recorded_per_seed(), POTTS_SMOKE.recorded_per_seed()):
        return "smoke (historical)"
    return "other"


def _budget_table() -> str:
    deep = DEEP_SCHEDULE
    codon = CODON_SMOKE
    potts = POTTS_SMOKE
    rows = (
        ("weights", str(codon.n_weights()), str(potts.n_weights()), str(deep.n_weights())),
        ("betas", str(len(codon.betas)), str(len(potts.betas)), str(len(deep.betas))),
        ("beta values", _beta_list(codon.betas), _beta_list(potts.betas), _beta_list(deep.betas)),
        ("warmup", str(codon.n_warmup), str(potts.n_warmup), str(deep.n_warmup)),
        ("samples per beta", str(codon.n_samples), str(potts.n_samples), str(deep.n_samples)),
        (
            "steps between samples",
            str(codon.steps_per_sample),
            str(potts.steps_per_sample),
            str(deep.steps_per_sample),
        ),
        ("base seeds", str(len(codon.base_seeds)), str(len(potts.base_seeds)), str(len(deep.base_seeds))),
        (
            "recorded rows / seed",
            str(codon.recorded_per_seed()),
            str(potts.recorded_per_seed()),
            str(deep.recorded_per_seed()),
        ),
        (
            "recorded rows, all seeds",
            str(codon.recorded_total()),
            str(potts.recorded_total()),
            str(deep.recorded_total()),
        ),
    )
    lines = [
        "| | codon smoke | potts / domain-wall smoke | deep (each search backend) |",
        "|---|---:|---:|---:|",
    ]
    for name, smoke_c, smoke_p, deep_cell in rows:
        lines.append(f"| {name} | {smoke_c} | {smoke_p} | {deep_cell} |")
    return "\n".join(lines)


def _beta_list(betas: tuple[float, ...]) -> str:
    return ", ".join(str(beta) for beta in betas)


def _fallback_instance(record: dict) -> str:
    problem = record.get("problem")
    backend = record.get("backend")
    schedule = record.get("schedule") or {}
    n_weights = schedule.get("n_weights")
    if problem == "codon_ising":
        return CODON_INSTANCE_LABEL
    if backend in ("thrml_potts", "thrml_domain_wall") and n_weights == 2:
        return potts_instance_label("smoke")
    if backend in ("thrml_potts", "thrml_domain_wall") and n_weights == 5:
        return potts_instance_label("default")
    return "—"


def _record_cells(record: dict) -> list[str]:
    notes = str(record.get("notes", ""))
    metrics = record["metrics"]
    schedule = record["schedule"]
    niche = _first(_NICHE, notes)
    niche_of = _first(_NICHE, notes, group=2)
    niche_cell = "—" if niche is None else f"{niche}/{niche_of}"
    invalid = _first(_INVALID, notes)
    if invalid is None and record.get("backend") != "thrml_domain_wall":
        invalid_cell = "—"
    else:
        invalid_cell = invalid if invalid is not None else "—"
    base_seed = _first(_BASE_SEED, notes)
    if base_seed is None and record.get("seeds"):
        base_seed = str(record["seeds"][0])
    replicate = _first(_REPLICATE, notes) or "—"
    instance = _first(_INSTANCE, notes) or _fallback_instance(record)
    elapsed = _first(_ELAPSED, notes) or "—"
    exact = _first(_EXACT, notes) or "—"
    sha = str(record.get("git_sha", ""))
    short = sha[:12] if sha else "—"
    return [
        str(record.get("backend", "")),
        str(record.get("problem", "")),
        instance,
        base_seed or "—",
        replicate,
        str(schedule["n_weights"]),
        str(len(schedule["betas"])),
        str(schedule["n_warmup"]),
        str(schedule["n_samples"]),
        str(schedule["steps_per_sample"]),
        str(record["eval_budget"]),
        str(metrics["nd_count"]),
        niche_cell,
        _num(metrics["hypervolume_2d"]),
        _num(metrics["generational_distance"]),
        _num(metrics["coverage"]),
        exact,
        invalid_cell,
        elapsed,
        short,
    ]


_HEADER = (
    "| backend | problem | instance | base seed | replicate | weights | betas | "
    "warmup | samples | steps | eval_budget | ND | niche | HV | GD | coverage | "
    "exact ND | invalid | elapsed_s | git_sha |"
)
_RULE = (
    "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"
    "---:|---:|---:|---|"
)


def _markdown_table(records: list[dict]) -> str:
    lines = [_HEADER, _RULE]
    for record in records:
        cells = _record_cells(record)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _aggregates(records: list[dict]) -> str:
    by_backend: dict[str, list[dict]] = {}
    for record in records:
        by_backend.setdefault(str(record["backend"]), []).append(record)
    lines = [
        "| backend | seeds | ND min/mean/max | HV min/mean/max | GD min/mean/max | coverage min/mean/max | invalid sum | elapsed sum s |",
        "|---|---:|---|---|---|---|---:|---:|",
    ]
    for backend in sorted(by_backend):
        group = by_backend[backend]
        nd = [float(row["metrics"]["nd_count"]) for row in group]
        hv = [
            float(row["metrics"]["hypervolume_2d"])
            for row in group
            if row["metrics"]["hypervolume_2d"] is not None
        ]
        gd = [
            float(row["metrics"]["generational_distance"])
            for row in group
            if row["metrics"]["generational_distance"] is not None
        ]
        cov = [
            float(row["metrics"]["coverage"])
            for row in group
            if row["metrics"]["coverage"] is not None
        ]
        invalid_sum = 0
        saw_invalid = False
        elapsed_sum = 0.0
        saw_elapsed = False
        for row in group:
            notes = str(row.get("notes", ""))
            invalid = _first(_INVALID, notes)
            if invalid is not None:
                saw_invalid = True
                invalid_sum += int(invalid)
            elapsed = _first(_ELAPSED, notes)
            if elapsed is not None:
                saw_elapsed = True
                elapsed_sum += float(elapsed)
        lines.append(
            "| "
            + " | ".join(
                [
                    backend,
                    str(len(group)),
                    _triple(nd, digits=2),
                    _triple(hv, digits=4),
                    _triple(gd, digits=4),
                    _triple(cov, digits=4),
                    str(invalid_sum) if saw_invalid else "—",
                    f"{elapsed_sum:.3f}" if saw_elapsed else "—",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _torx_section(torx: dict | None) -> str:
    if torx is None:
        return (
            "## Torx PSWAP\n\n"
            "This invocation did not sample Torx. "
            f"A full deep run draws `n={TORX_DEEP_N_SAMPLES}` at seeds {list(TORX_DEEP_SEEDS)} "
            f"with `p={TORX_DEEP_P_SWAP}`.\n"
        )
    lines = [
        "## Torx PSWAP rate check",
        "",
        "Separate from the search log. These rates are not a minimization front, "
        "so they are not appended to `benchmarks/records/runs.jsonl`. "
        "The circuit is Torx `PSWAP` on `BranchingSimulator`. "
        "It is not a THRML program, not a Z1 run, and not Thermalizers.",
        "",
        f"- p_swap: {torx['p_swap']}",
        f"- n_samples: {torx['n_samples']} (smoke is 20_000)",
        f"- git_sha: `{torx['git_sha']}`",
        f"- sanity bound: absolute swap error < {TORX_DEEP_ABS_TOLERANCE}",
        "",
        "| seed | stay | swap | |swap − p| | elapsed_s | within bound |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    swaps: list[float] = []
    for draw in torx["draws"]:
        swaps.append(float(draw["swap_rate"]))
        lines.append(
            "| {seed} | {stay:.4f} | {swap:.4f} | {err:.4f} | {elapsed:.3f} | {ok} |".format(
                seed=draw["seed"],
                stay=float(draw["stay_rate"]),
                swap=float(draw["swap_rate"]),
                err=abs(float(draw["swap_rate"]) - float(torx["p_swap"])),
                elapsed=float(draw["elapsed_s"]),
                ok="yes" if draw["within_bound"] else "no",
            )
        )
    mean_swap = sum(swaps) / len(swaps) if swaps else float("nan")
    lines.append("")
    lines.append(f"Mean swap rate across {len(swaps)} seeds: {mean_swap:.4f}.")
    return "\n".join(lines) + "\n"


def render_deep_results(
    *,
    deep_records: list[dict],
    prior_records: list[dict],
    torx: dict | None,
    generated_at: str,
    partial: str | None = None,
) -> str:
    """Build the deep-results markdown for one invocation."""
    sha = deep_records[0]["git_sha"] if deep_records else "unknown"
    partial_note = ""
    if partial:
        partial_note = (
            f"\nPartial invocation (`{partial}`). "
            "This file replaces the previous summary and does not include backends that were not run.\n"
        )
    prior_lines = [row for row in prior_records if prior_label(row) != "deep"]
    body = f"""# Deep benchmark results

Generated {generated_at} by `benchmarks/run_deep.py`.
`git_sha` on each row is `HEAD` when that row was recorded (`{sha}` for this invocation).
The JSONL log is append-only; this file is the latest invocation, not a rewrite of older lines.
{partial_note}
These search rows are THRML simulations on CPU, plus a NumPy exact-E_w Metropolis comparator on the codon chain. They are not Z1 measurements and they do not use Thermalizers.

Deep uses the smoke problem instances so the exact fronts match: codon `{CODON_INSTANCE_LABEL}`; Potts and domain wall `{potts_instance_label("deep")}`. `profile=default` is a different Potts chain (`{potts_instance_label("default")}`) and is not this matrix.

`eval_budget` is recorded objective rows (`n_weights * len(betas) * n_samples`), except domain wall, where invalid thermometers are left out of the archive and the budget counts scored categorical rows. `schedule_product` in the notes is the full sample count before that filter. Niche is quota-1 closer-in-niche survival. HV, GD, and coverage use `enumerate_front` as the reference (minimization). GD is mean Euclidean distance from the run front to that reference. Coverage is the fraction of the reference front weakly dominated by the run front.

## Budget

{_budget_table()}

Deep weights are `{DEEP_SCHEDULE.weight_source}` (the endpoints are included). Smoke weights are the two fixed vectors `{CODON_SMOKE.weight_source}`. Base seeds for every deep search backend: {list(DEEP_SCHEDULE.base_seeds)}. `WeightSweepLoop` still passes `base_seed + 1009 * weight_index` into `sample_weight`; those expanded seeds are the record's `seeds` array.

## Deep matrix

{_markdown_table(deep_records) if deep_records else "_No search records in this invocation._"}

## Seed aggregates

Min / mean / max across the seeds in this invocation.

{_aggregates(deep_records) if deep_records else "_No search records in this invocation._"}

## Earlier log rows

Rows already in `benchmarks/records/runs.jsonl` before this invocation, excluding any previous `profile=deep` lines. Historical smoke rows were written before notes carried `profile=smoke`. Their budgets (24 for codon, 16 for Potts and domain wall) are the CI smokes.

{_markdown_table(prior_lines) if prior_lines else "_No earlier rows._"}

{_torx_section(torx)}
## Re-run

From the repository root, with THRML, JAX, and (for the rate check) the `torx` extra:

```bash
pip install -e ".[dev,cpu,torx]"
export JAX_PLATFORMS=cpu
python benchmarks/run_deep.py
```

That appends one JSONL line per backend and seed to `benchmarks/records/runs.jsonl`, rewrites this file, and writes `benchmarks/records/torx_pswap_deep.json`. Front arrays go to `benchmarks/records/artifacts/` and stay gitignored.

Default CI does not run this. `.github/workflows/deep.yml` is `workflow_dispatch` only. A dispatch run uploads the log and this summary as artifacts; it does not commit them.

Subset of backends, for a local check:

```bash
python benchmarks/run_deep.py --only codon
python benchmarks/run_deep.py --skip-torx
```

`--only` rewrites this summary as a partial invocation. Commit a summary only from a full run.
"""
    return body
