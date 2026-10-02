"""Markdown for the continuous ExactEw yardstick.

IGD cells in the Extropic column are the ``igd=`` text printed by
``unsga3-bend/ab/igd_vs_pymoo.py``. This module does not compute IGD.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Literal, Never

from unsga3_extropic.problems.continuous import OracleName

from published_multiseed import PUBLISHED_IGD, PUBLISHED_MEDIANS, PUBLISHED_SOURCE

ProblemName = OracleName


def _unreachable(name: Never) -> Never:
    raise AssertionError(f"unhandled oracle problem {name!r}")


def igd_command(python: str, script: str, front: str, problem: ProblemName) -> list[str]:
    """Argv matching the Bend yardstick commands for one front file."""
    match problem:
        case "zdt1":
            return [
                python,
                script,
                "--front",
                front,
                "--problem",
                "zdt1",
                "--pf-points",
                "500",
            ]
        case "zdt2":
            return [
                python,
                script,
                "--front",
                front,
                "--problem",
                "zdt2",
                "--pf-points",
                "500",
                "--partitions",
                "12",
            ]
        case "dtlz2":
            return [
                python,
                script,
                "--front",
                front,
                "--problem",
                "dtlz2",
                "--partitions",
                "12",
            ]
        case _ as other:
            _unreachable(other)


def parse_igd_stdout(text: str) -> dict[str, str]:
    """Map ``key=value`` lines. Does not invent an ``igd`` entry."""
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


@dataclass(frozen=True)
class OracleIgdRow:
    """One seed scored by the external IGD script."""

    problem: str
    seed: int
    igd_text: str
    front_rows: str
    pf_rows: str
    pf_source: str
    objective_evals: int
    recorded_samples: int
    nd_count: int


def _median_text(values: list[str]) -> str:
    """Middle printed cell when sorted by value. Odd counts only."""
    ordered = sorted(values, key=float)
    if len(ordered) % 2 == 0:
        raise ValueError("median text is defined for an odd number of igd cells")
    return ordered[len(ordered) // 2]


def _ratio(extropic: str, published: str) -> str:
    denom = float(published)
    if denom == 0.0:
        return "—"
    return f"{float(extropic) / denom:.3f}"


def render_oracle_results(
    rows: list[OracleIgdRow],
    *,
    generated_at: str,
    git_sha: str,
    partial: bool,
) -> str:
    """ORACLE-style table. Extropic IGD text is copied from ``igd=`` lines."""
    by_problem: dict[str, list[OracleIgdRow]] = {"zdt1": [], "zdt2": [], "dtlz2": []}
    for row in rows:
        if row.problem not in by_problem:
            raise ValueError(f"unknown problem {row.problem}")
        by_problem[row.problem].append(row)
    for problem, group in by_problem.items():
        seeds = [row.seed for row in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seeds for {problem}")

    status = "Partial invocation." if partial else "Full seeds 1–15 on ZDT1, ZDT2, and DTLZ2."
    lines: list[str] = [
        "# Continuous ExactEw oracle results",
        "",
        f"Generated {generated_at}. `git_sha` at record time: `{git_sha}`.",
        "",
        status,
        "",
        "Sampler: **NumPy ExactEw** (`ExactEwContinuousBackend`). One-coordinate truncated-normal Metropolis (`step_scale=0.1`) on exact \\(E_w = w \\cdot f(x)\\), then the weight-sweep archive. **Not THRML-native.** Not an Ising model, not a Potts factor, and not a fitted surrogate. Not U-NSGA-III: no SBX, no polynomial mutation, no generational population, tournament not applicable. The classical continuous U-NSGA-III column is a separate file, `benchmarks/ORACLE_UNSGA3_RESULTS.md`.",
        "",
        "Bend and C# columns are the published PymooCompatible cells. They were not re-run here.",
        f"Source: {PUBLISHED_SOURCE}.",
        "",
        "Extropic IGD is only the `igd=` line from `unsga3-bend/ab/igd_vs_pymoo.py`. The in-repo generational distance is a different formula and is left null on these records.",
        "",
        "Reference fronts required by that script: ZDT1 and ZDT2 analytic PF, 500 points; DTLZ2 Das–Dennis at partitions 12 (91 points). The pymoo default ~136-point DTLZ2 sample is not used.",
        "",
        "Objective evaluations are initial state plus every proposal. `eval_budget` in the JSONL counts recorded archive rows (warmup excluded). The Bend yardstick these budgets track is `pop * gens` (ZDT1 5200, ZDT2 13000, DTLZ2 13800). DTLZ2 uses 91 Das–Dennis directions, so 13800 does not divide evenly; the ladder spends 13741 objective calls.",
        "",
        "## How the Extropic column was scored",
        "",
        "```bash",
        "python3 ab/igd_vs_pymoo.py --front EXTROPIC.csv --problem zdt1 --pf-points 500",
        "python3 ab/igd_vs_pymoo.py --front EXTROPIC.csv --problem zdt2 --pf-points 500 --partitions 12",
        "python3 ab/igd_vs_pymoo.py --front EXTROPIC.csv --problem dtlz2 --partitions 12",
        "```",
        "",
        "Front CSVs are the unique non-dominated archive, one objective vector per line, no header.",
        "",
        "## Medians",
        "",
        "| Problem | Seeds scored | median Bend (published) | median C# (published) | median Extropic (`igd=`) | Extropic objective evals | yardstick pop×gens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    yardstick = {"zdt1": 52 * 100, "zdt2": 52 * 250, "dtlz2": 92 * 150}
    order: tuple[ProblemName, ...] = ("zdt1", "zdt2", "dtlz2")
    for problem in order:
        group = sorted(by_problem[problem], key=lambda row: row.seed)
        if not group:
            lines.append(
                f"| {problem} | 0 | {PUBLISHED_MEDIANS[problem]['bend']} | {PUBLISHED_MEDIANS[problem]['csharp']} | — | — | {yardstick[problem]} |"
            )
            continue
        median = _median_text([row.igd_text for row in group])
        evals = group[0].objective_evals
        lines.append(
            f"| {problem} | {len(group)} | {PUBLISHED_MEDIANS[problem]['bend']} | "
            f"{PUBLISHED_MEDIANS[problem]['csharp']} | {median} | {evals} | {yardstick[problem]} |"
        )

    lines.extend(
        [
            "",
            "The published median column is the summary in the Bend document, not a recomputation. The Extropic median is the middle `igd=` cell after sorting the scored seeds for that problem.",
            "",
        ]
    )
    titles = {
        "zdt1": "ZDT1 n=30, partitions=12, Bend pop=52 gens=100",
        "zdt2": "ZDT2 n=30, partitions=12, Bend pop=52 gens=250",
        "dtlz2": "DTLZ2 M=3 k=10 (n=12), partitions=12, Bend pop=92 gens=150",
    }
    for problem in order:
        group = sorted(by_problem[problem], key=lambda row: row.seed)
        lines.extend(
            [
                f"## {titles[problem]}",
                "",
                "| seed | Bend IGD | C# IGD | Extropic IGD | Extropic/Bend | Extropic n | objective evals | recorded samples | pf_rows | pf_source |",
                "|-----:|---------:|-------:|-------------:|--------------:|-----------:|----------------:|-----------------:|--------:|-----------|",
            ]
        )
        if not group:
            lines.append("| — | — | — | — | — | — | — | — | — | not scored |")
            lines.append("")
            continue
        published = PUBLISHED_IGD[problem]
        for row in group:
            bend = published["bend"][row.seed]
            csharp = published["csharp"][row.seed]
            lines.append(
                f"| {row.seed} | {bend} | {csharp} | {row.igd_text} | {_ratio(row.igd_text, bend)} | "
                f"{row.front_rows} | {row.objective_evals} | {row.recorded_samples} | {row.pf_rows} | {row.pf_source} |"
            )
        lines.append("")
        lines.append(
            f"Seeds scored: {', '.join(str(row.seed) for row in group)}. "
            f"Extropic median of those `igd=` cells: {_median_text([row.igd_text for row in group])}."
        )
        lines.append("")
    lines.extend(
        [
            "## Reading the Extropic column",
            "",
            "A larger IGD is farther from the analytic front. This sampler does not share Bend's SBX, polynomial mutation, or population. Matching `pop * gens` spends a similar number of objective calls; it does not reproduce the U-NSGA-III trajectory. `Extropic n` is `front_rows` from the IGD script (unique non-dominated rows), which is not forced to equal the Bend population.",
            "",
        ]
    )
    return "\n".join(lines)


def median_matches_published(problem: str, stack: Literal["bend", "csharp"]) -> bool:
    """The documented summary median is the middle published cell."""
    cells = [PUBLISHED_IGD[problem][stack][seed] for seed in range(1, 16)]
    middle = _median_text(cells)
    # The summary prints six digits. The middle cell is already that text
    # when the published median is one of the seeds, which it is here.
    summary = PUBLISHED_MEDIANS[problem][stack]
    if middle == summary:
        return True
    return f"{statistics.median(float(cell) for cell in cells):.6f}" == summary
