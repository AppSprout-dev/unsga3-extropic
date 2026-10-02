"""Markdown for the classical continuous U-NSGA-III yardstick.

IGD is the in-repo mean distance from each Pareto-front sample to the
nearest obtained point (the ``igd=`` definition). Bend and C# cells are
the published strings. ExactEw cells are read from ``ORACLE_RESULTS.md``
so that column is kept, not recomputed.
"""

from __future__ import annotations

from pathlib import Path

from published_multiseed import PUBLISHED_IGD, PUBLISHED_MEDIANS, PUBLISHED_SOURCE

EXACTEW_SOURCE = "benchmarks/ORACLE_RESULTS.md (NumPy ExactEwContinuousBackend, not U-NSGA-III)"


def median_text(values: list[str]) -> str:
    """Middle printed cell when sorted by value. Odd counts only."""
    ordered = sorted(values, key=float)
    if len(ordered) % 2 == 0:
        raise ValueError("median text is defined for an odd number of igd cells")
    return ordered[len(ordered) // 2]


def load_exactew_igd(text: str) -> dict[str, dict[int, str]]:
    """Per-seed ExactEw ``igd=`` strings from the committed oracle table."""
    current: str | None = None
    found: dict[str, dict[int, str]] = {"zdt1": {}, "zdt2": {}, "dtlz2": {}}
    for line in text.splitlines():
        if line.startswith("## ZDT1"):
            current = "zdt1"
            continue
        if line.startswith("## ZDT2"):
            current = "zdt2"
            continue
        if line.startswith("## DTLZ2"):
            current = "dtlz2"
            continue
        if line.startswith("## "):
            current = None
            continue
        if current is None or not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4 or not cells[0].isdigit():
            continue
        found[current][int(cells[0])] = cells[3]
    return found


def _ratio(obtained: str, published: str) -> str:
    denom = float(published)
    if denom == 0.0:
        return "—"
    return f"{float(obtained) / denom:.3f}"


def render_classical_results(
    rows: list[dict],
    *,
    exactew: dict[str, dict[int, str]],
    generated_at: str,
    git_sha: str,
    partial: bool,
) -> str:
    """ORACLE-style table for the classical continuous column."""
    by_problem: dict[str, list[dict]] = {"zdt1": [], "zdt2": [], "dtlz2": []}
    for row in rows:
        problem = str(row["problem"])
        if problem not in by_problem:
            raise ValueError(f"unknown problem {problem}")
        by_problem[problem].append(row)
    for problem, group in by_problem.items():
        seeds = [int(row["seed"]) for row in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seeds for {problem}")

    status = (
        "Partial invocation."
        if partial
        else "Full seeds 1–15 on ZDT1, ZDT2, and DTLZ2."
    )
    lines: list[str] = [
        "# Classical continuous U-NSGA-III oracle results",
        "",
        f"Generated {generated_at}. `git_sha` at record time: `{git_sha}`.",
        "",
        status,
        "",
        "Sampler: **classical continuous U-NSGA-III** (`unsga3_extropic.classical`). Generational population, Das–Dennis reference directions, non-dominated ranking, perpendicular association, PymooCompatible tournament (same niche: better rank, then closer perpendicular distance; otherwise a coin), SBX (η=30, p_c=1), polynomial mutation (η=20, p_m=1/n). NumPy only. **Not THRML-native.** Not an Ising model, not a Potts factor, and not Extropic sampling. `WeightSweepLoop` does not call this path. `ExactEwContinuousBackend` is unchanged.",
        "",
        "Bend and C# columns are the published PymooCompatible cells. They were not re-run here.",
        f"Source: {PUBLISHED_SOURCE}.",
        "",
        f"ExactEw cells are the committed NumPy weight-sweep column. Source: {EXACTEW_SOURCE}.",
        "",
        "IGD is the in-repo yardstick formula, the same quantity as the `igd=` line of `unsga3-bend/ab/igd_vs_pymoo.py`: the mean Euclidean distance from each reference-front point to the nearest obtained point. `fidelity.inverted_generational_distance` computes it. `fidelity.generational_distance` is a different formula (obtained toward reference, RMS at `p=2`) and is not this column.",
        "",
        "Reference fronts: ZDT1 and ZDT2 analytic curves, 500 points (`f1 = i / 499`); DTLZ2 Das–Dennis at partitions 12, each direction scaled to unit L2 length (91 points, `pf_source=analytic-das-dennis-l2`). That is the Bend script's analytic fallback and the C# `ParetoFronts.Dtlz2` construction. pymoo is not imported, and pymoo's default ~136-point DTLZ2 sample is not used.",
        "",
        "Objective evaluations are `pop * n_gen`, including the initial population. ZDT1 is 52×100 = 5200. ZDT2 is 52×250 = 13000. DTLZ2 is 92×150 = 13800. The population is at least the Das–Dennis count (ZDT H=13, DTLZ2 H=91).",
        "",
        "## Medians",
        "",
        "| Problem | Seeds scored | median Bend (published) | median C# (published) | median ExactEw (`igd=`) | median classical U-NSGA-III | classical objective evals | yardstick pop×gens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    yardstick = {"zdt1": 52 * 100, "zdt2": 52 * 250, "dtlz2": 92 * 150}
    order = ("zdt1", "zdt2", "dtlz2")
    for problem in order:
        group = sorted(by_problem[problem], key=lambda row: int(row["seed"]))
        exactew_cells = [exactew.get(problem, {}).get(seed, "") for seed in range(1, 16)]
        exactew_median = (
            median_text(exactew_cells) if all(exactew_cells) else "—"
        )
        if not group:
            lines.append(
                f"| {problem} | 0 | {PUBLISHED_MEDIANS[problem]['bend']} | "
                f"{PUBLISHED_MEDIANS[problem]['csharp']} | {exactew_median} | — | — | {yardstick[problem]} |"
            )
            continue
        median = median_text([str(row["igd_text"]) for row in group])
        lines.append(
            f"| {problem} | {len(group)} | {PUBLISHED_MEDIANS[problem]['bend']} | "
            f"{PUBLISHED_MEDIANS[problem]['csharp']} | {exactew_median} | {median} | "
            f"{group[0]['objective_evals']} | {yardstick[problem]} |"
        )
    lines.extend(
        [
            "",
            "The published median is the summary in the Bend document. The ExactEw median is the middle committed `igd=` cell. The classical median is the middle in-repo IGD after sorting the scored seeds.",
            "",
        ]
    )
    titles = {
        "zdt1": "ZDT1 n=30, partitions=12, pop=52 gens=100",
        "zdt2": "ZDT2 n=30, partitions=12, pop=52 gens=250",
        "dtlz2": "DTLZ2 M=3 k=10 (n=12), partitions=12, pop=92 gens=150",
    }
    for problem in order:
        group = sorted(by_problem[problem], key=lambda row: int(row["seed"]))
        lines.extend(
            [
                f"## {titles[problem]}",
                "",
                "| seed | Bend IGD | C# IGD | ExactEw IGD | classical U-NSGA-III IGD | classical/Bend | classical n | objective evals | pf_rows | pf_source |",
                "|-----:|---------:|-------:|------------:|-------------------------:|---------------:|------------:|----------------:|--------:|-----------|",
            ]
        )
        if not group:
            lines.append("| — | — | — | — | — | — | — | — | — | not scored |")
            lines.append("")
            continue
        published = PUBLISHED_IGD[problem]
        for row in group:
            seed = int(row["seed"])
            bend = published["bend"][seed]
            csharp = published["csharp"][seed]
            exact = exactew.get(problem, {}).get(seed, "—")
            igd_text = str(row["igd_text"])
            lines.append(
                f"| {seed} | {bend} | {csharp} | {exact} | {igd_text} | {_ratio(igd_text, bend)} | "
                f"{row['front_rows']} | {row['objective_evals']} | {row['pf_rows']} | {row['pf_source']} |"
            )
        lines.append("")
        lines.append(
            f"Seeds scored: {', '.join(str(int(row['seed'])) for row in group)}. "
            f"Classical median of those IGD cells: {median_text([str(row['igd_text']) for row in group])}."
        )
        lines.append("")
    lines.extend(
        [
            "## Reading the classical column",
            "",
            "A larger IGD is farther from the reference front. This column is the generational algorithm the ExactEw weight-sweep is not: a population, SBX, polynomial mutation, and reference-direction survival. The RNG is NumPy's Generator, not Bend's LCG, so fronts are not bit-identical to the published Bend runs. The median is the comparison. `classical n` is the number of unique non-dominated objective rows written to the CSV.",
            "",
        ]
    )
    return "\n".join(lines)


def read_exactew_table(path: Path) -> dict[str, dict[int, str]]:
    return load_exactew_igd(path.read_text(encoding="utf-8"))
