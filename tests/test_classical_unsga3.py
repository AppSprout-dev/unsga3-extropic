"""SBX, polynomial mutation, and a tiny classical U-NSGA-III smoke.

No THRML. No pymoo. The operators are not imported by WeightSweepLoop.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np

from unsga3_extropic.classical.igd import pareto_front, score_front
from unsga3_extropic.classical.pm import polynomial_mutation
from unsga3_extropic.classical.sbx import sbx
from unsga3_extropic.classical.unsga3 import (
    _fronts,
    _winner,
    nondominated_objectives,
    run_unsga3,
)
from unsga3_extropic.fidelity import inverted_generational_distance
from unsga3_extropic.loop import WeightSweepLoop
from unsga3_extropic.problems.continuous import zdt1
from unsga3_extropic.weights import das_dennis_directions

ROOT = Path(__file__).resolve().parents[1]


def test_sbx_stays_inside_the_box_and_copies_identical_parents():
    rng = np.random.default_rng(0)
    parents = rng.random((40, 8))
    child_a, child_b = sbx(parents, parents[::-1], 0.0, 1.0, rng, eta=30.0, prob=1.0)
    assert child_a.shape == parents.shape
    assert np.all(child_a >= 0.0) and np.all(child_a <= 1.0)
    assert np.all(child_b >= 0.0) and np.all(child_b <= 1.0)

    same = np.full((6, 4), 0.3)
    copy_a, copy_b = sbx(same, same.copy(), 0.0, 1.0, np.random.default_rng(1))
    assert np.allclose(copy_a, same)
    assert np.allclose(copy_b, same)

    frozen_a, frozen_b = sbx(parents, parents[::-1], 0.0, 1.0, np.random.default_rng(2), prob=0.0)
    assert np.allclose(frozen_a, parents)
    assert np.allclose(frozen_b, parents[::-1])


def test_sbx_with_large_eta_stays_near_the_parents():
    rng = np.random.default_rng(4)
    left = rng.random((30, 5))
    right = rng.random((30, 5))
    child_a, child_b = sbx(left, right, 0.0, 1.0, rng, eta=1.0e6, prob=1.0)
    nearest = np.minimum(np.abs(child_a - left), np.abs(child_a - right))
    assert np.all(nearest < 1e-3)
    nearest_b = np.minimum(np.abs(child_b - left), np.abs(child_b - right))
    assert np.all(nearest_b < 1e-3)


def test_sbx_is_deterministic_for_a_seed():
    left = np.linspace(0.0, 1.0, 12).reshape(3, 4)
    right = 1.0 - left
    first_a, first_b = sbx(left, right, 0.0, 1.0, np.random.default_rng(9))
    second_a, second_b = sbx(left, right, 0.0, 1.0, np.random.default_rng(9))
    assert np.allclose(first_a, second_a)
    assert np.allclose(first_b, second_b)


def test_polynomial_mutation_respects_bounds_and_a_zero_rate():
    rng = np.random.default_rng(3)
    rows = rng.random((25, 10))
    mutated = polynomial_mutation(rows, 0.0, 1.0, rng, eta=20.0, prob=1.0)
    assert mutated.shape == rows.shape
    assert np.all(mutated >= 0.0) and np.all(mutated <= 1.0)
    assert not np.allclose(mutated, rows)

    copied = polynomial_mutation(rows, 0.0, 1.0, np.random.default_rng(3), prob=0.0)
    assert np.allclose(copied, rows)

    again = polynomial_mutation(rows, 0.0, 1.0, np.random.default_rng(8), eta=20.0, prob=None)
    repeat = polynomial_mutation(rows, 0.0, 1.0, np.random.default_rng(8), eta=20.0, prob=None)
    assert np.allclose(again, repeat)


def test_das_dennis_counts_for_the_yardstick_partitions():
    two = das_dennis_directions(2, 12)
    three = das_dennis_directions(3, 12)
    assert two.shape == (13, 2)
    assert three.shape == (91, 3)
    assert np.allclose(two.sum(axis=1), 1.0)
    assert np.allclose(three.sum(axis=1), 1.0)
    assert np.allclose(three * 12.0, np.round(three * 12.0))


def test_igd_is_mean_distance_from_the_reference_to_the_front():
    reference = np.array([[0.0, 1.0], [1.0, 0.0]])
    front = np.array([[0.0, 1.0]])
    assert np.isclose(inverted_generational_distance(front, reference), np.sqrt(2.0) / 2.0)
    assert inverted_generational_distance(np.zeros((0, 2)), reference) == float("inf")
    zdt_pf, source = pareto_front("zdt1")
    assert zdt_pf.shape == (500, 2)
    assert source == "analytic-zdt1 n=500"
    sphere, sphere_source = pareto_front("dtlz2")
    assert sphere.shape == (91, 3)
    assert sphere_source == "analytic-das-dennis-l2"
    assert np.allclose(np.sum(sphere**2, axis=1), 1.0)


def test_non_dominated_fronts_and_same_niche_tournament():
    objectives = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.4, 0.4],
            [2.0, 2.0],
        ]
    )
    fronts = _fronts(objectives)
    assert set(fronts[0]) == {0, 1, 2}
    assert fronts[1] == [3]
    ranks = np.array([0, 2])
    association = np.array([4, 4])
    distance = np.array([0.9, 0.1])
    assert _winner(0, 1, ranks, association, distance, np.random.default_rng(0)) == 0
    ranks = np.array([1, 1])
    distance = np.array([0.4, 0.05])
    assert _winner(0, 1, ranks, association, distance, np.random.default_rng(0)) == 1
    different = np.array([1, 2])
    picked = _winner(0, 1, np.array([0, 5]), different, distance, np.random.default_rng(1))
    assert picked in (0, 1)


def test_smoke_generation_counts_evals_and_stays_in_the_box():
    result = run_unsga3(
        zdt1,
        n_var=30,
        n_obj=2,
        pop_size=8,
        n_gen=3,
        partitions=4,
        seed=1,
    )
    assert result.n_evals == 24
    assert result.n_directions == 5
    assert result.decisions.shape == (8, 30)
    assert result.objectives.shape == (8, 2)
    assert np.all(result.decisions >= 0.0) and np.all(result.decisions <= 1.0)
    assert result.front.shape[1] == 2
    assert len(result.front) >= 1
    assert np.allclose(result.front, nondominated_objectives(result.objectives))
    igd, pf_rows, source = score_front("zdt1", result.front)
    assert pf_rows == 500
    assert source == "analytic-zdt1 n=500"
    assert np.isfinite(igd) and igd > 0.0
    repeat = run_unsga3(
        zdt1,
        n_var=30,
        n_obj=2,
        pop_size=8,
        n_gen=3,
        partitions=4,
        seed=1,
    )
    assert np.allclose(repeat.front, result.front)


def test_classical_report_keeps_exactew_and_published_medians():
    sys.path.insert(0, str(ROOT / "benchmarks"))
    path = ROOT / "benchmarks" / "classical_report.py"
    spec = importlib.util.spec_from_file_location("classical_report", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["classical_report"] = module
    spec.loader.exec_module(module)
    exactew = module.read_exactew_table(ROOT / "benchmarks" / "ORACLE_RESULTS.md")
    assert exactew["zdt1"][1] == "2.573954812980915"
    assert module.median_text(list(exactew["zdt1"].values())) == "2.3144182108708304"
    assert module.median_text(list(exactew["dtlz2"].values())) == "0.27690988609911626"
    text = module.render_classical_results(
        [
            {
                "problem": "zdt1",
                "seed": 1,
                "igd_text": "0.05827076935009233",
                "front_rows": 52,
                "pf_rows": 500,
                "pf_source": "analytic-zdt1 n=500",
                "objective_evals": 5200,
            }
        ],
        exactew=exactew,
        generated_at="2026-10-02T00:00:00Z",
        git_sha="abc",
        partial=True,
    )
    assert "classical continuous U-NSGA-III" in text
    assert "Not THRML-native" in text
    assert "0.071475" in text
    assert "0.082152" in text
    assert "2.3144182108708304" in text
    assert "0.05827076935009233" in text
    assert "WeightSweepLoop" in text


def test_weight_sweep_loop_does_not_call_the_classical_operators():
    source = inspect.getsource(WeightSweepLoop.run)
    for banned in ("crossover", "sbx", "mutate", "polynomial", "classical"):
        assert banned not in source.lower()
    loop_text = (ROOT / "src" / "unsga3_extropic" / "loop.py").read_text(encoding="utf-8")
    assert "unsga3_extropic.classical" not in loop_text
    for name in ("sbx.py", "pm.py", "unsga3.py", "igd.py", "__init__.py"):
        text = (ROOT / "src" / "unsga3_extropic" / "classical" / name).read_text(encoding="utf-8")
        assert "IsingEBM" not in text
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                assert "thrml" not in stripped.lower()
                assert "pymoo" not in stripped.lower()
