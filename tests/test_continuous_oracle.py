"""Continuous ZDT/DTLZ objectives and the NumPy ExactEw yardstick budget.

No THRML. No pymoo. IGD itself is not computed here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from unsga3_extropic.backends import ExactEwContinuousBackend
from unsga3_extropic.problems.continuous import (
    dtlz2,
    oracle_specs,
    zdt1,
    zdt2,
)
from unsga3_extropic.weights import simplex_weights

ROOT = Path(__file__).resolve().parents[1]


def _oracle_report():
    sys.path.insert(0, str(ROOT / "benchmarks"))
    path = ROOT / "benchmarks" / "oracle_report.py"
    spec = importlib.util.spec_from_file_location("oracle_report", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["oracle_report"] = module
    spec.loader.exec_module(module)
    return module


def test_zdt_endpoints_lie_on_the_analytic_front():
    zero = np.zeros(30)
    one = np.zeros(30)
    one[0] = 1.0
    assert np.allclose(zdt1(zero), [0.0, 1.0])
    assert np.allclose(zdt1(one), [1.0, 0.0])
    assert np.allclose(zdt2(zero), [0.0, 1.0])
    assert np.allclose(zdt2(one), [1.0, 0.0])
    quarter = np.zeros(30)
    quarter[0] = 0.25
    assert np.allclose(zdt1(quarter), [0.25, 1.0 - np.sqrt(0.25)])
    assert np.allclose(zdt2(quarter), [0.25, 1.0 - 0.25**2])


def test_zdt_off_front_is_worse_than_the_analytic_curve():
    x = np.full(30, 0.3)
    x[0] = 0.25
    f = zdt1(x)
    assert f[0] == 0.25
    assert f[1] > 1.0 - np.sqrt(0.25)


def test_dtlz2_sphere_when_distance_variables_are_one_half():
    x = np.full(12, 0.5)
    x[0] = 0.0
    x[1] = 0.0
    assert np.allclose(dtlz2(x), [1.0, 0.0, 0.0])
    x[0] = 1.0
    assert np.allclose(dtlz2(x), [0.0, 0.0, 1.0], atol=1e-12)
    x[0] = 0.3
    x[1] = 0.4
    f = dtlz2(x)
    assert np.isclose(np.sum(f**2), 1.0)


def test_oracle_budgets_track_pop_times_gens():
    specs = oracle_specs()
    assert specs["zdt1"].n_var == 30 and specs["zdt1"].pop == 52 and specs["zdt1"].gens == 100
    assert specs["zdt2"].gens == 250
    assert specs["dtlz2"].n_var == 12 and specs["dtlz2"].pop == 92 and specs["dtlz2"].gens == 150
    assert specs["zdt1"].objective_evals() == 5200
    assert specs["zdt2"].objective_evals() == 13000
    assert specs["dtlz2"].yardstick_evals() == 13800
    assert specs["dtlz2"].objective_evals() == 13741
    assert specs["zdt1"].recorded_samples() == 13 * 3 * 120
    for spec in specs.values():
        assert spec.partitions == 12
        assert spec.betas == (1.0, 4.0, 8.0)
        assert spec.step_scale == 0.1


def test_das_dennis_weight_counts_match_partitions_12():
    zdt = simplex_weights(13, 2)
    dtlz = simplex_weights(91, 3)
    assert zdt.shape == (13, 2)
    assert dtlz.shape == (91, 3)
    assert np.allclose(zdt.sum(axis=1), 1.0)
    assert np.allclose(dtlz.sum(axis=1), 1.0)
    assert np.allclose(dtlz * 12.0, np.round(dtlz * 12.0))


def test_continuous_metropolis_counts_each_objective_call_once():
    calls = {"n": 0}

    def objective(point: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        x0 = float(np.asarray(point, dtype=np.float64).reshape(-1)[0])
        return np.array([x0, 1.0 - x0])

    backend = ExactEwContinuousBackend(n_var=4, objective_fn=objective)
    kwargs = dict(
        seed=3,
        betas=(1.0, 4.0),
        n_warmup=2,
        n_samples=5,
        steps_per_sample=1,
    )
    w = np.array([0.3, 0.7])
    first = backend.sample_weight(w, **kwargs)
    expected = 1 + 2 * (2 + 5 * 1)
    assert first.n_evals == expected
    assert calls["n"] == expected
    calls["n"] = 0
    second = backend.sample_weight(w, **kwargs)
    assert np.allclose(first.decisions, second.decisions)
    assert np.allclose(first.objectives, second.objectives)
    assert np.all(first.decisions >= 0.0)
    assert np.all(first.decisions <= 1.0)
    assert np.allclose(first.energies, first.objectives @ w)
    assert first.decisions.shape == (10, 4)


def test_high_beta_coordinate_walk_moves_downhill():
    def objective(point: np.ndarray) -> np.ndarray:
        return np.array([float(np.asarray(point, dtype=np.float64).reshape(-1)[0])])

    backend = ExactEwContinuousBackend(n_var=1, objective_fn=objective, step_scale=0.1)
    result = backend.sample_weight(
        np.array([1.0]),
        seed=1,
        betas=(8.0,),
        n_warmup=80,
        n_samples=1,
        steps_per_sample=1,
    )
    assert float(result.decisions[-1, 0]) < 0.25


def test_published_medians_are_the_middle_cells():
    report = _oracle_report()
    for problem in ("zdt1", "zdt2", "dtlz2"):
        assert report.median_matches_published(problem, "bend")
        assert report.median_matches_published(problem, "csharp")
    assert report.PUBLISHED_MEDIANS["zdt1"] == {"bend": "0.071475", "csharp": "0.082152"}
    assert report.PUBLISHED_MEDIANS["zdt2"]["bend"] == "0.025621"
    assert report.PUBLISHED_MEDIANS["dtlz2"]["csharp"] == "0.004512"


def test_igd_argv_matches_the_bend_yardstick():
    report = _oracle_report()
    assert report.igd_command("python3", "ab/igd_vs_pymoo.py", "EXTROPIC.csv", "zdt1") == [
        "python3",
        "ab/igd_vs_pymoo.py",
        "--front",
        "EXTROPIC.csv",
        "--problem",
        "zdt1",
        "--pf-points",
        "500",
    ]
    assert report.igd_command("python3", "ab/igd_vs_pymoo.py", "EXTROPIC.csv", "zdt2") == [
        "python3",
        "ab/igd_vs_pymoo.py",
        "--front",
        "EXTROPIC.csv",
        "--problem",
        "zdt2",
        "--pf-points",
        "500",
        "--partitions",
        "12",
    ]
    assert report.igd_command("python3", "ab/igd_vs_pymoo.py", "EXTROPIC.csv", "dtlz2") == [
        "python3",
        "ab/igd_vs_pymoo.py",
        "--front",
        "EXTROPIC.csv",
        "--problem",
        "dtlz2",
        "--partitions",
        "12",
    ]
    assert "igd" not in report.parse_igd_stdout("skip: pymoo is not installed\n")


def test_report_copies_igd_text_and_labels_numpy():
    report = _oracle_report()
    row = report.OracleIgdRow(
        problem="zdt1",
        seed=1,
        igd_text="0.424242",
        front_rows="17",
        pf_rows="500",
        pf_source="analytic-zdt1 n=500",
        objective_evals=5200,
        recorded_samples=4680,
        nd_count=17,
    )
    text = report.render_oracle_results(
        [row],
        generated_at="2026-10-02T00:00:00Z",
        git_sha="abc",
        partial=True,
    )
    assert "0.424242" in text
    assert "NumPy ExactEw" in text
    assert "Not THRML-native" in text
    assert "0.061603" in text
    assert "0.097733" in text
    assert "analytic-zdt1 n=500" in text


def test_regenerated_report_keeps_exactew_dominated_pointer():
    """A rewrite of ORACLE_RESULTS.md must still point at the ExactEw card."""
    report = _oracle_report()
    row = report.OracleIgdRow(
        problem="dtlz2",
        seed=1,
        igd_text="0.27690988609911626",
        front_rows="10",
        pf_rows="91",
        pf_source="pymoo-das-dennis",
        objective_evals=13741,
        recorded_samples=12285,
        nd_count=10,
    )
    text = report.render_oracle_results(
        [row],
        generated_at="2026-10-02T18:49:22Z",
        git_sha="3db013e7cc4a3cf1b9cdda29bcd2219a32b7ee2f",
        partial=True,
    )
    card = "docs/spikes/dominated/2026-10-07-exactew-continuous-not-unsga3.md"
    pointer = (
        "These ExactEw numbers stay in this file; the dominated-spike card is "
        f"[{card}](../{card})."
    )
    assert pointer in text
    assert pointer == report.EXACTEW_DOMINATED_POINTER
    committed = (ROOT / "benchmarks" / "ORACLE_RESULTS.md").read_text(encoding="utf-8")
    assert pointer in committed
