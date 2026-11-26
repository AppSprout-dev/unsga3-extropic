"""Targeted regression tests for the multi-repo audit fix pass."""

from __future__ import annotations

import math

import numpy as np
import pytest

from unsga3_extropic.archive import Archive, hypervolume_2d, nondominated_mask
from unsga3_extropic.backends.ew_metropolis import (
    ExactEwContinuousBackend,
    ExactEwMetropolisBackend,
)
from unsga3_extropic.domain_wall import compile_domain_wall
from unsga3_extropic.fidelity import compare_fronts, generational_distance
from unsga3_extropic.results import append_run_record, stub_run_record, validate_run_record
from unsga3_extropic.weights import simplex_weights


def _naive_nd_mask(objs: np.ndarray, *, minimize: bool = True) -> np.ndarray:
    """Reference O(n^2) pairwise dominance loop with exact 1e-12 slack."""
    n = len(objs)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if minimize:
            dom = np.all(objs <= objs[i] + 1e-12, axis=1) & np.any(
                objs < objs[i] - 1e-12, axis=1
            )
        else:
            dom = np.all(objs >= objs[i] - 1e-12, axis=1) & np.any(
                objs > objs[i] + 1e-12, axis=1
            )
        if np.any(dom):
            keep[i] = False
    return keep


def test_exact_ew_continuous_detailed_balance_uniform_at_beta_zero():
    """At beta=0 on [0, 1], truncated-normal MH must preserve Uniform(0, 1).

    The previously inverted Hastings ratio Z(new)/Z(old) piled ~1.45x excess
    mass at the box boundaries and would decisively fail this test.
    """
    backend = ExactEwContinuousBackend(
        n_var=1,
        objective_fn=lambda x: np.array([0.0, 0.0]),
        lower=0.0,
        upper=1.0,
        step_scale=0.25,
    )
    res = backend.sample_weight(
        np.array([0.5, 0.5]),
        seed=42,
        betas=(0.0,),
        n_warmup=500,
        n_samples=12_000,
        steps_per_sample=2,
    )
    xs = res.decisions[:, 0]
    assert np.all((xs >= 0.0) & (xs <= 1.0))
    # 5 equal bins of width 0.2 each should hold ~0.20 of mass (edge vs interior ratio ~1.0).
    counts, _ = np.histogram(xs, bins=np.linspace(0.0, 1.0, 6))
    probs = counts / len(xs)
    edge_mass = probs[0] + probs[-1]
    assert abs(edge_mass - 0.40) < 0.03, f"boundary bias detected: bin probs={probs}"
    assert np.all(np.abs(probs - 0.20) < 0.025), f"non-uniform bins: {probs}"


def test_exact_ew_continuous_huge_step_scale_terminates_via_inv_cdf():
    """step_scale >> (upper - lower) must not hang in rejection sampling."""
    backend = ExactEwContinuousBackend(
        n_var=2,
        objective_fn=lambda x: np.array([float(x[0]), float(x[1])]),
        lower=0.0,
        upper=1.0,
        step_scale=1e6,
    )
    res = backend.sample_weight(
        np.array([0.5, 0.5]),
        seed=7,
        betas=(1.0,),
        n_warmup=2,
        n_samples=25,
        steps_per_sample=1,
    )
    assert res.decisions.shape == (25, 2)
    assert np.all((res.decisions >= 0.0) & (res.decisions <= 1.0))


def test_exact_ew_metropolis_does_not_re_evaluate_recorded_samples():
    """Cached (x, f, e) avoids uncounted duplicate objective_fn calls."""
    calls = 0

    def counted_obj(bits: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += len(bits)
        b = np.asarray(bits, dtype=np.float64)
        return np.stack([b.sum(-1), (1.0 - b).sum(-1)], axis=-1)

    backend = ExactEwMetropolisBackend(n_bits=6, objective_fn=counted_obj)
    res = backend.sample_weight(
        np.array([0.4, 0.6]),
        seed=9,
        betas=(0.5, 2.0),
        n_warmup=4,
        n_samples=10,
        steps_per_sample=2,
    )
    assert calls == res.n_evals
    assert np.allclose(res.energies, res.objectives @ np.array([0.4, 0.6]))


def test_compare_fronts_hv_is_monotonic_with_fixed_reference():
    """A dominated front with larger outliers must NOT get a higher hypervolume."""
    ref = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    good = ref.copy()
    worse_outlier = np.array([[0.0, 1.0], [5.0, 5.0]])

    s_good = compare_fronts(good, ref)
    s_worse = compare_fronts(worse_outlier, ref)
    assert s_good["hv_ref"] == s_worse["hv_ref"]
    assert s_good["hypervolume_2d"] > s_worse["hypervolume_2d"]

    s_custom = compare_fronts(good, ref, hv_ref=(2.0, 2.0))
    assert s_custom["hv_ref"] == [2.0, 2.0]
    assert s_custom["hypervolume_2d"] == pytest.approx(3.25)

    with pytest.raises(ValueError, match="finite"):
        compare_fronts(good, ref, hv_ref=(float("nan"), 2.0))


def test_generational_distance_empty_is_inf():
    ref = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert math.isinf(generational_distance(np.zeros((0, 2)), ref))


def test_hypervolume_2d_shape_validation_and_exact_duplicates():
    with pytest.raises(ValueError, match="shape"):
        hypervolume_2d(np.zeros((0, 3)), ref=(1.0, 1.0))
    with pytest.raises(ValueError, match="shape"):
        hypervolume_2d(np.array([0.5, 0.5]), ref=(1.0, 1.0))

    # Duplicate and tightly spaced points (< 1e-6 apart) keep exact float64 area.
    pts = np.array([[0.0, 1.0], [0.0, 1.0], [1e-7, 1.0 - 1e-7], [1.0, 0.0]])
    hv = hypervolume_2d(pts, ref=(2.0, 2.0))
    assert hv == pytest.approx(3.0 + (2.0 - 1e-7) * 1e-7)


@pytest.mark.parametrize("minimize", [True, False])
def test_2d_sweep_line_matches_naive_pairwise_across_edge_cases(minimize: bool):
    rng = np.random.default_rng(12345)
    clouds = [
        np.zeros((0, 2)),
        np.array([[0.5, 0.5]]),
        np.array([[0.0, 1.0], [0.0, 1.0], [0.0 + 5e-13, 1.0 - 5e-13], [0.0 + 5e-13, 0.0]]),
        np.array([[1.0, 1.0], [1.0 + 9e-13, 1.0 - 2e-12], [1.0 - 2e-12, 1.0 + 9e-13]]),
        np.array([[1.0, 2.0], [1.0, 2.0 + 5e-13], [1.0, 2.0 - 2e-12], [1.0 + 2e-12, 0.0]]),
        rng.normal(size=(250, 2)),
        np.round(rng.uniform(-5.0, 5.0, size=(200, 2)), 1),  # heavy ties
    ]
    for cloud in clouds:
        got = nondominated_mask(cloud, minimize=minimize)
        expected = _naive_nd_mask(cloud, minimize=minimize)
        assert np.array_equal(got, expected), f"mismatch on cloud shape {cloud.shape}"


def test_archive_nondominated_deduplicates_nan_rows():
    arc = Archive()
    arc.add(
        np.array([[0.0], [1.0], [2.0]]),
        np.array([[np.nan, 0.0], [np.nan, 0.0], [0.5, 0.5]]),
    )
    xs, fs = arc.nondominated()
    assert len(xs) == 2
    assert len(fs) == 2


def test_results_validation_rejects_nonfinite_and_short_date(tmp_path):
    base = stub_run_record(front_path=tmp_path / "f.npz")
    validate_run_record(base)

    bad_date = dict(base, date="2026-10-02")
    with pytest.raises(ValueError, match="date"):
        validate_run_record(bad_date)

    bad_beta = dict(base, schedule={**base["schedule"], "betas": [1.0, float("nan")]})
    with pytest.raises(ValueError, match="finite"):
        validate_run_record(bad_beta)

    bad_hv = dict(base, metrics={**base["metrics"], "hypervolume_2d": float("inf")})
    with pytest.raises(ValueError, match="finite"):
        validate_run_record(bad_hv)

    bad_ref = dict(base, metrics={**base["metrics"], "hv_ref": [1.0, float("nan")]})
    with pytest.raises(ValueError, match="finite"):
        append_run_record(tmp_path / "runs.jsonl", bad_ref)


def test_simplex_weights_high_m_is_bounded():
    w = simplex_weights(50, n_obj=15)
    assert w.shape == (50, 15)
    assert np.all(w >= 0.0)
    assert np.allclose(w.sum(axis=1), 1.0)


def test_domain_wall_valid_mask_rejects_non_pm1_and_inf():
    unary = np.zeros((2, 3), dtype=np.float64)
    edges = [(0, 1)]
    pair = np.zeros((1, 3, 3), dtype=np.float64)
    image = compile_domain_wall(unary, edges, pair)
    good = np.array([[1.0, -1.0, 1.0, 1.0]])
    bad_five = np.array([[5.0, -1.0, 1.0, 1.0]])
    bad_inf = np.array([[np.inf, -1.0, 1.0, 1.0]])
    bad_zero = np.array([[0.0, -1.0, 1.0, 1.0]])
    assert image.valid_mask(good).tolist() == [True]
    assert image.valid_mask(bad_five).tolist() == [False]
    assert image.valid_mask(bad_inf).tolist() == [False]
    assert image.valid_mask(bad_zero).tolist() == [False]


@pytest.mark.thrml
def test_thrml_ising_handles_single_spin():
    pytest.importorskip("jax")
    pytest.importorskip("thrml")
    from unsga3_extropic.backends.thrml_ising import ThrmlIsingBackend

    def build(w):
        return np.array([float(w[0] - w[1])], dtype=np.float32), [], np.zeros((0,), dtype=np.float32)

    def obj(spins):
        return np.stack([-spins[:, 0], spins[:, 0]], axis=-1)

    backend = ThrmlIsingBackend(n_spins=1, build_ising=build, objective_fn=obj)
    res = backend.sample_weight(
        np.array([1.0, 0.0]),
        seed=0,
        betas=(1.0, 4.0),
        n_warmup=2,
        n_samples=4,
        steps_per_sample=1,
    )
    assert res.decisions.shape == (8, 1)
    assert res.objectives.shape == (8, 2)
    assert np.allclose(res.energies, res.objectives @ np.array([1.0, 0.0]))
