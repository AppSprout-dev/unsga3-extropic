"""Opt-in EBR tools: default path unchanged, argmin selection, plateau stop."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from unsga3_extropic.backends import ExactEwMetropolisBackend
from unsga3_extropic.backends.base import BackendResult
from unsga3_extropic.ebr import (
    LowestEnergySelection,
    PlateauStop,
    lowest_energy_index,
    select_lowest_energy,
)
from unsga3_extropic.loop import AnnealConfig, WeightSweepLoop


def _toy_objective(bits: np.ndarray) -> np.ndarray:
    b = np.asarray(bits, dtype=np.float64)
    return np.stack([b.sum(-1), (1 - b).sum(-1) + b[..., 0] * b[..., -1]], axis=-1)


def _small_run(**backend_kwargs) -> BackendResult:
    backend = ExactEwMetropolisBackend(n_bits=8, objective_fn=_toy_objective, **backend_kwargs)
    return backend.sample_weight(
        np.array([0.3, 0.7]),
        seed=123,
        betas=(0.5, 2.0),
        n_warmup=5,
        n_samples=7,
        steps_per_sample=2,
    )


def _exact_digest(r: BackendResult) -> str:
    """FMA-independent digest over exact integer-valued decisions, objectives, and n_evals."""
    h = hashlib.sha256()
    for a in (r.decisions, r.objectives):
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    h.update(str(r.n_evals).encode())
    return h.hexdigest()


def _full_digest(r: BackendResult) -> str:
    """Host-local full digest including float64 dot-product energies."""
    h = hashlib.sha256()
    for a in (r.decisions, r.objectives, r.energies):
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    h.update(str(r.n_evals).encode())
    return h.hexdigest()


# --- default path ---------------------------------------------------------


def test_default_exact_ew_matches_main_digest():
    base = _small_run()
    none_run = _small_run(plateau=None)
    assert _full_digest(base) == _full_digest(none_run)
    assert _exact_digest(base) == _exact_digest(none_run)
    w = np.array([0.3, 0.7])
    np.testing.assert_allclose(base.energies, base.objectives @ w, rtol=0.0, atol=1e-14)
    # Decisions and objectives are integer-valued on _toy_objective, so exact digest is CPU-portable.
    assert base.n_evals == 1 + 2 * (5 + 7 * 2)


def test_plateau_that_never_fires_is_byte_identical():
    # 7 samples per stage, window 7: k never reaches the window.
    base = _small_run()
    with_plateau = _small_run(plateau=PlateauStop(window=7))
    assert _full_digest(with_plateau) == _full_digest(base)


def test_default_loop_unchanged_by_import():
    import unsga3_extropic.ebr  # noqa: F401

    def run():
        backend = ExactEwMetropolisBackend(n_bits=6, objective_fn=_toy_objective)
        cfg = AnnealConfig(betas=(1.0, 4.0), n_warmup=3, n_samples=5, steps_per_sample=1)
        return WeightSweepLoop(backend=backend, n_weights=3, anneal=cfg, seed=7).run()

    a, b = run(), run()
    assert a.total_evals == b.total_evals
    np.testing.assert_array_equal(a.archive.objectives, b.archive.objectives)
    assert len(a.archive.objectives) == 3 * 2 * 5


# --- selection ------------------------------------------------------------


def test_lowest_energy_index_is_argmin_with_earliest_tie():
    assert lowest_energy_index(np.array([3.0, -1.0, 2.0, -1.0])) == 1


def test_lowest_energy_index_skips_nonfinite():
    assert lowest_energy_index(np.array([np.nan, 2.0, -np.inf, 1.0])) == 3
    with pytest.raises(ValueError):
        lowest_energy_index(np.array([np.nan, np.inf]))
    with pytest.raises(ValueError):
        lowest_energy_index(np.array([]))


def test_select_lowest_energy_returns_argmin_row():
    rng = np.random.default_rng(0)
    e = rng.normal(size=50)
    x = rng.integers(0, 2, size=(50, 4)).astype(float)
    f = rng.normal(size=(50, 2))
    res = BackendResult(decisions=x, objectives=f, energies=e, n_evals=77, n_invalid=3)
    out = select_lowest_energy(res)
    i = int(np.argmin(e))
    np.testing.assert_array_equal(out.decisions, x[i : i + 1])
    np.testing.assert_array_equal(out.objectives, f[i : i + 1])
    assert out.energies.tolist() == [e.min()]
    assert (out.n_evals, out.n_invalid) == (77, 3)


def test_select_lowest_energy_passes_empty_through():
    res = BackendResult(
        decisions=np.zeros((0, 3)), objectives=np.zeros((0, 2)), energies=np.zeros(0),
        n_evals=5, n_invalid=5,
    )
    assert select_lowest_energy(res) is res


def test_wrapper_keeps_argmin_of_inner_backend():
    inner = _small_run()
    wrapped = LowestEnergySelection(
        ExactEwMetropolisBackend(n_bits=8, objective_fn=_toy_objective)
    ).sample_weight(
        np.array([0.3, 0.7]), seed=123, betas=(0.5, 2.0), n_warmup=5, n_samples=7,
        steps_per_sample=2,
    )
    i = int(np.argmin(inner.energies))
    assert len(wrapped.energies) == 1
    assert wrapped.energies[0] == inner.energies.min()
    np.testing.assert_array_equal(wrapped.decisions[0], inner.decisions[i])
    assert wrapped.n_evals == inner.n_evals


# --- plateau stop ---------------------------------------------------------


def test_plateau_should_stop_on_best_trace():
    stop = PlateauStop(window=3, tol=0.5)
    assert not stop.should_stop([5.0, 4.0, 3.0])  # k=2 < window
    assert not stop.should_stop([5.0, 4.0, 3.0, 2.0])  # 5 - 2 = 3 > 0.5
    assert stop.should_stop([5.0, 4.0, 4.0, 4.0, 4.0])  # 4 - 4 = 0
    assert stop.should_stop([5.0, 4.0, 3.9, 3.8, 3.6])  # 4 - 3.6 = 0.4 <= 0.5


def test_plateau_stop_index_uses_running_best():
    stop = PlateauStop(window=2, tol=0.0)
    # best trace: 3, 2, 2, 2 -> fires at k=3 (best[1] - best[3] = 0)
    assert stop.stop_index(np.array([3.0, 2.0, 9.0, 5.0, 1.0])) == 3
    # strictly improving: never fires
    assert stop.stop_index(np.array([5.0, 4.0, 3.0, 2.0, 1.0])) is None
    # constant: earliest possible stop is index == window
    assert PlateauStop(window=4).stop_index(np.zeros(10)) == 4


@pytest.mark.parametrize("window", [0, -1, 1.5, True])
def test_plateau_rejects_bad_window(window):
    with pytest.raises(ValueError):
        PlateauStop(window=window)


@pytest.mark.parametrize("tol", [-1e-9, float("nan"), float("inf")])
def test_plateau_rejects_bad_tol(tol):
    with pytest.raises(ValueError):
        PlateauStop(window=2, tol=tol)


def test_backend_plateau_stops_each_stage_on_flat_energy():
    def flat(bits):
        return np.zeros((np.asarray(bits).shape[0], 2))

    backend = ExactEwMetropolisBackend(n_bits=5, objective_fn=flat, plateau=PlateauStop(window=3))
    betas = (0.5, 1.0, 2.0)
    res = backend.sample_weight(
        np.array([0.5, 0.5]), seed=1, betas=betas, n_warmup=4, n_samples=20, steps_per_sample=2
    )
    # each stage records window + 1 = 4 samples, then moves to the next beta
    assert len(res.energies) == len(betas) * 4
    assert res.n_evals == 1 + len(betas) * (4 + 4 * 2)


def _downhill_flip(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Flip the first 1 to 0, so energy = number of ones drops every step."""
    y = x.copy()
    ones = np.flatnonzero(y == 1)
    if len(ones):
        y[ones[0]] = 0
    return y


def _count_ones(bits: np.ndarray) -> np.ndarray:
    b = np.asarray(bits, dtype=np.float64)
    return np.stack([b.sum(-1), np.zeros(b.shape[0])], axis=-1)


def _flat(bits: np.ndarray) -> np.ndarray:
    return np.zeros((np.asarray(bits).shape[0], 2))


def test_backend_plateau_does_not_fire_while_improving():
    # 40 ones, one forced downhill flip per step: strictly improving for all 30 samples.
    window, n_samples = 8, 30
    assert window < n_samples
    backend = ExactEwMetropolisBackend(
        n_bits=40,
        objective_fn=_count_ones,
        propose=_downhill_flip,
        init_fn=lambda rng: np.ones(40),
        plateau=PlateauStop(window=window),
    )
    res = backend.sample_weight(
        np.array([1.0, 0.0]), seed=0, betas=(50.0,), n_warmup=0, n_samples=n_samples,
        steps_per_sample=1,
    )
    assert len(res.energies) == n_samples
    assert np.all(np.diff(res.energies) < 0)


def test_backend_plateau_same_window_fires_on_flat_energy():
    # Paired with the test above: the same window does fire on a flat trace.
    window, n_samples = 8, 30
    backend = ExactEwMetropolisBackend(
        n_bits=40,
        objective_fn=_flat,
        propose=_downhill_flip,
        init_fn=lambda rng: np.ones(40),
        plateau=PlateauStop(window=window),
    )
    res = backend.sample_weight(
        np.array([1.0, 0.0]), seed=0, betas=(50.0,), n_warmup=0, n_samples=n_samples,
        steps_per_sample=1,
    )
    assert len(res.energies) == window + 1


# --- non-finite energies --------------------------------------------------


def test_running_best_ignores_nan():
    rb = PlateauStop.running_best
    assert rb(float("inf"), float("nan")) == float("inf")
    assert rb(2.0, float("nan")) == 2.0
    assert rb(2.0, 1.0) == 1.0
    assert rb(2.0, float("inf")) == 2.0
    assert rb(2.0, float("-inf")) == float("-inf")


def test_plateau_leading_nan_does_not_poison_trace():
    stop = PlateauStop(window=2)
    nan = float("nan")
    # best trace: inf, 3, 3, 3 -> fires at k=3 (3 - 3 = 0)
    assert stop.stop_index(np.array([nan, 3.0, 5.0, 4.0, 1.0])) == 3
    # NaN in the middle is skipped too: best 3, 3, 2, 2, 2 -> fires at k=4
    assert stop.stop_index(np.array([3.0, nan, 2.0, nan, nan, 0.0])) == 4
    # still improving after a leading NaN: never fires
    assert stop.stop_index(np.array([nan, 5.0, 4.0, 3.0, 2.0])) is None


def test_plateau_only_nan_or_posinf_never_fires():
    stop = PlateauStop(window=1)
    assert stop.stop_index(np.full(10, np.nan)) is None
    assert stop.stop_index(np.full(10, np.inf)) is None
    assert not stop.should_stop([float("nan"), float("nan")])


def test_plateau_first_finite_after_inf_is_improvement():
    stop = PlateauStop(window=1, tol=1e9)
    assert not stop.should_stop([float("inf"), 5.0])
    assert stop.should_stop([5.0, 5.0])


def test_plateau_neg_inf_fires_after_window_and_not_before():
    stop = PlateauStop(window=2)
    ninf = float("-inf")
    # best: 3, -inf, -inf, -inf -> k=2 compares 3 vs -inf (unbounded gain), k=3 fires
    assert stop.stop_index(np.array([3.0, ninf, 7.0, 0.0, 0.0])) == 3
    assert not stop.should_stop([3.0, 2.0, ninf])
    assert stop.should_stop([ninf, ninf, ninf])


def test_backend_plateau_all_nan_energy_records_every_sample():
    def nan_obj(bits):
        return np.full((np.asarray(bits).shape[0], 2), np.nan)

    backend = ExactEwMetropolisBackend(n_bits=6, objective_fn=nan_obj, plateau=PlateauStop(window=2))
    res = backend.sample_weight(
        np.array([0.5, 0.5]), seed=3, betas=(1.0, 2.0), n_warmup=2, n_samples=10, steps_per_sample=1
    )
    assert len(res.energies) == 2 * 10
