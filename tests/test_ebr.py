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

# Digest of ExactEwMetropolisBackend output for _small_run() at main da7e505,
# before the plateau field existed. The default must keep matching it.
_MAIN_DIGEST = "26d1fd96ae50ffbe2cf98e1cb0c78b4817b8f10cf4306eec7df208932667238b"


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


def _digest(r: BackendResult) -> str:
    h = hashlib.sha256()
    for a in (r.decisions, r.objectives, r.energies):
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    h.update(str(r.n_evals).encode())
    return h.hexdigest()


# --- default path ---------------------------------------------------------


def test_default_exact_ew_matches_main_digest():
    assert _digest(_small_run()) == _MAIN_DIGEST
    assert _digest(_small_run(plateau=None)) == _MAIN_DIGEST


def test_plateau_that_never_fires_is_byte_identical():
    # 7 samples per stage, window 7: k never reaches the window.
    assert _digest(_small_run(plateau=PlateauStop(window=7))) == _MAIN_DIGEST


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


def test_backend_plateau_does_not_fire_while_improving():
    # energy = number of ones; start all ones, only downhill flips exist at beta large
    def ones(bits):
        b = np.asarray(bits, dtype=np.float64)
        return np.stack([b.sum(-1), np.zeros(b.shape[0])], axis=-1)

    backend = ExactEwMetropolisBackend(
        n_bits=12,
        objective_fn=ones,
        init_fn=lambda rng: np.ones(12),
        plateau=PlateauStop(window=100),
    )
    res = backend.sample_weight(
        np.array([1.0, 0.0]), seed=0, betas=(50.0,), n_warmup=0, n_samples=30, steps_per_sample=1
    )
    assert len(res.energies) == 30
