"""Potts parameter checks that do not sample (no JAX required)."""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from unsga3_extropic.archive import nondominated_mask
from unsga3_extropic.backends.thrml_potts import (
    ThrmlPottsBackend,
    validate_potts_instance,
)
from unsga3_extropic.problems import PottsChainProblem


def _backend(
    n: int,
    edges: list[tuple[int, int]],
    *,
    k: int = 3,
    coloring=None,
    unary: np.ndarray | None = None,
    pairwise: np.ndarray | None = None,
) -> ThrmlPottsBackend:
    def build(_w: np.ndarray):
        u = (
            np.zeros((n, k), np.float32)
            if unary is None
            else np.asarray(unary, dtype=np.float32)
        )
        ww = (
            np.zeros((len(list(edges)), k, k), np.float32)
            if pairwise is None
            else np.asarray(pairwise, dtype=np.float32)
        )
        return u, edges, ww

    return ThrmlPottsBackend(
        n_sites=n,
        n_categories=k,
        build_potts=build,
        objective_fn=lambda states: np.zeros(np.asarray(states).shape[:-1] + (1,)),
        coloring=coloring,
    )


def _call(backend: ThrmlPottsBackend) -> None:
    backend.sample_weight(
        np.ones(1),
        seed=0,
        betas=(1.0,),
        n_warmup=0,
        n_samples=1,
        steps_per_sample=1,
    )


def test_rejects_same_color_edge_before_thrml_import(monkeypatch):
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "jax" or name.startswith("thrml"):
            raise AssertionError(f"imported {name} before rejecting the edge")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    backend = _backend(4, [(0, 2)])
    with pytest.raises(ValueError, match="free block"):
        _call(backend)


def test_explicit_coloring_rejects_an_edge_the_parity_rule_would_allow():
    # 0 and 1 have opposite parity. This coloring puts them in one block.
    backend = _backend(4, [(0, 1)], coloring=((0, 1), (2, 3)))
    with pytest.raises(ValueError, match="free block"):
        _call(backend)


def test_rejects_unary_shape_and_empty_schedule():
    backend = _backend(3, [(0, 1)], unary=np.zeros((2, 3), np.float32))
    with pytest.raises(ValueError, match="unary"):
        _call(backend)
    ok = _backend(2, [(0, 1)])
    with pytest.raises(ValueError, match="betas"):
        ok.sample_weight(
            np.ones(1), seed=0, betas=(), n_warmup=0, n_samples=1, steps_per_sample=1
        )
    with pytest.raises(ValueError, match="seed"):
        ok.sample_weight(
            np.ones(1),
            seed=-1,
            betas=(1.0,),
            n_warmup=0,
            n_samples=1,
            steps_per_sample=1,
        )


def test_coloring_must_partition_the_sites():
    unary = np.zeros((3, 2), np.float32)
    pair = np.zeros((1, 2, 2), np.float32)
    with pytest.raises(ValueError, match="does not cover"):
        validate_potts_instance(3, 2, unary, [(0, 1)], pair, ((0,), (1,)))
    with pytest.raises(ValueError, match="more than one"):
        validate_potts_instance(2, 2, unary[:2], [(0, 1)], pair, ((0, 1), (1,)))
    with pytest.raises(ValueError, match="empty"):
        validate_potts_instance(2, 2, unary[:2], [], np.zeros((0, 2, 2)), ((0, 1), ()))


def test_build_potts_matches_scalarized_energy():
    problem = PottsChainProblem(n_sites=5, n_categories=3)
    w = np.array([0.25, 0.5])
    unary_w, edges, pair_w = problem.build_potts(w)
    assert edges == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert np.allclose(unary_w, -0.25 * problem.unary)
    assert pair_w.shape == (4, 3, 3)
    assert np.allclose(pair_w, -0.5 * problem.pair)
    rng = np.random.default_rng(0)
    for _ in range(12):
        state = rng.integers(0, problem.n_categories, size=problem.n_sites)
        objs = problem.energies_from_states(state)
        e_w = float(w @ objs)
        # Factor energy is -W[state] at beta = 1.
        e_factor = -float(unary_w[np.arange(problem.n_sites), state].sum())
        for e_i, (i, j) in enumerate(edges):
            e_factor -= float(pair_w[e_i, state[i], state[j]])
        assert e_w == pytest.approx(e_factor, abs=1e-5)


def test_preferred_and_agreeing_states():
    problem = PottsChainProblem(n_sites=6, n_categories=3)
    preferred = problem.energies_from_states(problem.preferred)
    assert preferred[0] == pytest.approx(0.0)
    assert preferred[1] == pytest.approx(5.0)
    constant = np.zeros(6, dtype=np.int64)
    f0, f1 = problem.energies_from_states(constant)
    assert f1 == pytest.approx(0.0)
    assert f0 == pytest.approx(4.0)
    batch = np.stack([problem.preferred, constant], axis=0)
    stacked = problem.energies_from_states(batch)
    assert stacked.shape == (2, 2)
    assert np.allclose(stacked[0], preferred)
    assert np.allclose(stacked[1], [f0, f1])


def test_enumerate_front_is_nondominated_and_trades_off():
    problem = PottsChainProblem(n_sites=4, n_categories=3)
    states, front = problem.enumerate_front()
    assert len(states) == len(front) >= 2
    assert np.all(nondominated_mask(front))
    assert np.allclose(problem.energies_from_states(states), front)
    assert front[:, 0].min() == pytest.approx(0.0)
    assert front[:, 1].min() == pytest.approx(0.0)
    with pytest.raises(ValueError, match="enumerate_front"):
        PottsChainProblem(n_sites=9, n_categories=3).enumerate_front()
