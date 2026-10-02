"""THRML sampling of the domain-wall Ising image."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("thrml")

import jax.numpy as jnp
import thrml
import thrml.models as thrml_models

from unsga3_extropic.backends.thrml_domain_wall import ThrmlDomainWallBackend
from unsga3_extropic.backends.thrml_ising import ThrmlIsingBackend
from unsga3_extropic.domain_wall import (
    DomainWallImage,
    checkerboard_permutation,
    compile_domain_wall,
    ising_energy,
)
from unsga3_extropic.problems import PottsChainProblem

pytestmark = pytest.mark.thrml


def test_k3_samples_reproduce_potts_objectives(monkeypatch):
    problem = PottsChainProblem(n_sites=4, n_categories=3)
    backend = problem.make_domain_wall_backend(constraint_strength=8.0)
    feasible_calls: list[int] = []
    ising_calls: list[str] = []
    spin_factors: list[int] = []
    real_feasible = DomainWallImage.feasible
    real_ising = ThrmlIsingBackend.sample_weight
    real_factor = thrml_models.SpinEBMFactor

    def feasible(self, spins, objective_fn):
        feasible_calls.append(int(np.asarray(spins).shape[0]))
        return real_feasible(self, spins, objective_fn)

    def ising_sample(self, *args, **kwargs):
        ising_calls.append("thrml_ising")
        return real_ising(self, *args, **kwargs)

    def spin_factor(*args, **kwargs):
        spin_factors.append(1)
        return real_factor(*args, **kwargs)

    monkeypatch.setattr(DomainWallImage, "feasible", feasible)
    monkeypatch.setattr(ThrmlIsingBackend, "sample_weight", ising_sample)
    monkeypatch.setattr(thrml_models, "SpinEBMFactor", spin_factor)

    w = np.array([0.4, 0.6])
    kwargs = dict(seed=3, betas=(1.0, 4.0), n_warmup=1, n_samples=3, steps_per_sample=1)
    first = backend.sample_weight(w, **kwargs)
    second = backend.sample_weight(w, **kwargs)
    assert feasible_calls
    assert ising_calls == []
    assert spin_factors
    assert first.n_evals + first.n_invalid == 6
    assert first.decisions.shape == (first.n_evals, problem.n_sites)
    assert np.allclose(first.objectives, problem.energies_from_states(first.decisions))
    assert np.allclose(first.energies, first.objectives @ w)
    assert set(np.unique(first.decisions)).issubset(set(range(problem.n_categories)))
    assert np.array_equal(first.decisions, second.decisions)
    assert first.n_invalid == second.n_invalid


def test_k2_chain_uses_thrml_ising_backend(monkeypatch):
    problem = PottsChainProblem(n_sites=4, n_categories=2)
    backend = problem.make_domain_wall_backend()
    ising_calls: list[str] = []
    real_ising = ThrmlIsingBackend.sample_weight

    def ising_sample(self, *args, **kwargs):
        ising_calls.append("thrml_ising")
        return real_ising(self, *args, **kwargs)

    monkeypatch.setattr(ThrmlIsingBackend, "sample_weight", ising_sample)
    result = backend.sample_weight(
        np.array([0.5, 0.5]),
        seed=1,
        betas=(2.0,),
        n_warmup=1,
        n_samples=4,
        steps_per_sample=1,
    )
    assert ising_calls == ["thrml_ising"]
    assert result.n_invalid == 0
    assert result.n_evals == 4
    assert np.allclose(result.objectives, problem.energies_from_states(result.decisions))
    assert np.allclose(result.energies, result.objectives @ np.array([0.5, 0.5]))


def test_checkerboard_path_drops_an_invalid_thermometer(monkeypatch):
    """One site, K=4: a path of three spins, so ThrmlIsingBackend can sample it."""

    def build(_w: np.ndarray):
        unary = np.zeros((1, 4), dtype=np.float32)
        pair = np.zeros((0, 4, 4), dtype=np.float32)
        return unary, [], pair

    def objective_fn(states: np.ndarray) -> np.ndarray:
        return np.asarray(states, dtype=np.float64)

    backend = ThrmlDomainWallBackend(
        n_sites=1,
        n_categories=4,
        build_potts=build,
        objective_fn=objective_fn,
        constraint_strength=4.0,
    )
    assert backend.program_kind(np.ones(1)) == "thrml_ising"
    image = compile_domain_wall(*build(np.ones(1)))
    permutation = checkerboard_permutation(
        image.n_spins, image.constraint_edges + image.inter_edges
    )
    assert permutation is not None
    assert np.array_equal(permutation, np.arange(image.n_spins))

    def fake_sample_states(key, program, schedule, state, clamped, observe):
        rows = np.zeros((int(schedule.n_samples), 3), dtype=bool)
        rows[1, 1] = True  # [-1, +1, -1] is not a thermometer
        return [jnp.asarray(rows)]

    monkeypatch.setattr(thrml, "sample_states", fake_sample_states)
    result = backend.sample_weight(
        np.ones(1),
        seed=0,
        betas=(1.0,),
        n_warmup=0,
        n_samples=2,
        steps_per_sample=1,
    )
    assert result.n_invalid == 1
    assert result.n_evals == 1
    assert np.array_equal(result.decisions, np.array([[0]]))
    assert np.allclose(result.objectives, np.array([[0.0]]))


def test_spin_factor_energy_matches_numpy_on_a_thermometer():
    problem = PottsChainProblem(n_sites=3, n_categories=3)
    w = np.array([0.25, 0.5])
    unary, edges, pair = problem.build_potts(w)
    image = compile_domain_wall(unary, edges, pair)
    state = np.array([1, 2, 0], dtype=np.int64)
    spins = image.encode(state)
    biases, edge_list, couplings = image.ising_terms(0.0)
    expected = float(ising_energy(spins, biases, edge_list, couplings))

    nodes = [thrml.SpinNode() for _ in range(image.n_spins)]
    factors = [
        thrml_models.SpinEBMFactor(
            [thrml.Block(nodes)], jnp.asarray(biases, dtype=jnp.float32)
        )
    ]
    if edge_list:
        factors.append(
            thrml_models.SpinEBMFactor(
                [
                    thrml.Block([nodes[i] for i, _j in edge_list]),
                    thrml.Block([nodes[j] for _i, j in edge_list]),
                ],
                jnp.asarray(couplings, dtype=jnp.float32),
            )
        )
    ebm = thrml_models.FactorizedEBM(factors)
    energy = float(ebm.energy([jnp.asarray(spins > 0.0)], [thrml.Block(nodes)]))
    assert energy == pytest.approx(expected, abs=1e-5)
