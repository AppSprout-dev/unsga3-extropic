"""THRML Potts smoke: energy identity, coloring, and a short anneal."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("thrml")

import thrml.models as thrml_models
from thrml import CategoricalNode, SamplingSchedule

from unsga3_extropic.backends import thrml_potts as potts_backend
from unsga3_extropic.problems import PottsChainProblem

pytestmark = pytest.mark.thrml


def test_schedule_has_no_beta_and_node_has_no_k():
    names = [field.name for field in dataclasses.fields(SamplingSchedule)]
    assert "beta" not in names
    assert not hasattr(CategoricalNode(), "n_categories")


def test_sample_energy_matches_objectives_and_factorized_ebm(monkeypatch):
    problem = PottsChainProblem(n_sites=4, n_categories=3)
    backend = problem.make_backend()
    w = np.array([0.4, 0.6])
    calls: list[int] = []
    real = potts_backend.factorized_potts_energy

    def wrapped(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(potts_backend, "factorized_potts_energy", wrapped)
    kwargs = dict(
        seed=3,
        betas=(1.0, 4.0),
        n_warmup=1,
        n_samples=3,
        steps_per_sample=1,
    )
    first = backend.sample_weight(w, **kwargs)
    second = backend.sample_weight(w, **kwargs)
    assert calls, "sample_weight should score energies with FactorizedEBM"
    assert first.decisions.shape == (6, 4)
    assert first.objectives.shape == (6, 2)
    assert first.energies.shape == (6,)
    assert first.n_evals == 6
    assert first.decisions.dtype == np.dtype(np.int64)
    assert set(np.unique(first.decisions)).issubset(set(range(problem.n_categories)))
    assert np.allclose(first.objectives, problem.energies_from_states(first.decisions))
    assert np.allclose(first.energies, first.objectives @ w, atol=1e-4)
    unary, edges, pair = problem.build_potts(w)
    direct = real(problem.n_sites, unary, edges, pair, first.decisions)
    assert np.allclose(first.energies, direct, atol=1e-5)
    doubled = real(problem.n_sites, 2.0 * unary, edges, 2.0 * pair, first.decisions)
    assert np.allclose(doubled, 2.0 * direct, atol=1e-4)
    assert np.array_equal(first.decisions, second.decisions)


def test_category_count_is_on_the_conditional(monkeypatch):
    problem = PottsChainProblem(n_sites=4, n_categories=3)
    seen: list[int] = []
    real = thrml_models.CategoricalGibbsConditional

    def wrapped(n_categories):
        seen.append(int(n_categories))
        return real(n_categories)

    monkeypatch.setattr(thrml_models, "CategoricalGibbsConditional", wrapped)
    problem.make_backend().sample_weight(
        np.array([0.5, 0.5]),
        seed=1,
        betas=(1.0,),
        n_warmup=0,
        n_samples=1,
        steps_per_sample=1,
    )
    assert seen == [problem.n_categories, problem.n_categories]
    assert int(real(problem.n_categories).n_categories) == problem.n_categories


def test_unary_weight_prefers_the_residue():
    problem = PottsChainProblem(n_sites=4, n_categories=3)
    result = problem.make_backend().sample_weight(
        np.array([1.0, 0.0]),
        seed=2,
        betas=(8.0,),
        n_warmup=2,
        n_samples=8,
        steps_per_sample=1,
    )
    match = result.decisions == problem.preferred.reshape(1, -1)
    assert float(match.mean()) > 0.95
    assert float(result.objectives[:, 0].mean()) < 0.3


def test_pairwise_weight_prefers_agreement():
    problem = PottsChainProblem(n_sites=6, n_categories=3)
    result = problem.make_backend().sample_weight(
        np.array([0.0, 1.0]),
        seed=1,
        betas=(8.0,),
        n_warmup=12,
        n_samples=6,
        steps_per_sample=2,
    )
    # Five edges; a domain wall costs 1. Ground states sit at 0.
    assert float(result.objectives[:, 1].mean()) < 0.75
