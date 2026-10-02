"""THRML API contract and a short Ising smoke (skipped without jax/thrml)."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("thrml")

from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

from unsga3_extropic.problems import CodonIsingProblem

pytestmark = pytest.mark.thrml


def test_docs_quickstart_sample_shape():
    """Same construction as the docs.thrml.ai / THRML README quickstart."""
    nodes = [SpinNode() for _ in range(5)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(4)]
    model = IsingEBM(
        nodes,
        edges,
        jnp.zeros((5,)),
        jnp.ones((4,)) * 0.5,
        jnp.array(1.0),
    )
    free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
    k_init, k_samp = jax.random.split(jax.random.key(0), 2)
    state = hinton_init(k_init, model, free_blocks, ())
    schedule = SamplingSchedule(n_warmup=2, n_samples=4, steps_per_sample=2)
    samples = sample_states(
        k_samp, program, schedule, state, [], [Block(nodes)]
    )
    assert len(samples) == 1
    assert tuple(samples[0].shape) == (4, 5)
    assert samples[0].dtype == np.dtype(bool)


def test_backend_shapes_energy_and_seed():
    problem = CodonIsingProblem(n_spins=6, J=1.0, seed=1)
    backend = problem.make_backend()
    w = np.array([0.4, 0.6])
    kwargs = dict(
        seed=3,
        betas=(1.0, 4.0),
        n_warmup=2,
        n_samples=4,
        steps_per_sample=1,
    )
    first = backend.sample_weight(w, **kwargs)
    second = backend.sample_weight(w, **kwargs)
    assert first.decisions.shape == (8, 6)
    assert first.objectives.shape == (8, 2)
    assert first.energies.shape == (8,)
    assert set(np.unique(first.decisions)).issubset({0.0, 1.0})
    spins = 2.0 * first.decisions - 1.0
    assert np.allclose(first.objectives, problem.energies_from_spins(spins))
    assert np.allclose(first.energies, first.objectives @ w)
    assert np.array_equal(first.decisions, second.decisions)


def test_positive_field_prefers_plus_one():
    problem = CodonIsingProblem(n_spins=4, J=1.0, seed=0)
    problem.h[:] = 1.0
    result = problem.make_backend().sample_weight(
        np.array([1.0, 0.0]),
        seed=2,
        betas=(8.0,),
        n_warmup=4,
        n_samples=8,
        steps_per_sample=1,
    )
    spins = 2.0 * result.decisions - 1.0
    assert float((spins > 0).mean()) > 0.95
    # h = +1 on every site, so E_codon = -sum s_i is -n when every spin is +1.
    assert float(result.objectives[:, 0].mean()) < -3.5


def test_ferromagnetic_chain_reaches_ground_structure():
    problem = CodonIsingProblem(n_spins=6, J=1.0, seed=0)
    problem.h[:] = 0.0
    result = problem.make_backend().sample_weight(
        np.array([0.0, 1.0]),
        seed=1,
        betas=(8.0,),
        n_warmup=12,
        n_samples=8,
        steps_per_sample=2,
    )
    # Perfect ferro alignment has E_struct = -(n - 1).
    assert float(result.objectives[:, 1].mean()) < -4.5
