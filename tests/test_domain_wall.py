"""Domain-wall encoding of the Potts chain. No JAX."""

from __future__ import annotations

import numpy as np
import pytest

from unsga3_extropic.domain_wall import (
    checkerboard_permutation,
    compile_domain_wall,
    ising_energy,
)
from unsga3_extropic.problems import PottsChainProblem


def _image(problem: PottsChainProblem, w: np.ndarray):
    unary, edges, pair = problem.build_potts(w)
    return compile_domain_wall(unary, edges, pair)


def test_decoded_thermometers_match_potts_objectives():
    problem = PottsChainProblem(n_sites=4, n_categories=3)
    w = np.array([0.25, 0.5])
    image = _image(problem, w)
    states = problem.enumerate_states()
    spins = image.encode(states)
    assert image.valid_mask(spins).all()
    assert np.array_equal(image.plus_counts(spins), states)
    decisions, objectives, n_invalid = image.feasible(
        spins, problem.energies_from_states
    )
    assert n_invalid == 0
    assert np.array_equal(decisions, states)
    assert np.allclose(objectives, problem.energies_from_states(states))


def test_invalid_thermometer_is_counted_and_not_scored():
    problem = PottsChainProblem(n_sites=3, n_categories=3)
    image = _image(problem, np.array([0.4, 0.6]))
    valid_state = np.array([0, 0, 2], dtype=np.int64)
    good = image.encode(valid_state)
    bad = good.copy()
    # Site 0 has spins 0, 1. [-1, +1] is not a thermometer; its +1 count is 1.
    bad[0] = -1.0
    bad[1] = 1.0
    assert image.plus_counts(bad)[0] == 1
    assert not bool(image.valid_mask(bad))
    batch = np.stack([good, bad], axis=0)
    seen: list[np.ndarray] = []

    def objective_fn(states: np.ndarray) -> np.ndarray:
        seen.append(np.asarray(states).copy())
        return problem.energies_from_states(states)

    decisions, objectives, n_invalid = image.feasible(batch, objective_fn)
    assert n_invalid == 1
    assert len(seen) == 1
    assert np.array_equal(seen[0], valid_state.reshape(1, -1))
    assert np.array_equal(decisions, valid_state.reshape(1, -1))
    assert np.allclose(objectives, problem.energies_from_states(valid_state))
    silent = image.plus_counts(bad)
    assert not np.array_equal(decisions, silent.reshape(1, -1))


def test_ising_energy_matches_potts_up_to_a_state_independent_shift():
    problem = PottsChainProblem(n_sites=3, n_categories=3)
    w = np.array([0.25, 0.5])
    image = _image(problem, w)
    states = problem.enumerate_states()
    spins = image.encode(states)
    potts = problem.energies_from_states(states) @ w
    for strength in (0.0, 4.0):
        biases, edges, couplings = image.ising_terms(strength)
        shift = ising_energy(spins, biases, edges, couplings) - potts
        assert np.allclose(shift, shift[0])

    bad = spins[0].copy()
    bad[0] = -1.0
    bad[1] = 1.0
    base = image.ising_terms(0.0)
    penalised = image.ising_terms(6.0)
    valid_gap = ising_energy(spins, *penalised) - ising_energy(spins, *base)
    invalid_gap = float(ising_energy(bad, *penalised) - ising_energy(bad, *base))
    assert np.allclose(valid_gap, valid_gap[0])
    assert invalid_gap > float(valid_gap[0]) + 1.0


def test_k3_chain_is_not_checkerboard_and_k2_chain_is():
    wide = PottsChainProblem(n_sites=4, n_categories=3)
    wide_image = _image(wide, np.array([1.0, 0.0]))
    wide_edges = wide_image.constraint_edges + wide_image.inter_edges
    assert checkerboard_permutation(wide_image.n_spins, wide_edges) is None
    assert wide.make_domain_wall_backend().program_kind(np.array([1.0, 0.0])) == "spin_ebm"
    assert len(wide_image.color_blocks) == 4

    narrow = PottsChainProblem(n_sites=5, n_categories=2)
    narrow_image = _image(narrow, np.ones(2))
    permutation = checkerboard_permutation(
        narrow_image.n_spins, narrow_image.constraint_edges + narrow_image.inter_edges
    )
    assert permutation is not None
    _biases, edges, _couplings = narrow_image.ising_terms(0.0)
    for left, right in edges:
        assert int(permutation[left]) % 2 != int(permutation[right]) % 2
    assert narrow.make_domain_wall_backend().program_kind(np.ones(2)) == "thrml_ising"


def test_compile_rejects_a_non_chain_and_a_negative_penalty():
    problem = PottsChainProblem(n_sites=3, n_categories=3)
    unary, _edges, pair = problem.build_potts(np.ones(2))
    with pytest.raises(ValueError, match="chain"):
        compile_domain_wall(unary, [(0, 2), (1, 2)], pair)
    image = _image(problem, np.ones(2))
    with pytest.raises(ValueError, match="constraint_strength"):
        image.ising_terms(-1.0)
