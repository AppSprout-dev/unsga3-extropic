"""Codon Ising energy consistency (no THRML sample required)."""

from __future__ import annotations

import numpy as np
import pytest

from unsga3_extropic.archive import nondominated_mask
from unsga3_extropic.problems import CodonIsingProblem


def test_build_ising_matches_energy():
    p = CodonIsingProblem(n_spins=8, J=1.0, seed=1)
    w = np.array([0.4, 0.6])
    biases, edges, jw = p.build_ising(w)
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = rng.choice([-1.0, 1.0], size=p.n_spins)
        objs = p.energies_from_spins(s)
        e_w = float(w @ objs)
        # THRML form: E = -(b·s + sum Jij si sj)  (beta absorbed in schedule)
        e_thrml = -float(biases @ s)
        for (i, j), J in zip(edges, jw):
            e_thrml -= float(J) * s[i] * s[j]
        assert abs(e_w - e_thrml) < 1e-5


def test_aligned_and_antialigned_energies():
    problem = CodonIsingProblem(n_spins=5, J=2.0, seed=2, global_bias=0.0)
    plus = np.ones(5)
    e_codon, e_struct = problem.energies_from_spins(plus)
    assert e_codon == pytest.approx(-problem.h.sum())
    assert e_struct == pytest.approx(-2.0 * 4)
    anti = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    _e_codon, e_anti = problem.energies_from_spins(anti)
    assert e_anti == pytest.approx(8.0)


def test_global_bias_and_weight_scaling():
    raw = np.random.default_rng(0).choice(np.array([-1.0, 1.0]), size=4)
    problem = CodonIsingProblem(n_spins=4, J=1.5, seed=0, global_bias=0.3)
    assert np.allclose(problem.h, raw + 0.3)
    biases, edges, couplings = problem.build_ising(np.array([0.25, 0.5]))
    assert edges == [(0, 1), (1, 2), (2, 3)]
    assert np.allclose(biases, 0.25 * problem.h)
    assert np.allclose(couplings, np.full(3, 0.75))


def test_enumerate_front_is_aligned_and_nondominated():
    problem = CodonIsingProblem(n_spins=6, J=1.0, seed=3, global_bias=0.05)
    bits, front = problem.enumerate_front()
    assert len(bits) == len(front) >= 1
    assert np.all(nondominated_mask(front))
    spins = 2.0 * bits - 1.0
    assert np.allclose(problem.energies_from_spins(spins), front)

    flat = CodonIsingProblem(n_spins=2, J=0.0, seed=0)
    flat.h[:] = 0.0
    flat_bits, flat_front = flat.enumerate_front()
    assert len(flat_bits) == len(flat_front) == 1
    assert np.allclose(flat_front, [[0.0, 0.0]])

    with pytest.raises(ValueError, match="n_spins"):
        CodonIsingProblem(n_spins=17).enumerate_front()
