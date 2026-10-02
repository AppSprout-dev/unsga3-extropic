"""Codon Ising energy consistency (no THRML sample required)."""

from __future__ import annotations

import numpy as np

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
