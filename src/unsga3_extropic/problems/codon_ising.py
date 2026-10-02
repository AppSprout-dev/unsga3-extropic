"""THRML-native multi-term Ising problem (codon-style, not ZDT1).

Two competing Ising-expressible energies over a spin chain:

  E_codon  = - sum_i h_i s_i          # preferred "codon" (unary biases)
  E_struct = - J sum_<i,j> s_i s_j    # ferromagnetic "structure" couplings

Scalarization E_w = w0 * E_codon + w1 * E_struct remains Ising:
  biases  = w0 * h
  J_edges = w1 * J   (nearest-neighbor)

Objectives reported for the archive are the raw energy terms
(f0, f1) = (E_codon, E_struct) — minimization. Soft global (host-adaptive
bias) can be folded into h via ``global_bias``.

This is the Extropic-native path: no surrogate fit, exact THRML energy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unsga3_extropic.backends.thrml_ising import ThrmlIsingBackend


@dataclass
class CodonIsingProblem:
    """Codon-style two-term Ising chain."""

    n_spins: int = 12
    J: float = 1.0
    seed: int = 0
    # optional soft global preference toward +1 (host-adaptive bias pattern)
    global_bias: float = 0.0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        # preferred codon: random ± preferred spins (heterogeneous unary fields)
        self.h = rng.choice(np.array([-1.0, 1.0]), size=self.n_spins).astype(np.float64)
        self.h = self.h + self.global_bias
        self.edges: list[tuple[int, int]] = [
            (i, i + 1) for i in range(self.n_spins - 1)
        ]

    def energies_from_spins(self, spins_pm1: np.ndarray) -> np.ndarray:
        """Return (..., 2) array of (E_codon, E_struct)."""
        s = np.asarray(spins_pm1, dtype=np.float64)
        # E_codon = - h · s
        e_codon = -np.tensordot(s, self.h, axes=([-1], [0]))
        # E_struct = - J sum s_i s_{i+1}
        e_struct = np.zeros(s.shape[:-1], dtype=np.float64)
        for i, j in self.edges:
            e_struct = e_struct - self.J * s[..., i] * s[..., j]
        return np.stack([e_codon, e_struct], axis=-1)

    def build_ising(self, w: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
        """Ising params for E_w = w0 E_codon + w1 E_struct.

        THRML IsingEBM uses E = -β (b·s + sum J_ij s_i s_j), so matching
        E_w = - w0 h·s - w1 J sum s_i s_j means:
          biases = w0 * h
          edge weights = w1 * J
        """
        w = np.asarray(w, dtype=np.float64)
        biases = (w[0] * self.h).astype(np.float32)
        j_weights = np.full(len(self.edges), w[1] * self.J, dtype=np.float32)
        return biases, list(self.edges), j_weights

    def make_backend(self) -> ThrmlIsingBackend:
        return ThrmlIsingBackend(
            n_spins=self.n_spins,
            build_ising=self.build_ising,
            objective_fn=self.energies_from_spins,
            spin_to_decision=lambda pm1: ((pm1 + 1.0) / 2.0).astype(np.float64),
        )

    def enumerate_front(self) -> tuple[np.ndarray, np.ndarray]:
        """Exact ND front by enumerating all 2^n configs (n_spins <= 16)."""
        if self.n_spins > 16:
            raise ValueError("enumerate_front only for n_spins <= 16")
        n = self.n_spins
        k = np.arange(2**n, dtype=np.uint32)[:, None]
        shifts = np.arange(n - 1, -1, -1, dtype=np.uint32)
        bits = ((k >> shifts) & 1).astype(np.float64)
        spins = 2.0 * bits - 1.0
        objs = self.energies_from_spins(spins)
        from unsga3_extropic.archive import nondominated_mask, unique_rows

        mask = nondominated_mask(objs)
        return bits[mask], unique_rows(objs[mask])
