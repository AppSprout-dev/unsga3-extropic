"""Two-term Potts chain sampled with THRML categorical factors.

A short chain of variables in ``{0, ..., K-1}`` with two minimization
objectives:

    f_unary    = sum_i U[i, x_i]
    f_pairwise = sum_{i=0}^{n-2} P[x_i, x_{i+1}]

``U[i, k] = 0`` when ``k == i mod K`` and ``1`` otherwise, so site ``i``
prefers residue ``i mod K``. ``P`` is ``0`` on the diagonal and ``1`` off
it, so neighbors prefer to agree. The two terms compete.

``CategoricalEBMFactor`` contributes ``-W[state]`` (THRML 0.1.4), so the
weights passed to the sampler are::

    W_unary[i, k] = -w0 * U[i, k]
    W_pair[e, a, b] = -w1 * P[a, b]

``FactorizedEBM.energy`` on those weights equals ``w · f``. Annealing
multiplies ``W`` by beta inside ``ThrmlPottsBackend``.

This is an in-repo smoke. It is not the codon-optimization walkthrough.
``make_domain_wall_backend`` is the p-bit image of this same energy
(example 03 domain-wall encoding). That path is a THRML simulation, not a
device run and not a copy of ``codon_opt``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unsga3_extropic.archive import nondominated_mask
from unsga3_extropic.backends.thrml_domain_wall import ThrmlDomainWallBackend
from unsga3_extropic.backends.thrml_potts import (
    ThrmlPottsBackend,
    even_odd_coloring,
)

# Enumeration stays cheap: 3**8 = 6561, 4**6 = 4096, 3**9 = 19683 is refused.
_MAX_ENUM = 10_000


@dataclass
class PottsChainProblem:
    """Unary-plus-pairwise categorical chain."""

    n_sites: int = 6
    n_categories: int = 3

    def __post_init__(self) -> None:
        if self.n_sites < 2:
            raise ValueError("n_sites must be >= 2")
        if not (2 <= int(self.n_categories) <= 256):
            raise ValueError("n_categories must lie in 2..256")
        n = self.n_sites
        k = int(self.n_categories)
        preferred = np.arange(n) % k
        unary = np.ones((n, k), dtype=np.float64)
        unary[np.arange(n), preferred] = 0.0
        pair = np.ones((k, k), dtype=np.float64)
        np.fill_diagonal(pair, 0.0)
        self.preferred = preferred.astype(np.int64)
        self.unary = unary
        self.pair = pair
        self.edges: list[tuple[int, int]] = [(i, i + 1) for i in range(n - 1)]

    def coloring(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Explicit even/odd free blocks for this path."""
        return even_odd_coloring(self.n_sites)

    def energies_from_states(self, states: np.ndarray) -> np.ndarray:
        """Return ``(..., 2)`` array of ``(f_unary, f_pairwise)``."""
        x = np.asarray(states, dtype=np.int64)
        if x.shape[-1] != self.n_sites:
            raise ValueError(
                f"states shape {x.shape} does not end with n_sites={self.n_sites}"
            )
        sites = np.arange(self.n_sites)
        f_unary = self.unary[sites, x].sum(axis=-1)
        f_pair = np.zeros(x.shape[:-1], dtype=np.float64)
        for i, j in self.edges:
            f_pair = f_pair + self.pair[x[..., i], x[..., j]]
        return np.stack([np.asarray(f_unary, dtype=np.float64), f_pair], axis=-1)

    def build_potts(
        self, w: np.ndarray
    ) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
        """THRML weights for ``E_w = w0 f_unary + w1 f_pairwise``.

        Factor energy is ``-W[state]``, so the weights are the negated
        scalarized costs. Beta is not folded in; the backend rescales ``W``.
        """
        w = np.asarray(w, dtype=np.float64)
        unary_w = (-w[0] * self.unary).astype(np.float32)
        pair_w = np.stack(
            [(-w[1] * self.pair).astype(np.float32) for _ in self.edges],
            axis=0,
        )
        return unary_w, list(self.edges), pair_w

    def make_backend(self) -> ThrmlPottsBackend:
        return ThrmlPottsBackend(
            n_sites=self.n_sites,
            n_categories=self.n_categories,
            build_potts=self.build_potts,
            objective_fn=self.energies_from_states,
            coloring=self.coloring(),
        )

    def make_domain_wall_backend(
        self, *, constraint_strength: float = 4.0
    ) -> ThrmlDomainWallBackend:
        """Ising image of this Potts energy (domain-wall / thermometer spins)."""
        return ThrmlDomainWallBackend(
            n_sites=self.n_sites,
            n_categories=self.n_categories,
            build_potts=self.build_potts,
            objective_fn=self.energies_from_states,
            constraint_strength=constraint_strength,
        )

    def enumerate_states(self) -> np.ndarray:
        """Every configuration, shape ``(K**n, n)``, or raise if that is large."""
        count = int(self.n_categories) ** int(self.n_sites)
        if count > _MAX_ENUM:
            raise ValueError(
                "enumerate_front only when n_categories ** n_sites "
                f"<= {_MAX_ENUM} (got {count})"
            )
        idx = np.arange(count, dtype=np.int64)[:, None]
        scales = self.n_categories ** np.arange(
            self.n_sites - 1, -1, -1, dtype=np.int64
        )
        return (idx // scales) % int(self.n_categories)

    def enumerate_front(self) -> tuple[np.ndarray, np.ndarray]:
        """Exact non-dominated front (one decision per unique objective)."""
        states = self.enumerate_states()
        objs = self.energies_from_states(states)
        mask = nondominated_mask(objs)
        states_nd = states[mask]
        objs_nd = objs[mask]
        kept_x: list[np.ndarray] = []
        kept_f: list[np.ndarray] = []
        seen: set[tuple[float, ...]] = set()
        for row_x, row_f in zip(states_nd, objs_nd):
            key = tuple(np.round(row_f, 6).tolist())
            if key in seen:
                continue
            seen.add(key)
            kept_x.append(row_x)
            kept_f.append(row_f)
        if not kept_f:
            return states_nd[:0], objs_nd[:0]
        return np.stack(kept_x, axis=0), np.stack(kept_f, axis=0)
