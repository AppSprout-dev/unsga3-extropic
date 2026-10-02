"""Domain-wall Ising image of a uniform-K Potts chain.

This is the p-bit compilation in THRML example 03
(https://docs.thrml.ai/en/latest/03_codon_optimization.html): each
categorical site with ``K`` states becomes ``K - 1`` spins, state ``k`` is
the thermometer with the first ``k`` spins ``+1`` and the rest ``-1``, and
the Potts weights (the same ``W`` that ``CategoricalEBMFactor`` uses) become
Ising biases and couplings.

The formulas are the ones in that example's ``compile_dwc``:

- unary spin bias: first difference of the Potts bias, ``(W[k+1] - W[k]) / 2``
- pairwise spin coupling: second difference of the Potts coupling, divided by 4
- boundary corrections from the neighboring Potts coupling, also divided by 4
- ferromagnetic thermometer edges of weight ``P / 4``, plus the
  ``first_minus_last`` field that makes ``P`` state-independent on valid
  thermometers

``SpinEBMFactor`` energy is ``-sum W * s_1 * ... * s_m`` with ``s = ±1``
(https://docs.thrml.ai/en/latest/api-discrete-ebm.html), which is the
``beta = 1`` case of ``IsingEBM``. On a valid thermometer the Ising energy
equals the Potts factor energy up to a shift that does not depend on the
categorical state.

A spin pattern that is not a thermometer does not name a category. Callers
count those rows and leave them out of the feasible objective array. This
module does not vendor ``codon_opt`` and does not talk to a device.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DomainWallImage:
    """Static domain-wall layout plus the Potts-derived Ising couplings.

    ``bias_base`` and ``inter_weights`` do not include the thermometer
    penalty ``P`` or the inverse temperature. ``ising_terms`` adds ``P``.
    Beta stays outside, as it does for ``IsingEBM`` and for the example 03
    weight rescaling.
    """

    n_sites: int
    n_categories: int
    n_spins: int
    pos_of_spin: np.ndarray
    spin_pos_index: np.ndarray
    offset: np.ndarray
    bias_base: np.ndarray
    first_minus_last: np.ndarray
    constraint_edges: tuple[tuple[int, int], ...]
    inter_edges: tuple[tuple[int, int], ...]
    inter_weights: np.ndarray
    color_blocks: tuple[tuple[int, ...], ...]

    def encode(self, states: np.ndarray) -> np.ndarray:
        """Categorical states ``(..., n_sites)`` to thermometer spins ``±1``."""
        x = np.asarray(states, dtype=np.int64)
        if x.shape[-1] != self.n_sites:
            raise ValueError(
                f"states shape {x.shape} does not end with n_sites={self.n_sites}"
            )
        if x.size and (int(x.min()) < 0 or int(x.max()) >= self.n_categories):
            raise ValueError(
                f"states must lie in [0, {self.n_categories})"
            )
        chosen = x[..., self.pos_of_spin]
        return np.where(self.spin_pos_index < chosen, 1.0, -1.0)

    def plus_counts(self, spins_pm1: np.ndarray) -> np.ndarray:
        """Count of ``+1`` spins at each site. Not a category unless valid."""
        spins = self._as_spins(spins_pm1)
        counts = np.zeros(spins.shape[:-1] + (self.n_sites,), dtype=np.int64)
        for site in range(self.n_sites):
            start = int(self.offset[site])
            width = self.n_categories - 1
            if width == 0:
                continue
            counts[..., site] = np.count_nonzero(spins[..., start : start + width] > 0, axis=-1)
        return counts

    def valid_mask(self, spins_pm1: np.ndarray) -> np.ndarray:
        """True where every site is a single domain wall (a thermometer)."""
        spins = self._as_spins(spins_pm1)
        finite_pm1 = np.all((spins > 0.0) | (spins < 0.0), axis=-1)
        valid = finite_pm1
        width = self.n_categories - 1
        if width >= 2:
            for site in range(self.n_sites):
                start = int(self.offset[site])
                block = spins[..., start : start + width]
                defect = (block[..., :-1] < 0.0) & (block[..., 1:] > 0.0)
                valid = valid & ~np.any(defect, axis=-1)
        return valid

    def feasible(
        self,
        spins_pm1: np.ndarray,
        objective_fn,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Decoded states and Potts objectives for valid thermometers.

        Invalid rows are counted and omitted. Their ``+1`` count is not
        passed to ``objective_fn``, so it is not scored as a category.
        """
        spins = self._as_spins(spins_pm1)
        if spins.ndim == 1:
            spins = spins.reshape(1, self.n_spins)
        valid = self.valid_mask(spins)
        counts = self.plus_counts(spins)
        n_invalid = int((~valid).sum())
        good = counts[valid]
        if len(good) == 0:
            empty = np.zeros((0, self.n_sites), dtype=np.int64)
            probe = np.asarray(objective_fn(empty), dtype=np.float64)
            n_obj = int(probe.shape[-1]) if probe.ndim == 2 else 1
            return (
                empty,
                np.zeros((0, n_obj), dtype=np.float64),
                n_invalid,
            )
        objectives = np.asarray(objective_fn(good), dtype=np.float64)
        return good, objectives, n_invalid

    def objectives_with_nan(self, spins_pm1: np.ndarray, objective_fn) -> np.ndarray:
        """Potts objectives, with a non-finite row for each invalid thermometer."""
        spins = self._as_spins(spins_pm1)
        squeeze = spins.ndim == 1
        if squeeze:
            spins = spins.reshape(1, self.n_spins)
        valid = self.valid_mask(spins)
        counts = self.plus_counts(spins)
        if not np.any(valid):
            empty = np.zeros((0, self.n_sites), dtype=np.int64)
            probe = np.asarray(objective_fn(empty), dtype=np.float64)
            n_obj = int(probe.shape[-1]) if probe.ndim == 2 else 1
            out = np.full(valid.shape + (n_obj,), np.nan, dtype=np.float64)
        else:
            scored = np.asarray(objective_fn(counts[valid]), dtype=np.float64)
            out = np.full(valid.shape + (scored.shape[-1],), np.nan, dtype=np.float64)
            out[valid] = scored
        if squeeze:
            return out[0]
        return out

    def ising_terms(
        self, constraint_strength: float
    ) -> tuple[np.ndarray, tuple[tuple[int, int], ...], np.ndarray]:
        """Unscaled ``(biases, edges, couplings)`` at thermometer penalty ``P``.

        ``IsingEBM`` multiplies these by beta. A ``SpinEBMFactor`` program
        multiplies them by beta itself, which is what example 03 does.
        Constraint edges come first, then the between-site couplings.
        """
        strength = float(constraint_strength)
        if strength < 0.0:
            raise ValueError("constraint_strength must be >= 0")
        biases = self.bias_base + self.first_minus_last * (strength / 4.0)
        edges = self.constraint_edges + self.inter_edges
        weights = np.concatenate(
            [
                np.full(len(self.constraint_edges), strength / 4.0, dtype=np.float64),
                np.asarray(self.inter_weights, dtype=np.float64),
            ]
        )
        return biases, edges, weights

    def _as_spins(self, spins_pm1: np.ndarray) -> np.ndarray:
        spins = np.asarray(spins_pm1, dtype=np.float64)
        if spins.shape[-1] != self.n_spins:
            raise ValueError(
                f"spins shape {spins.shape} does not end with n_spins={self.n_spins}"
            )
        return spins


def compile_domain_wall(
    unary: np.ndarray,
    edges,
    pairwise: np.ndarray,
) -> DomainWallImage:
    """Compile THRML Potts weights on a chain into a domain-wall image.

    ``unary`` has shape ``(n_sites, K)`` and ``pairwise`` has shape
    ``(n_sites - 1, K, K)``. Both are weights in the THRML convention
    (factor energy ``-W[state]``), which is what ``PottsChainProblem.build_potts``
    returns. ``edges`` must be the path ``(0, 1), ..., (n-2, n-1)``.
    """
    weights = np.asarray(unary, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError(f"unary must have shape (n_sites, K), got {weights.shape}")
    n_sites, n_categories = int(weights.shape[0]), int(weights.shape[1])
    if n_sites < 1:
        raise ValueError("n_sites must be >= 1")
    if n_categories < 2:
        raise ValueError("n_categories must be >= 2 so the image has spins")
    expected = [(i, i + 1) for i in range(n_sites - 1)]
    clean = [(int(i), int(j)) for i, j in edges]
    if clean != expected:
        raise ValueError(
            "domain-wall compilation follows the example 03 chain; "
            f"edges must be {expected}, got {clean}"
        )
    pair = np.asarray(pairwise, dtype=np.float64)
    if pair.shape != (n_sites - 1, n_categories, n_categories):
        raise ValueError(
            f"pairwise shape {pair.shape} != "
            f"({n_sites - 1}, {n_categories}, {n_categories})"
        )
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(pair)):
        raise ValueError("Potts weights must be finite")

    ks = np.full(n_sites, n_categories, dtype=np.int64)
    pos_of_spin: list[int] = []
    spin_pos_index: list[int] = []
    for site, k_site in enumerate(ks):
        for spin_index in range(int(k_site) - 1):
            pos_of_spin.append(site)
            spin_pos_index.append(spin_index)
    pos = np.asarray(pos_of_spin, dtype=np.int64)
    local = np.asarray(spin_pos_index, dtype=np.int64)
    n_spins = int(pos.shape[0])
    if n_spins < 1:
        raise ValueError("domain-wall image has no spins")
    offset = np.zeros(n_sites, dtype=np.int64)
    for site in range(1, n_sites):
        offset[site] = offset[site - 1] + int(ks[site - 1]) - 1

    bias_base = np.zeros(n_spins, dtype=np.float64)
    first_minus_last = np.zeros(n_spins, dtype=np.float64)
    for spin in range(n_spins):
        site = int(pos[spin])
        local_index = int(local[spin])
        k_site = int(ks[site])
        field = (weights[site, local_index + 1] - weights[site, local_index]) / 2.0
        if local_index == 0:
            first_minus_last[spin] += 1.0
        if local_index == k_site - 2:
            first_minus_last[spin] -= 1.0
        if site > 0:
            left = pair[site - 1]
            k_left = int(ks[site - 1])
            field += (
                left[0, local_index + 1]
                - left[0, local_index]
                + left[k_left - 1, local_index + 1]
                - left[k_left - 1, local_index]
            ) / 4.0
        if site < n_sites - 1:
            right = pair[site]
            k_right = int(ks[site + 1])
            field += (
                right[local_index + 1, 0]
                - right[local_index, 0]
                + right[local_index + 1, k_right - 1]
                - right[local_index, k_right - 1]
            ) / 4.0
        bias_base[spin] = field

    constraint_edges = tuple(
        (int(offset[site] + local_index), int(offset[site] + local_index + 1))
        for site in range(n_sites)
        for local_index in range(int(ks[site]) - 2)
    )
    inter_edges_list: list[tuple[int, int]] = []
    inter_weights_list: list[float] = []
    for site in range(n_sites - 1):
        coupling = pair[site]
        for left_spin in range(int(ks[site]) - 1):
            for right_spin in range(int(ks[site + 1]) - 1):
                inter_edges_list.append(
                    (int(offset[site] + left_spin), int(offset[site + 1] + right_spin))
                )
                inter_weights_list.append(
                    (
                        coupling[left_spin + 1, right_spin + 1]
                        - coupling[left_spin, right_spin + 1]
                        - coupling[left_spin + 1, right_spin]
                        + coupling[left_spin, right_spin]
                    )
                    / 4.0
                )
    inter_edges = tuple(inter_edges_list)
    inter_weights = np.asarray(inter_weights_list, dtype=np.float64)

    grouped: dict[tuple[int, int], list[int]] = {}
    for spin in range(n_spins):
        key = (int(pos[spin]) % 2, int(local[spin]) % 2)
        grouped.setdefault(key, []).append(spin)
    color_blocks = tuple(tuple(indices) for indices in grouped.values())
    _reject_monochrome(constraint_edges + inter_edges, color_blocks)

    return DomainWallImage(
        n_sites=n_sites,
        n_categories=n_categories,
        n_spins=n_spins,
        pos_of_spin=pos,
        spin_pos_index=local,
        offset=offset,
        bias_base=bias_base,
        first_minus_last=first_minus_last,
        constraint_edges=constraint_edges,
        inter_edges=inter_edges,
        inter_weights=inter_weights,
        color_blocks=color_blocks,
    )


def ising_energy(
    spins_pm1: np.ndarray,
    biases: np.ndarray,
    edges: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    weights: np.ndarray,
) -> np.ndarray:
    """``-(b · s + sum J_ij s_i s_j)`` at beta = 1.

    This is ``SpinEBMFactor`` / ``IsingEBM`` energy for ``±1`` spins.
    """
    spins = np.asarray(spins_pm1, dtype=np.float64)
    field = np.tensordot(spins, np.asarray(biases, dtype=np.float64), axes=([-1], [0]))
    pair = np.zeros(spins.shape[:-1], dtype=np.float64)
    for (left, right), coupling in zip(edges, np.asarray(weights, dtype=np.float64)):
        pair = pair + float(coupling) * spins[..., left] * spins[..., right]
    return -(field + pair)


def checkerboard_permutation(
    n_spins: int,
    edges: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> np.ndarray | None:
    """Map each spin to a ``0 .. n-1`` index whose parity is a 2-coloring.

    ``ThrmlIsingBackend`` free blocks are ``nodes[::2]`` and ``nodes[1::2]``,
    so an edge must join opposite parities. Returns ``None`` when the graph
    is not 2-colored or the two color classes do not fit in those parities.
    The returned array sends the domain-wall spin index to the checkerboard
    index.
    """
    if n_spins < 1:
        raise ValueError("n_spins must be >= 1")
    adjacency: list[list[int]] = [[] for _ in range(n_spins)]
    for left, right in edges:
        i, j = int(left), int(right)
        if not (0 <= i < n_spins and 0 <= j < n_spins) or i == j:
            raise ValueError(f"edge {(i, j)} is not a pair inside 0..{n_spins - 1}")
        adjacency[i].append(j)
        adjacency[j].append(i)

    even_capacity = (n_spins + 1) // 2
    odd_capacity = n_spins // 2
    even_used = 0
    odd_used = 0
    color = np.full(n_spins, -1, dtype=np.int64)
    for start in range(n_spins):
        if color[start] != -1:
            continue
        relative: dict[int, int] = {start: 0}
        queue = [start]
        while queue:
            node = queue.pop()
            for neighbor in adjacency[node]:
                expected = 1 - relative[node]
                if neighbor not in relative:
                    relative[neighbor] = expected
                    queue.append(neighbor)
                elif relative[neighbor] != expected:
                    return None
        group_even_rel = [node for node, bit in relative.items() if bit == 0]
        group_odd_rel = [node for node, bit in relative.items() if bit == 1]
        if (
            even_used + len(group_even_rel) <= even_capacity
            and odd_used + len(group_odd_rel) <= odd_capacity
        ):
            even_group, odd_group = group_even_rel, group_odd_rel
        elif (
            even_used + len(group_odd_rel) <= even_capacity
            and odd_used + len(group_even_rel) <= odd_capacity
        ):
            even_group, odd_group = group_odd_rel, group_even_rel
        else:
            return None
        for node in even_group:
            color[node] = 0
        for node in odd_group:
            color[node] = 1
        even_used += len(even_group)
        odd_used += len(odd_group)

    even_slots = list(range(0, n_spins, 2))
    odd_slots = list(range(1, n_spins, 2))
    permutation = np.empty(n_spins, dtype=np.int64)
    even_cursor = 0
    odd_cursor = 0
    for spin in range(n_spins):
        if color[spin] == 0:
            permutation[spin] = even_slots[even_cursor]
            even_cursor += 1
        else:
            permutation[spin] = odd_slots[odd_cursor]
            odd_cursor += 1
    return permutation


def _reject_monochrome(
    edges: tuple[tuple[int, int], ...],
    color_blocks: tuple[tuple[int, ...], ...],
) -> None:
    color_of = [-1] * sum(len(block) for block in color_blocks)
    seen = 0
    for color, block in enumerate(color_blocks):
        for spin in block:
            if color_of[spin] != -1:
                raise RuntimeError(f"spin {spin} is in more than one color")
            color_of[spin] = color
            seen += 1
    if seen != len(color_of):
        raise RuntimeError("4-coloring does not cover every spin")
    for left, right in edges:
        if color_of[left] == color_of[right]:
            raise RuntimeError(
                f"domain-wall edge {(left, right)} lies inside one Gibbs block"
            )
