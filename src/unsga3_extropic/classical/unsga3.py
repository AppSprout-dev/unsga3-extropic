"""Generational continuous U-NSGA-III.

Population of size N, Das–Dennis directions (H), non-dominated ranking,
association by perpendicular distance in normalized objective space, and
NSGA-III niching to fill the next population. Mating is the PymooCompatible
tournament: when two parents share a niche, the better rank wins and a rank
tie prefers the smaller perpendicular distance; otherwise the winner is a
coin flip. Variation is SBX (η=30, p_c=1) and polynomial mutation
(η=20, p_m=1/n).

Objective calls equal ``N * n_gen``: the initial population is generation 1,
and each later generation evaluates N offspring. Decision variables stay in
the box, ``[0, 1]`` for ZDT and DTLZ2.

NumPy only. This module does not import THRML, does not build an Ising or
Potts energy, and is not called by ``WeightSweepLoop``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unsga3_extropic.archive import nondominated_mask, perpendicular_distances
from unsga3_extropic.classical.pm import polynomial_mutation
from unsga3_extropic.classical.sbx import sbx
from unsga3_extropic.weights import das_dennis_directions

_ASF_EPS = 1e-6
_DUP_DECIMALS = 12
_DUP_RETRIES = 20


@dataclass(frozen=True)
class UnsGa3Result:
    """Final population and its non-dominated objective rows."""

    decisions: np.ndarray
    objectives: np.ndarray
    front: np.ndarray
    n_evals: int
    pop_size: int
    n_gen: int
    partitions: int
    n_directions: int


def run_unsga3(
    objective,
    *,
    n_var: int,
    n_obj: int,
    pop_size: int,
    n_gen: int,
    partitions: int,
    seed: int,
    lower: float = 0.0,
    upper: float = 1.0,
    sbx_eta: float = 30.0,
    sbx_prob: float = 1.0,
    pm_eta: float = 20.0,
    pm_prob: float | None = None,
) -> UnsGa3Result:
    """Minimize ``objective(X) -> (n, n_obj)`` for ``n_gen`` generations.

    ``pop_size`` must be at least the Das–Dennis count and even, so parents
    pair as ``(0, 1), (2, 3), ...``. ``objective`` is called on the initial
    population and on each offspring batch.
    """
    if n_var < 1:
        raise ValueError("n_var must be >= 1")
    if n_obj < 2:
        raise ValueError("n_obj must be >= 2")
    if n_gen < 1:
        raise ValueError("n_gen must be >= 1")
    if upper <= lower:
        raise ValueError("upper must be greater than lower")
    directions = das_dennis_directions(n_obj, partitions)
    n_directions = int(directions.shape[0])
    if pop_size < n_directions:
        raise ValueError(
            f"pop_size {pop_size} is below the Das–Dennis count {n_directions}"
        )
    if pop_size % 2 != 0:
        raise ValueError("pop_size must be even so SBX pairs cover the population")
    rate = (1.0 / n_var) if pm_prob is None else float(pm_prob)
    rng = np.random.default_rng(seed)
    xl = float(lower)
    xu = float(upper)
    decisions = rng.uniform(xl, xu, size=(pop_size, n_var))
    objectives = _evaluate(objective, decisions, n_obj)
    n_evals = pop_size
    ideal = np.full(n_obj, np.inf, dtype=np.float64)
    for _generation in range(1, n_gen):
        ranks, assoc, dist, ideal = _prepare(objectives, directions, ideal)
        children = _offspring(
            decisions,
            ranks,
            assoc,
            dist,
            xl,
            xu,
            rng,
            sbx_eta=sbx_eta,
            sbx_prob=sbx_prob,
            pm_eta=pm_eta,
            pm_prob=rate,
        )
        child_obj = _evaluate(objective, children, n_obj)
        n_evals += pop_size
        merged_x = np.vstack([decisions, children])
        merged_f = np.vstack([objectives, child_obj])
        decisions, objectives, ideal = _survive(
            merged_x, merged_f, pop_size, directions, rng, ideal
        )
    return UnsGa3Result(
        decisions=decisions,
        objectives=objectives,
        front=nondominated_objectives(objectives),
        n_evals=n_evals,
        pop_size=pop_size,
        n_gen=n_gen,
        partitions=partitions,
        n_directions=n_directions,
    )


def nondominated_objectives(objectives: np.ndarray) -> np.ndarray:
    """Non-dominated rows, dropping duplicates at 12 decimal places."""
    rows = np.asarray(objectives, dtype=np.float64)
    if len(rows) == 0:
        return rows.reshape(0, rows.shape[1] if rows.ndim == 2 else 0)
    kept = rows[nondominated_mask(rows)]
    if len(kept) == 0:
        return kept
    _, index = np.unique(np.round(kept, decimals=_DUP_DECIMALS), axis=0, return_index=True)
    return kept[np.sort(index)]


def pymoo_tournament_index(
    ranks: np.ndarray,
    association: np.ndarray,
    distance: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Index of the PymooCompatible tournament winner. Feasible points only."""
    n = int(len(ranks))
    if n < 2:
        raise ValueError("tournament needs at least two individuals")
    left = int(rng.integers(0, n))
    right = int(rng.integers(0, n - 1))
    if right >= left:
        right += 1
    return _winner(left, right, ranks, association, distance, rng)


def _winner(
    left: int,
    right: int,
    ranks: np.ndarray,
    association: np.ndarray,
    distance: np.ndarray,
    rng: np.random.Generator,
) -> int:
    if int(association[left]) != int(association[right]):
        return left if rng.random() < 0.5 else right
    if int(ranks[left]) < int(ranks[right]):
        return left
    if int(ranks[right]) < int(ranks[left]):
        return right
    if float(distance[left]) < float(distance[right]):
        return left
    if float(distance[right]) < float(distance[left]):
        return right
    return left if rng.random() < 0.5 else right


def _evaluate(objective, decisions: np.ndarray, n_obj: int) -> np.ndarray:
    values = np.asarray(objective(decisions), dtype=np.float64)
    if values.shape != (len(decisions), n_obj):
        raise ValueError(
            f"objective returned {values.shape}, expected {(len(decisions), n_obj)}"
        )
    return values


def _offspring(
    decisions: np.ndarray,
    ranks: np.ndarray,
    association: np.ndarray,
    distance: np.ndarray,
    xl: float,
    xu: float,
    rng: np.random.Generator,
    *,
    sbx_eta: float,
    sbx_prob: float,
    pm_eta: float,
    pm_prob: float,
) -> np.ndarray:
    n = len(decisions)
    parents = np.fromiter(
        (
            pymoo_tournament_index(ranks, association, distance, rng)
            for _ in range(n)
        ),
        dtype=int,
        count=n,
    )
    first, second = sbx(
        decisions[parents[0::2]],
        decisions[parents[1::2]],
        xl,
        xu,
        rng,
        eta=sbx_eta,
        prob=sbx_prob,
    )
    children = np.empty_like(decisions)
    children[0::2] = first
    children[1::2] = second
    children = polynomial_mutation(children, xl, xu, rng, eta=pm_eta, prob=pm_prob)
    return _repair_duplicates(decisions, children, xl, xu, rng, pm_eta=pm_eta, pm_prob=pm_prob)


def _repair_duplicates(
    parents: np.ndarray,
    children: np.ndarray,
    xl: float,
    xu: float,
    rng: np.random.Generator,
    *,
    pm_eta: float,
    pm_prob: float,
) -> np.ndarray:
    """Polynomial-mutate a child whose 12-decimal key is already used.

    The repaired row is what gets evaluated. Extra mutation attempts are
    not extra objective calls. After ``_DUP_RETRIES`` the child is kept.
    """
    seen = {_decision_key(row) for row in parents}
    repaired = children.copy()
    for index in range(len(repaired)):
        for _attempt in range(_DUP_RETRIES):
            if _decision_key(repaired[index]) not in seen:
                break
            repaired[index : index + 1] = polynomial_mutation(
                repaired[index : index + 1],
                xl,
                xu,
                rng,
                eta=pm_eta,
                prob=1.0,
            )
        seen.add(_decision_key(repaired[index]))
    return repaired


def _decision_key(row: np.ndarray) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(row, dtype=np.float64), decimals=_DUP_DECIMALS).tolist())


def _prepare(
    objectives: np.ndarray,
    directions: np.ndarray,
    ideal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ranks and niche geometry for mating. Updates the running ideal."""
    updated = np.minimum(ideal, np.min(objectives, axis=0))
    ranks = _ranks(objectives)
    front0 = np.flatnonzero(ranks == 0)
    translated = objectives - updated
    intercepts = _intercepts(np.maximum(translated[front0], 0.0), np.maximum(translated, 0.0))
    normalized = np.maximum(translated, 0.0) / intercepts
    association, distance = _associate(normalized, directions)
    return ranks, association, distance, updated


def _survive(
    decisions: np.ndarray,
    objectives: np.ndarray,
    n_survive: int,
    directions: np.ndarray,
    rng: np.random.Generator,
    ideal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    updated = np.minimum(ideal, np.min(objectives, axis=0))
    fronts = _fronts(objectives)
    selected: list[int] = []
    splitting: list[int] | None = None
    for front in fronts:
        if len(selected) + len(front) <= n_survive:
            selected.extend(front)
        else:
            splitting = front
            break
    if splitting is None or len(selected) >= n_survive:
        chosen = np.asarray(selected[:n_survive], dtype=int)
        return decisions[chosen], objectives[chosen], updated

    considered = np.asarray(selected + splitting, dtype=int)
    translated = np.maximum(objectives - updated, 0.0)
    front0 = np.asarray(fronts[0], dtype=int)
    intercepts = _intercepts(translated[front0], translated[considered])
    normalized = translated / intercepts
    association, distance = _associate(normalized[considered], directions)
    position = {int(index): slot for slot, index in enumerate(considered)}
    niche_count = np.zeros(len(directions), dtype=int)
    for index in selected:
        niche_count[association[position[index]]] += 1
    picked = _niching(
        splitting,
        association,
        distance,
        position,
        niche_count,
        n_survive - len(selected),
        rng,
    )
    chosen = np.asarray(selected + picked, dtype=int)
    return decisions[chosen], objectives[chosen], updated


def _niching(
    splitting: list[int],
    association: np.ndarray,
    distance: np.ndarray,
    position: dict[int, int],
    niche_count: np.ndarray,
    n_pick: int,
    rng: np.random.Generator,
) -> list[int]:
    """Fill ``n_pick`` seats from the splitting front.

    The niche with the fewest already accepted members is next. A count of
    zero takes the closest remaining member of that niche. A positive count
    takes one of them at random.
    """
    remaining = set(splitting)
    picked: list[int] = []
    while len(picked) < n_pick and remaining:
        members: dict[int, list[int]] = {}
        for index in remaining:
            members.setdefault(int(association[position[index]]), []).append(index)
        counts = {niche: int(niche_count[niche]) for niche in members}
        smallest = min(counts.values())
        niches = [niche for niche, count in counts.items() if count == smallest]
        niche = int(niches[0] if len(niches) == 1 else rng.choice(niches))
        pool = members[niche]
        if niche_count[niche] == 0:
            choice = min(pool, key=lambda index: (float(distance[position[index]]), index))
        else:
            choice = int(pool[0] if len(pool) == 1 else rng.choice(pool))
        picked.append(choice)
        remaining.remove(choice)
        niche_count[niche] += 1
    return picked


def _associate(
    normalized: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    distances = perpendicular_distances(normalized, directions)
    association = np.argmin(distances, axis=1).astype(int)
    perp = distances[np.arange(len(normalized)), association]
    return association, perp


def _intercepts(front: np.ndarray, considered: np.ndarray) -> np.ndarray:
    """NSGA-III hyperplane intercepts, with the nadir as the fallback."""
    n_obj = front.shape[1]
    nadir = considered.max(axis=0) if len(considered) else np.ones(n_obj)
    nadir = np.where(nadir <= 1e-12, 1.0, nadir)
    if len(front) < n_obj:
        return nadir
    weights = np.eye(n_obj, dtype=np.float64)
    weights[weights == 0.0] = _ASF_EPS
    asf = np.max(front[:, None, :] / weights[None, :, :], axis=2)
    extreme = front[np.argmin(asf, axis=0)]
    try:
        coef = np.linalg.solve(extreme, np.ones(n_obj, dtype=np.float64))
    except np.linalg.LinAlgError:
        return nadir
    if not np.all(np.isfinite(coef)) or np.any(coef <= 1e-12):
        return nadir
    intercepts = 1.0 / coef
    if not np.all(np.isfinite(intercepts)) or np.any(intercepts <= 1e-12):
        return nadir
    return intercepts


def _ranks(objectives: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(objectives), dtype=int)
    for rank, front in enumerate(_fronts(objectives)):
        ranks[np.asarray(front, dtype=int)] = rank
    return ranks


def _fronts(objectives: np.ndarray) -> list[list[int]]:
    """Non-dominated fronts, rank 0 first. Minimization."""
    rows = np.asarray(objectives, dtype=np.float64)
    n = len(rows)
    if n == 0:
        return []
    less_equal = np.all(rows[:, None, :] <= rows[None, :, :] + 1e-12, axis=2)
    strictly = np.any(rows[:, None, :] < rows[None, :, :] - 1e-12, axis=2)
    dominates = less_equal & strictly
    np.fill_diagonal(dominates, False)
    dominated_by = dominates.sum(axis=0).astype(int)
    children: list[list[int]] = [[] for _ in range(n)]
    for winner, loser in zip(*np.nonzero(dominates)):
        children[int(winner)].append(int(loser))
    current = np.flatnonzero(dominated_by == 0).tolist()
    fronts: list[list[int]] = []
    while current:
        fronts.append(current)
        nxt: list[int] = []
        for winner in current:
            for loser in children[winner]:
                dominated_by[loser] -= 1
                if dominated_by[loser] == 0:
                    nxt.append(loser)
        current = nxt
    if sum(len(front) for front in fronts) != n:
        missing = [index for index in range(n) if dominated_by[index] > 0]
        if missing:
            fronts.append(missing)
    return fronts
