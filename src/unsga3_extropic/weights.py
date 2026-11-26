"""Reference-direction / weight-vector generation (diversity niches)."""

from __future__ import annotations

from math import comb

import numpy as np


def simplex_weights(n_weights: int, n_obj: int = 2) -> np.ndarray:
    """Generate ``n_weights`` non-negative weight vectors on the simplex.

    For 2 objectives this is a uniform grid on w1 + w2 = 1.
    For M > 2 uses a Das–Dennis structured set when its size stays bounded,
    falling back to deterministically seeded Dirichlet samples to avoid
    combinatorial blowup at high M.
    """
    if n_obj < 2:
        raise ValueError("n_obj must be >= 2")
    if n_weights < 1:
        raise ValueError("n_weights must be >= 1")

    if n_obj == 2:
        if n_weights == 1:
            return np.array([[0.5, 0.5]], dtype=np.float64)
        w1 = np.linspace(0.0, 1.0, n_weights)
        # avoid pure (0,1)/(1,0) extremes if n_weights>=3 — keep endpoints
        # for niche coverage (U-NSGA-III reference directions include extremes)
        w = np.stack([w1, 1.0 - w1], axis=1)
        return w.astype(np.float64)

    # Das–Dennis for small H such that C(H+M-1, M-1) stays bounded.
    max_lattice = max(int(n_weights) * 4, 10_000)
    if _n_das_dennis(1, n_obj) > max_lattice:
        rng = np.random.default_rng(0)
        return rng.dirichlet(np.ones(n_obj), size=n_weights).astype(np.float64)

    H = 1
    while _n_das_dennis(H, n_obj) < n_weights and H < 40:
        if _n_das_dennis(H + 1, n_obj) > max_lattice:
            break
        H += 1
    pts = _das_dennis(H, n_obj)
    if len(pts) > n_weights:
        # subsample evenly
        idx = np.linspace(0, len(pts) - 1, n_weights).astype(int)
        pts = pts[idx]
    elif len(pts) < n_weights:
        rng = np.random.default_rng(0)
        extra = rng.dirichlet(np.ones(n_obj), size=n_weights - len(pts))
        pts = np.vstack([pts, extra])
    return pts.astype(np.float64)


def _n_das_dennis(H: int, M: int) -> int:
    # number of points = C(H+M-1, M-1)
    return comb(H + M - 1, M - 1)


def _das_dennis(H: int, M: int) -> np.ndarray:
    """Structured reference points on the unit simplex (Das & Dennis)."""
    pts: list[list[float]] = []

    def rec(left: int, depth: int, cur: list[float]) -> None:
        if depth == M - 1:
            pts.append(cur + [left / H])
            return
        for i in range(left + 1):
            rec(left - i, depth + 1, cur + [i / H])

    if H == 0:
        return np.ones((1, M)) / M
    rec(H, 0, [])
    return np.asarray(pts, dtype=np.float64)


def das_dennis_directions(n_obj: int, partitions: int) -> np.ndarray:
    """Every Das–Dennis point with ``partitions`` divisions.

    The count is ``C(partitions + n_obj - 1, n_obj - 1)``. Two objectives
    at ``partitions = 12`` are 13 points, the same set as
    ``simplex_weights(13, 2)``. Three objectives are 91 points. This is
    the reference set the classical continuous column associates to. It
    does not draw a second, random simplex.
    """
    if n_obj < 2:
        raise ValueError("n_obj must be >= 2")
    if partitions < 1:
        raise ValueError("partitions must be >= 1")
    pts = _das_dennis(int(partitions), int(n_obj))
    if len(pts) != _n_das_dennis(int(partitions), int(n_obj)):
        raise AssertionError("Das–Dennis count does not match the binomial size")
    return pts.astype(np.float64)
