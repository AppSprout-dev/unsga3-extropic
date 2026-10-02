"""Reference-direction / weight-vector generation (diversity niches)."""

from __future__ import annotations

import numpy as np


def simplex_weights(n_weights: int, n_obj: int = 2) -> np.ndarray:
    """Generate ``n_weights`` non-negative weight vectors on the simplex.

    For 2 objectives this is a uniform grid on w1 + w2 = 1.
    For M>2 uses a Das–Dennis-style structured set when possible, else
    Dirichlet samples (seeded for reproducibility via ``rng``).
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

    # Das–Dennis for small H such that C(H+M-1, M-1) ~= n_weights
    H = 1
    while _n_das_dennis(H, n_obj) < n_weights:
        H += 1
        if H > 40:
            break
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
    from math import comb

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
