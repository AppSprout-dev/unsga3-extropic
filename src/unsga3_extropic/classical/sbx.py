"""Simulated binary crossover on a continuous box.

η = 30 and pair probability 1.0 match the Bend / C# PymooCompatible
operators (Deb & Agrawal, bounded SBX). Each variable is crossed with
probability 1/2. One uniform draw builds both children, then a coin may
swap them. Values within 1e-14 are copied. Results are clamped to the box.

NumPy only. Not a THRML operator and not used by ``WeightSweepLoop``.
"""

from __future__ import annotations

import numpy as np

_NEAR = 1e-14


def sbx(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    xl: np.ndarray | float,
    xu: np.ndarray | float,
    rng: np.random.Generator,
    *,
    eta: float = 30.0,
    prob: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross rows of ``parent_a`` with the matching rows of ``parent_b``.

    Both arrays have shape ``(n_pairs, n_var)``. ``prob`` is the probability
    that a pair enters SBX. ``prob <= 0`` returns copies of the parents.
    """
    if eta < 0:
        raise ValueError("eta must be >= 0")
    a = np.asarray(parent_a, dtype=np.float64)
    b = np.asarray(parent_b, dtype=np.float64)
    if a.ndim != 2 or b.shape != a.shape:
        raise ValueError(f"parents must share shape (n_pairs, n_var), got {a.shape} and {b.shape}")
    lower = np.broadcast_to(np.asarray(xl, dtype=np.float64), a.shape).copy()
    upper = np.broadcast_to(np.asarray(xu, dtype=np.float64), a.shape).copy()
    child_a = a.copy()
    child_b = b.copy()
    if prob <= 0.0 or a.shape[0] == 0:
        return child_a, child_b

    do_pair = rng.random(a.shape[0]) <= prob
    cross = rng.random(a.shape) <= 0.5
    cross &= do_pair[:, None]
    cross &= np.abs(a - b) >= _NEAR

    y1 = np.minimum(a, b)
    y2 = np.maximum(a, b)
    delta = np.maximum(y2 - y1, 0.0)
    delta_safe = np.where(cross, delta, 1.0)
    inv = 1.0 / (eta + 1.0)
    exponent = -(eta + 1.0)
    rand = rng.random(a.shape)

    def _alpha(gap: np.ndarray) -> np.ndarray:
        beta = 1.0 + 2.0 * np.maximum(gap, 0.0) / delta_safe
        return 2.0 - np.power(beta, exponent)

    alpha1 = _alpha(y1 - lower)
    alpha2 = _alpha(upper - y2)
    spread1 = _betaq(rand, alpha1, inv)
    spread2 = _betaq(rand, alpha2, inv)
    first = 0.5 * ((y1 + y2) - spread1 * delta)
    second = 0.5 * ((y1 + y2) + spread2 * delta)
    first = np.clip(first, lower, upper)
    second = np.clip(second, lower, upper)
    swap = rng.random(a.shape) > 0.5
    mixed_a = np.where(swap, second, first)
    mixed_b = np.where(swap, first, second)
    child_a = np.where(cross, mixed_a, a)
    child_b = np.where(cross, mixed_b, b)
    return child_a, child_b


def _betaq(rand: np.ndarray, alpha: np.ndarray, inv: float) -> np.ndarray:
    """Deb's polynomial spread. ``rand <= 1/alpha`` uses the lower branch."""
    alpha_safe = np.maximum(alpha, 1e-16)
    use_lo = rand <= (1.0 / alpha_safe)
    lo = np.power(np.maximum(rand * alpha_safe, 0.0), inv)
    hi_base = np.maximum(2.0 - rand * alpha_safe, 1e-16)
    hi = np.power(1.0 / hi_base, inv)
    return np.where(use_lo, lo, hi)
