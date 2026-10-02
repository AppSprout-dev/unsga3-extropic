"""Polynomial mutation on a continuous box.

η = 20. The per-variable probability used by the yardstick is ``1 / n``.
A variable whose bounds differ by less than 1e-14 is left unchanged.
The result is clamped to the box.

NumPy only. Not a THRML operator and not used by ``WeightSweepLoop``.
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-14


def polynomial_mutation(
    population: np.ndarray,
    xl: np.ndarray | float,
    xu: np.ndarray | float,
    rng: np.random.Generator,
    *,
    eta: float = 20.0,
    prob: float | None = None,
) -> np.ndarray:
    """Mutate each row. ``prob`` defaults to ``1 / n_var``. ``prob <= 0`` copies."""
    if eta < 0:
        raise ValueError("eta must be >= 0")
    rows = np.asarray(population, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError(f"population must be 2-D, got {rows.shape}")
    n_var = rows.shape[1]
    rate = (1.0 / n_var) if prob is None else float(prob)
    out = rows.copy()
    if rate <= 0.0 or rows.shape[0] == 0 or n_var == 0:
        return out
    lower = np.broadcast_to(np.asarray(xl, dtype=np.float64), rows.shape)
    upper = np.broadcast_to(np.asarray(xu, dtype=np.float64), rows.shape)
    span = upper - lower
    mutate = rng.random(rows.shape) <= rate
    mutate &= span >= _TINY
    rand = rng.random(rows.shape)
    span_safe = np.where(span >= _TINY, span, 1.0)
    delta1 = np.clip((rows - lower) / span_safe, 0.0, 1.0)
    delta2 = np.clip((upper - rows) / span_safe, 0.0, 1.0)
    mut_pow = 1.0 / (eta + 1.0)
    power = eta + 1.0
    xy1 = np.clip(1.0 - delta1, 0.0, None)
    val_lo = 2.0 * rand + (1.0 - 2.0 * rand) * np.power(xy1, power)
    dq_lo = np.power(np.clip(val_lo, 0.0, None), mut_pow) - 1.0
    xy2 = np.clip(1.0 - delta2, 0.0, None)
    val_hi = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * np.power(xy2, power)
    dq_hi = 1.0 - np.power(np.clip(val_hi, 0.0, None), mut_pow)
    delta_q = np.where(rand < 0.5, dq_lo, dq_hi)
    mutated = np.clip(rows + delta_q * span, lower, upper)
    return np.where(mutate, mutated, out)
