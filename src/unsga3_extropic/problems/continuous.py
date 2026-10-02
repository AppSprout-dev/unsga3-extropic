"""Continuous ZDT1, ZDT2, and DTLZ2 for the NumPy exact-E_w yardstick.

These objectives are the same boxes Bend and C# use in
``unsga3-bend`` ``docs/ORACLE-MULTISEED.md``:

- ZDT1, ``n = 30``, each coordinate in ``[0, 1]``
- ZDT2, ``n = 30``, each coordinate in ``[0, 1]``
- DTLZ2, ``M = 3``, ``k = 10``, so ``n = 12``, each coordinate in ``[0, 1]``

THRML Ising cannot express them. ``g`` on ZDT and the spherical map on
DTLZ2 are nonlinear functions of a real vector, not a pairwise spin
energy. ``ExactEwContinuousBackend`` evaluates ``E_w = w · f(x)`` in
NumPy. Nothing here builds an ``IsingEBM`` or fits a surrogate.

The anneal spends the Bend population-times-generations objective budget
when that product splits evenly across the Das–Dennis directions at
``partitions = 12``. DTLZ2 has 91 directions and a yardstick of
``92 * 150 = 13800`` calls, which is not divisible by 91, so the ladder
spends ``13741`` calls (59 fewer). Betas are ``(1, 4, 8)``, the upper end
of the in-repo smoke ladder. About ten percent of each beta's proposals
are warmup. This schedule is not a search over betas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Never

import numpy as np

OracleName = Literal["zdt1", "zdt2", "dtlz2"]

ORACLE_BETAS: tuple[float, ...] = (1.0, 4.0, 8.0)
ORACLE_SEEDS: tuple[int, ...] = tuple(range(1, 16))
PARTITIONS = 12


def _unreachable(name: Never) -> Never:
    raise AssertionError(f"unhandled oracle problem {name!r}")


def _as_batch(x: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1), True
    if arr.ndim != 2:
        raise ValueError(f"decision array must be 1-D or 2-D, got {arr.shape}")
    return arr, False


def zdt1(x: np.ndarray) -> np.ndarray:
    """ZDT1. ``f1 = x0``, ``g = 1 + 9 mean(x1:)``, ``f2 = g (1 - sqrt(f1/g))``."""
    rows, single = _as_batch(x)
    if rows.shape[1] < 2:
        raise ValueError("ZDT1 needs at least 2 variables")
    f1 = rows[:, 0]
    g = 1.0 + 9.0 * np.sum(rows[:, 1:], axis=1) / (rows.shape[1] - 1)
    f2 = g * (1.0 - np.sqrt(f1 / g))
    out = np.column_stack([f1, f2])
    return out[0] if single else out


def zdt2(x: np.ndarray) -> np.ndarray:
    """ZDT2. Same ``g`` as ZDT1, with ``f2 = g (1 - (f1/g)^2)``."""
    rows, single = _as_batch(x)
    if rows.shape[1] < 2:
        raise ValueError("ZDT2 needs at least 2 variables")
    f1 = rows[:, 0]
    g = 1.0 + 9.0 * np.sum(rows[:, 1:], axis=1) / (rows.shape[1] - 1)
    ratio = f1 / g
    f2 = g * (1.0 - ratio * ratio)
    out = np.column_stack([f1, f2])
    return out[0] if single else out


def dtlz2(x: np.ndarray, *, n_obj: int = 3) -> np.ndarray:
    """DTLZ2. Last ``n - M + 1`` coordinates form ``g``; the front is the sphere.

    The oracle uses ``M = 3`` and ``k = 10`` (``n = 12``). ``g = 0`` when
    those distance coordinates equal ``0.5``.
    """
    if n_obj < 2:
        raise ValueError("n_obj must be >= 2")
    rows, single = _as_batch(x)
    if rows.shape[1] < n_obj:
        raise ValueError(f"DTLZ2 needs at least {n_obj} variables")
    g = np.sum((rows[:, n_obj - 1 :] - 0.5) ** 2, axis=1)
    f = np.empty((rows.shape[0], n_obj), dtype=np.float64)
    for i in range(n_obj):
        col = np.ones(rows.shape[0], dtype=np.float64)
        for j in range(n_obj - i - 1):
            col *= np.cos(rows[:, j] * np.pi / 2.0)
        if i > 0:
            col *= np.sin(rows[:, n_obj - i - 1] * np.pi / 2.0)
        f[:, i] = (1.0 + g) * col
    return f[0] if single else f


def evaluate(name: OracleName, x: np.ndarray) -> np.ndarray:
    """Evaluate one oracle problem. ``x`` is a single decision vector."""
    match name:
        case "zdt1":
            return zdt1(x)
        case "zdt2":
            return zdt2(x)
        case "dtlz2":
            return dtlz2(x, n_obj=3)
        case _ as other:
            _unreachable(other)


@dataclass(frozen=True)
class ContinuousOracleSpec:
    """One continuous problem at the Bend ORACLE-MULTISEED population budget."""

    name: OracleName
    n_var: int
    n_obj: int
    partitions: int
    pop: int
    gens: int
    n_weights: int
    n_warmup: int
    n_samples: int
    betas: tuple[float, ...] = ORACLE_BETAS
    steps_per_sample: int = 1
    lower: float = 0.0
    upper: float = 1.0
    step_scale: float = 0.1

    def yardstick_evals(self) -> int:
        """Bend / C# objective calls under ``n_eval = pop * gens``."""
        return self.pop * self.gens

    def objective_evals_per_weight(self) -> int:
        """Initial point plus one objective call per proposal, one weight."""
        return 1 + len(self.betas) * (
            self.n_warmup + self.n_samples * self.steps_per_sample
        )

    def objective_evals(self) -> int:
        return self.n_weights * self.objective_evals_per_weight()

    def recorded_samples(self) -> int:
        """Rows that enter the archive. Warmup proposals are not recorded."""
        return self.n_weights * len(self.betas) * self.n_samples

    def objective_fn(self):
        spec = self

        def fn(point: np.ndarray) -> np.ndarray:
            vector = np.asarray(point, dtype=np.float64).reshape(-1)
            if vector.shape != (spec.n_var,):
                raise ValueError(
                    f"{spec.name} expects {spec.n_var} variables, got {vector.shape[0]}"
                )
            return evaluate(spec.name, vector)

        return fn


def oracle_specs() -> dict[OracleName, ContinuousOracleSpec]:
    """ZDT1 52×100, ZDT2 52×250, DTLZ2 92×150, partitions 12.

    Weight counts are the Das–Dennis sizes at those partitions:
    ``M = 2`` gives 13 directions, ``M = 3`` gives 91.
    """
    return {
        "zdt1": ContinuousOracleSpec(
            name="zdt1",
            n_var=30,
            n_obj=2,
            partitions=PARTITIONS,
            pop=52,
            gens=100,
            n_weights=13,
            n_warmup=13,
            n_samples=120,
        ),
        "zdt2": ContinuousOracleSpec(
            name="zdt2",
            n_var=30,
            n_obj=2,
            partitions=PARTITIONS,
            pop=52,
            gens=250,
            n_weights=13,
            n_warmup=33,
            n_samples=300,
        ),
        "dtlz2": ContinuousOracleSpec(
            name="dtlz2",
            n_var=12,
            n_obj=3,
            partitions=PARTITIONS,
            pop=92,
            gens=150,
            n_weights=91,
            n_warmup=5,
            n_samples=45,
        ),
    }
