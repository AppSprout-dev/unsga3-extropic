"""Exact E_w Metropolis–Hastings backend for general (non-Ising) objectives.

Use when the scalarized energy is not pairwise-Ising expressible (e.g. ZDT1
with sqrt terms). See ``docs/fidelity_hooks.md`` and
``/workspace/extropic-first-job/toys/zdt1_true_ew/`` for the fidelity toy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from unsga3_extropic.backends.base import BackendResult

ObjectiveFn = Callable[[np.ndarray], np.ndarray]  # bits (..., n) -> (..., M)
ProposeFn = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def _default_bit_flip(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = x.copy()
    i = int(rng.integers(0, len(y)))
    y[i] = 1 - y[i]
    return y


@dataclass
class ExactEwMetropolisBackend:
    """Annealed MH on bitstrings with exact E_w = w · f(x)."""

    n_bits: int
    objective_fn: ObjectiveFn
    propose: ProposeFn | None = None
    init_fn: Callable[[np.random.Generator], np.ndarray] | None = None

    def sample_weight(
        self,
        w: np.ndarray,
        *,
        seed: int,
        betas: tuple[float, ...] = (0.2, 0.5, 1.0, 2.0, 4.0, 8.0),
        n_warmup: int = 40,
        n_samples: int = 48,
        steps_per_sample: int = 2,
    ) -> BackendResult:
        w = np.asarray(w, dtype=np.float64)
        rng = np.random.default_rng(seed)
        propose = self.propose or _default_bit_flip
        if self.init_fn is not None:
            x = self.init_fn(rng)
        else:
            x = rng.integers(0, 2, size=self.n_bits).astype(np.float64)

        def energy(bits: np.ndarray) -> float:
            f = np.asarray(self.objective_fn(bits.reshape(1, -1)), dtype=np.float64)[0]
            return float(np.dot(w, f))

        e = energy(x)
        collected_x: list[np.ndarray] = []
        collected_f: list[np.ndarray] = []
        collected_e: list[float] = []
        n_evals = 1  # initial energy

        for beta in betas:
            # warmup
            for _ in range(n_warmup):
                y = propose(x, rng)
                ey = energy(y)
                n_evals += 1
                dE = ey - e
                if dE <= 0.0 or rng.random() < np.exp(-beta * dE):
                    x, e = y, ey
            # samples
            for _ in range(n_samples):
                for _ in range(steps_per_sample):
                    y = propose(x, rng)
                    ey = energy(y)
                    n_evals += 1
                    dE = ey - e
                    if dE <= 0.0 or rng.random() < np.exp(-beta * dE):
                        x, e = y, ey
                f = np.asarray(self.objective_fn(x.reshape(1, -1)), dtype=np.float64)[0]
                collected_x.append(x.copy())
                collected_f.append(f)
                collected_e.append(e)

        return BackendResult(
            decisions=np.stack(collected_x, axis=0),
            objectives=np.stack(collected_f, axis=0),
            energies=np.asarray(collected_e, dtype=np.float64),
            n_evals=n_evals,
        )
