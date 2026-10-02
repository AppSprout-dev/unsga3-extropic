"""Exact E_w Metropolis–Hastings backends for general objectives.

``ExactEwMetropolisBackend`` is bit-flip Metropolis on ``{0,1}^n``.
``ExactEwContinuousBackend`` is truncated-Gaussian Metropolis on a box.
Both use exact ``E_w = w · f(x)``. NumPy only: neither builds a THRML
program. Use them when the scalarized energy is not pairwise Ising or a
Potts factor. See ``docs/fidelity_hooks.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

# Fixed proposal width for the box kernel. Ten percent of the unit interval.
# Not chosen by searching IGD.
_DEFAULT_STEP_SCALE = 0.1

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


def _std_norm_cdf(t: float) -> float:
    return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))


def _truncation_mass(center: float, sigma: float, lower: float, upper: float) -> float:
    """Gaussian probability of landing inside ``[lower, upper]``."""
    mass = _std_norm_cdf((upper - center) / sigma) - _std_norm_cdf((lower - center) / sigma)
    return mass if mass > 1e-300 else 1e-300


def _truncated_normal_coordinate(
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    sigma: float,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, float, float]:
    """Perturb one coordinate by a normal truncated to the box.

    Returns the candidate, ``log Z(old)``, and ``log Z(new)`` for the
    Hastings correction. ``Z`` is the Gaussian mass inside the box.
    """
    y = np.array(x, dtype=np.float64, copy=True)
    index = int(rng.integers(0, y.shape[0]))
    old = float(y[index])
    while True:
        candidate = old + float(rng.normal(0.0, sigma))
        if lower <= candidate <= upper:
            break
    y[index] = candidate
    log_z_old = math.log(_truncation_mass(old, sigma, lower, upper))
    log_z_new = math.log(_truncation_mass(candidate, sigma, lower, upper))
    return y, log_z_old, log_z_new


@dataclass
class ExactEwContinuousBackend:
    """Annealed Metropolis on a box with exact E_w = w · f(x).

    NumPy only. This is not a THRML program and not an Ising or Potts
    energy. Continuous ZDT and DTLZ objectives are nonlinear functions of
    a real vector; they are not factorized into ``IsingEBM`` couplings, and
    this backend does not fit a surrogate to pretend that they are.

    Each proposal adds ``Normal(0, step_scale)`` to one coordinate and
    rejects draws that leave the box, which is a truncated normal. The
    Hastings ratio is ``Z(new) / Z(old)``, where ``Z`` is the Gaussian
    mass still inside the box. ``step_scale`` defaults to 0.1 on the unit
    interval. ``n_evals`` is the number of objective calls: the initial
    point plus one call per in-box proposal. The vector stored with a
    sample is that call's result, not a second evaluation.
    """

    n_var: int
    objective_fn: ObjectiveFn
    lower: float = 0.0
    upper: float = 1.0
    step_scale: float = _DEFAULT_STEP_SCALE

    def __post_init__(self) -> None:
        if self.n_var < 1:
            raise ValueError("n_var must be >= 1")
        if not self.upper > self.lower:
            raise ValueError("upper must be greater than lower")
        if not self.step_scale > 0.0:
            raise ValueError("step_scale must be > 0")

    def sample_weight(
        self,
        w: np.ndarray,
        *,
        seed: int,
        betas: tuple[float, ...] = (1.0, 4.0, 8.0),
        n_warmup: int = 0,
        n_samples: int = 1,
        steps_per_sample: int = 1,
    ) -> BackendResult:
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        if w.shape != (w.size,) or w.size < 1:
            raise ValueError(f"w must be a non-empty 1-D weight vector, got {w.shape}")
        rng = np.random.default_rng(seed)
        x = rng.uniform(self.lower, self.upper, size=self.n_var)

        def energy_of(point: np.ndarray) -> tuple[float, np.ndarray]:
            raw = np.asarray(self.objective_fn(point), dtype=np.float64).reshape(-1)
            if raw.shape != w.shape:
                raise ValueError(
                    f"objective length {raw.shape[0]} does not match w length {w.shape[0]}"
                )
            return float(np.dot(w, raw)), raw

        e, f = energy_of(x)
        collected_x: list[np.ndarray] = []
        collected_f: list[np.ndarray] = []
        collected_e: list[float] = []
        n_evals = 1

        def propose_and_maybe_accept(beta: float) -> None:
            nonlocal x, e, f, n_evals
            y, log_z_old, log_z_new = _truncated_normal_coordinate(
                x,
                rng,
                sigma=self.step_scale,
                lower=self.lower,
                upper=self.upper,
            )
            ey, fy = energy_of(y)
            n_evals += 1
            # log(accept) = -beta dE + log Z(new) - log Z(old)
            log_alpha = -beta * (ey - e) + log_z_new - log_z_old
            if log_alpha >= 0.0 or math.log(float(rng.random())) < log_alpha:
                x, e, f = y, ey, fy

        for beta in betas:
            for _ in range(n_warmup):
                propose_and_maybe_accept(float(beta))
            for _ in range(n_samples):
                for _ in range(steps_per_sample):
                    propose_and_maybe_accept(float(beta))
                collected_x.append(np.array(x, dtype=np.float64, copy=True))
                collected_f.append(np.array(f, dtype=np.float64, copy=True))
                collected_e.append(e)

        return BackendResult(
            decisions=np.stack(collected_x, axis=0),
            objectives=np.stack(collected_f, axis=0),
            energies=np.asarray(collected_e, dtype=np.float64),
            n_evals=n_evals,
        )
