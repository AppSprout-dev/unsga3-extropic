"""Opt-in energy-based selection and plateau stopping (EBR spike).

Two small tools taken from the "one energy, three jobs" idea in
Energy-Based Reasoners (arXiv:2611.08977). See ``docs/ebr_note.md``.

* ``select_lowest_energy`` keeps the single lowest-``E_w`` row of a
  ``BackendResult``. ``LowestEnergySelection`` wraps any
  ``SamplingBackend`` with it, so each weight contributes one row.
* ``PlateauStop`` ends a sampling stage when the running best energy has
  not dropped by more than ``tol`` over the last ``window`` recorded
  samples. ``ExactEwMetropolisBackend(plateau=...)`` applies it to each
  beta stage.

The energy here is the objective-derived ``E_w = w · f(x)``. Nothing is
learned and nothing is descended by backprop. Neither tool is used unless
a caller asks for it: ``WeightSweepLoop`` and every backend keep their
default behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unsga3_extropic.backends.base import BackendResult, SamplingBackend


def lowest_energy_index(energies: np.ndarray) -> int:
    """Index of the lowest finite energy. Ties go to the earliest row.

    Raises ``ValueError`` for an empty vector or one with no finite entry.
    """
    e = np.asarray(energies, dtype=np.float64).reshape(-1)
    if e.size == 0:
        raise ValueError("no energies to select from")
    finite = np.isfinite(e)
    if not finite.any():
        raise ValueError("no finite energy to select")
    masked = np.where(finite, e, np.inf)
    return int(np.argmin(masked))


def select_lowest_energy(result: BackendResult) -> BackendResult:
    """Keep only the lowest-energy row. ``n_evals`` and ``n_invalid`` are kept.

    An empty result (for example, every domain-wall pattern was invalid) is
    returned unchanged.
    """
    if len(result.energies) == 0:
        return result
    i = lowest_energy_index(result.energies)
    return BackendResult(
        decisions=np.asarray(result.decisions)[i : i + 1].copy(),
        objectives=np.asarray(result.objectives)[i : i + 1].copy(),
        energies=np.asarray(result.energies, dtype=np.float64)[i : i + 1].copy(),
        n_evals=result.n_evals,
        n_invalid=result.n_invalid,
    )


@dataclass
class LowestEnergySelection:
    """``SamplingBackend`` wrapper: sample as usual, return the argmin row."""

    backend: SamplingBackend

    def sample_weight(
        self,
        w: np.ndarray,
        *,
        seed: int,
        betas: tuple[float, ...],
        n_warmup: int,
        n_samples: int,
        steps_per_sample: int,
    ) -> BackendResult:
        res = self.backend.sample_weight(
            w,
            seed=seed,
            betas=betas,
            n_warmup=n_warmup,
            n_samples=n_samples,
            steps_per_sample=steps_per_sample,
        )
        return select_lowest_energy(res)


@dataclass(frozen=True)
class PlateauStop:
    """Stop when the best energy improved by at most ``tol`` over ``window`` samples.

    ``best_trace[k]`` is the lowest energy seen after recorded sample ``k``
    (non-increasing). The rule fires at the first ``k >= window`` with
    ``best_trace[k - window] - best_trace[k] <= tol``. With ``window``
    samples recorded there is nothing to compare, so the earliest stop
    leaves ``window + 1`` samples.
    """

    window: int
    tol: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.window, bool) or int(self.window) != self.window or self.window < 1:
            raise ValueError(f"window must be an integer >= 1, got {self.window!r}")
        if not (np.isfinite(self.tol) and self.tol >= 0.0):
            raise ValueError(f"tol must be finite and >= 0, got {self.tol!r}")

    def should_stop(self, best_trace: list[float] | np.ndarray) -> bool:
        k = len(best_trace) - 1
        if k < self.window:
            return False
        return float(best_trace[k - self.window]) - float(best_trace[k]) <= self.tol

    def stop_index(self, energies: np.ndarray) -> int | None:
        """First sample index where the rule fires on this trace, else ``None``."""
        e = np.asarray(energies, dtype=np.float64).reshape(-1)
        best: list[float] = []
        for k, value in enumerate(e):
            best.append(float(value) if not best else min(best[-1], float(value)))
            if self.should_stop(best):
                return k
        return None
