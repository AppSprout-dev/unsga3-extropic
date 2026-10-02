"""Backend protocol: sample under a scalarized energy E_w."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class BackendResult:
    """Samples from one weight vector / anneal run.

    ``n_evals`` is the number of scored rows. ``n_invalid`` counts rows the
    backend refused to score. Ising, Potts, and Metropolis leave it at 0.
    The domain-wall sampler counts spin patterns that are not thermometers
    and omits them from ``decisions`` and ``objectives``.
    """

    decisions: np.ndarray  # (n_samples, n_vars)
    objectives: np.ndarray  # (n_samples, n_obj)
    energies: np.ndarray  # (n_samples,) scalarized E_w
    n_evals: int
    n_invalid: int = 0


@runtime_checkable
class SamplingBackend(Protocol):
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
        """Draw samples for scalarization weights ``w`` (annealed)."""
        ...
