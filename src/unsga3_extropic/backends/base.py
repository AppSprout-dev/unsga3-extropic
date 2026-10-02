"""Backend protocol: sample under a scalarized energy E_w."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class BackendResult:
    """Samples from one weight vector / anneal run."""

    decisions: np.ndarray  # (n_samples, n_vars)
    objectives: np.ndarray  # (n_samples, n_obj)
    energies: np.ndarray  # (n_samples,) scalarized E_w
    n_evals: int


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
