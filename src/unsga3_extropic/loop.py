"""Outer loop: weight-sweep + archive non-dominated (U-NSGA-III-shaped).

Classical U-NSGA-III: evaluate → rank (ND layers) → niche (reference dirs) →
select → vary.

Thermo mapping used here:
  niches ≈ distinct weight / reference directions
  variation/search ≈ annealed sampling under E_w
  ranking ≈ offline non-dominated archive across all weight runs
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unsga3_extropic.archive import Archive, hypervolume_2d
from unsga3_extropic.backends.base import BackendResult, SamplingBackend
from unsga3_extropic.weights import simplex_weights


@dataclass
class AnnealConfig:
    betas: tuple[float, ...] = (0.2, 0.5, 1.0, 2.0, 4.0, 8.0)
    n_warmup: int = 40
    n_samples: int = 48
    steps_per_sample: int = 2


@dataclass
class LoopResult:
    archive: Archive
    weights: np.ndarray
    per_weight: list[BackendResult] = field(default_factory=list)
    total_evals: int = 0

    def nondominated_front(self, *, decimals: int = 6):
        return self.archive.nondominated(decimals=decimals)

    def summary(self) -> dict:
        x, f = self.nondominated_front()
        out: dict = {
            "n_weights": int(len(self.weights)),
            "total_evals": int(self.total_evals),
            "archive_raw": int(len(self.archive.objectives)),
            "archive_nd_unique": int(len(f)),
        }
        if f.ndim == 2 and f.shape[1] == 2 and len(f) > 0:
            # normalize ref from data extent
            ref = (float(f[:, 0].max() + 0.1 * abs(f[:, 0].max() or 1)),
                   float(f[:, 1].max() + 0.1 * abs(f[:, 1].max() or 1)))
            out["hv_2d_data_ref"] = hypervolume_2d(f, ref=ref)
            # list so LoopResult.summary() is json.dumps-safe
            out["hv_ref"] = [ref[0], ref[1]]
        return out


@dataclass
class WeightSweepLoop:
    """U-NSGA-III-shaped diversity via weight sweep + ND archive."""

    backend: SamplingBackend
    n_weights: int = 5
    n_obj: int = 2
    anneal: AnnealConfig = field(default_factory=AnnealConfig)
    seed: int = 7
    weights: np.ndarray | None = None

    def run(self) -> LoopResult:
        wmat = (
            np.asarray(self.weights, dtype=np.float64)
            if self.weights is not None
            else simplex_weights(self.n_weights, self.n_obj)
        )
        archive = Archive()
        per: list[BackendResult] = []
        total = 0
        for i, w in enumerate(wmat):
            res = self.backend.sample_weight(
                w,
                seed=self.seed + 1009 * i,
                betas=self.anneal.betas,
                n_warmup=self.anneal.n_warmup,
                n_samples=self.anneal.n_samples,
                steps_per_sample=self.anneal.steps_per_sample,
            )
            archive.add(res.decisions, res.objectives, weight_id=i)
            per.append(res)
            total += res.n_evals
        return LoopResult(
            archive=archive, weights=wmat, per_weight=per, total_evals=total
        )
