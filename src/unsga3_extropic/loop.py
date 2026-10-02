"""Outer loop: weight-sweep sampling, then archive niching.

Classical U-NSGA-III: evaluate → rank (ND layers) → niche (reference dirs) →
select → vary.

Mapping used here:
  niches ≈ the same weight vectors that scalarize each anneal
  variation/search ≈ annealed sampling under E_w (``sample_weight`` only)
  ranking ≈ offline non-dominated archive across all weight runs
  survival ≈ closer occupant of each reference direction, on that archive

There is no crossover and no mutation. Reference directions passed by the
caller are the ones used for association.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unsga3_extropic.archive import Archive, NicheResult, hypervolume_2d
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

    def niche_front(self, *, quota: int = 1, decimals: int = 6) -> NicheResult:
        """Closer occupants of ``self.weights``. The non-dominated front stays put."""
        return self.archive.niche_survival(
            self.weights, quota=quota, decimals=decimals
        )

    def summary(self) -> dict:
        _x, f = self.nondominated_front()
        niche = self.niche_front()
        out: dict = {
            "n_weights": int(len(self.weights)),
            "total_evals": int(self.total_evals),
            "archive_raw": int(len(self.archive.objectives)),
            "archive_nd_unique": int(len(f)),
            "archive_niche": int(len(niche.objectives)),
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
    """Sample each reference direction, then niche the pooled archive.

    Candidates come only from ``SamplingBackend.sample_weight``. When
    ``weights`` is set, those rows are both the scalarizations and the
    association directions. Otherwise ``simplex_weights`` builds them once.
    """

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
