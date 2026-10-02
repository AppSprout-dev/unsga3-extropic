"""Weight-sweep multiobjective search on THRML Ising and Potts substrates.

Weight-sweep scalarizations, sample with THRML block Gibbs or exact E_w
Metropolis, then keep an offline non-dominated archive.

See README for which path is THRML-native, and ``docs/fidelity_hooks.md``
for comparing an exact-E_w archive with an external classical reference.
"""

from __future__ import annotations

from unsga3_extropic.archive import Archive, nondominated_mask, unique_rows
from unsga3_extropic.loop import AnnealConfig, LoopResult, WeightSweepLoop
from unsga3_extropic.weights import simplex_weights

__version__ = "0.2.0"

__all__ = [
    "Archive",
    "AnnealConfig",
    "LoopResult",
    "WeightSweepLoop",
    "nondominated_mask",
    "simplex_weights",
    "unique_rows",
    "__version__",
]
