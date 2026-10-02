"""Weight-sweep multiobjective search on THRML Ising and Potts substrates.

Weight-sweep scalarizations, sample with THRML block Gibbs or exact E_w
Metropolis, keep an offline non-dominated archive, and survive the closer
occupant of each reference direction. The Potts chain also has a domain-wall
Ising image sampled in THRML; that image is not a device run.

An optional Torx circuit lives in ``unsga3_extropic.torx_circuit``. This
module does not import it. It is not part of the search loop.

See README for which path is THRML-native, and ``docs/fidelity_hooks.md``
for comparing an exact-E_w archive with an external classical reference.
"""

from __future__ import annotations

from unsga3_extropic.archive import Archive, NicheResult, nondominated_mask, unique_rows
from unsga3_extropic.loop import AnnealConfig, LoopResult, WeightSweepLoop
from unsga3_extropic.weights import simplex_weights

__version__ = "0.5.0"

__all__ = [
    "Archive",
    "AnnealConfig",
    "LoopResult",
    "NicheResult",
    "WeightSweepLoop",
    "nondominated_mask",
    "simplex_weights",
    "unique_rows",
    "__version__",
]
