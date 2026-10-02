"""U-NSGA-III on Extropic substrate.

Center: many-objective U-NSGA-III *intent* (vector fitness, preference diversity,
iterative search) mapped onto Extropic-compatible energy sampling:

  weight-sweep scalarizations → sample (THRML or exact Ew MH) → offline ND archive.

Bend / C# U-NSGA-III are fidelity references only — see docs/fidelity_hooks.md
and the ZDT1 toys under ``../toys/``.
"""

from __future__ import annotations

from unsga3_extropic.archive import Archive, nondominated_mask, unique_rows
from unsga3_extropic.loop import AnnealConfig, LoopResult, WeightSweepLoop
from unsga3_extropic.weights import simplex_weights

__version__ = "0.1.0"

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
