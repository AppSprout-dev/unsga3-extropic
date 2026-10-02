"""Classical continuous U-NSGA-III. NumPy only.

Generational population, Das–Dennis directions, non-dominated ranking,
perpendicular association, and the PymooCompatible mating rule (same
niche: better rank, then closer perpendicular distance; otherwise a coin).
Variation is simulated binary crossover and polynomial mutation.

This package is a fidelity column for ZDT1, ZDT2, and DTLZ2. It is not a
THRML program, not an Ising or Potts energy, and not called by
``WeightSweepLoop``. SBX and polynomial mutation stay in this package.
"""

from __future__ import annotations

from unsga3_extropic.classical.igd import pareto_front, score_front
from unsga3_extropic.classical.pm import polynomial_mutation
from unsga3_extropic.classical.sbx import sbx
from unsga3_extropic.classical.unsga3 import UnsGa3Result, run_unsga3

__all__ = [
    "UnsGa3Result",
    "pareto_front",
    "polynomial_mutation",
    "run_unsga3",
    "sbx",
    "score_front",
]
