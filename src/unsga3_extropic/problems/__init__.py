"""Demo / benchmark problems."""

from unsga3_extropic.problems.codon_ising import CodonIsingProblem
from unsga3_extropic.problems.continuous import dtlz2, oracle_specs, zdt1, zdt2
from unsga3_extropic.problems.potts_chain import PottsChainProblem

__all__ = [
    "CodonIsingProblem",
    "PottsChainProblem",
    "dtlz2",
    "oracle_specs",
    "zdt1",
    "zdt2",
]
