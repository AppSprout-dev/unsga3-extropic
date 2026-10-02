"""Pareto samples and the yardstick IGD for ZDT1, ZDT2, and DTLZ2.

IGD is ``fidelity.inverted_generational_distance``: the mean Euclidean
distance from each reference point to the nearest obtained point. ZDT
references are the analytic curves at 500 points. DTLZ2 is the Das–Dennis
set at the run's partitions, each row scaled to unit L2 length (91 points
when ``partitions = 12`` and ``M = 3``). pymoo is not imported.
"""

from __future__ import annotations

from typing import Never

import numpy as np

from unsga3_extropic.fidelity import inverted_generational_distance
from unsga3_extropic.problems.continuous import OracleName
from unsga3_extropic.weights import das_dennis_directions

PF_POINTS = 500
PARTITIONS = 12


def _unreachable(name: Never) -> Never:
    raise AssertionError(f"unhandled oracle problem {name!r}")


def zdt1_pf(n_points: int = PF_POINTS) -> np.ndarray:
    """``f2 = 1 - sqrt(f1)`` on ``f1`` from 0 to 1, inclusive."""
    count = max(int(n_points), 2)
    f1 = np.linspace(0.0, 1.0, count)
    return np.column_stack([f1, 1.0 - np.sqrt(f1)])


def zdt2_pf(n_points: int = PF_POINTS) -> np.ndarray:
    """``f2 = 1 - f1^2`` on ``f1`` from 0 to 1, inclusive."""
    count = max(int(n_points), 2)
    f1 = np.linspace(0.0, 1.0, count)
    return np.column_stack([f1, 1.0 - f1 * f1])


def dtlz2_pf(n_obj: int = 3, partitions: int = PARTITIONS) -> np.ndarray:
    """Das–Dennis directions L2-normalized onto the unit sphere."""
    weights = das_dennis_directions(n_obj, partitions)
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    return weights / np.maximum(norms, 1e-16)


def pareto_front(
    problem: OracleName,
    *,
    pf_points: int = PF_POINTS,
    partitions: int = PARTITIONS,
) -> tuple[np.ndarray, str]:
    """Reference set and the ``pf_source`` label for one yardstick problem."""
    match problem:
        case "zdt1":
            front = zdt1_pf(pf_points)
            return front, f"analytic-zdt1 n={front.shape[0]}"
        case "zdt2":
            front = zdt2_pf(pf_points)
            return front, f"analytic-zdt2 n={front.shape[0]}"
        case "dtlz2":
            front = dtlz2_pf(3, partitions)
            return front, "analytic-das-dennis-l2"
        case _ as other:
            _unreachable(other)


def score_front(
    problem: OracleName,
    front: np.ndarray,
    *,
    pf_points: int = PF_POINTS,
    partitions: int = PARTITIONS,
) -> tuple[float, int, str]:
    """``(igd, pf_rows, pf_source)`` for a minimization front."""
    reference, source = pareto_front(problem, pf_points=pf_points, partitions=partitions)
    igd = inverted_generational_distance(front, reference)
    return igd, int(reference.shape[0]), source
