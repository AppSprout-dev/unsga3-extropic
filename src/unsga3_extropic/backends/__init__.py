"""Sampling backends for scalarized energies."""

from unsga3_extropic.backends.base import BackendResult, SamplingBackend
from unsga3_extropic.backends.ew_metropolis import ExactEwMetropolisBackend
from unsga3_extropic.backends.thrml_domain_wall import ThrmlDomainWallBackend
from unsga3_extropic.backends.thrml_ising import ThrmlIsingBackend
from unsga3_extropic.backends.thrml_potts import ThrmlPottsBackend

__all__ = [
    "BackendResult",
    "ExactEwMetropolisBackend",
    "SamplingBackend",
    "ThrmlDomainWallBackend",
    "ThrmlIsingBackend",
    "ThrmlPottsBackend",
]
