"""Named anneal budgets for the in-repo codon and Potts problems.

``smoke`` is the CI demo. ``default`` is the demo without ``--smoke``.
``deep`` is the on-demand multi-seed budget run by ``benchmarks/run_deep.py``.
Deep uses the same problem instances as smoke (codon chain of 12 spins;
Potts chain of 6 sites and 3 categories) so the exact fronts match.
The larger default Potts chain (8 sites) stays on ``profile=default``.

None of these profiles search for a beta schedule. Deep is a THRML
simulation budget (plus a NumPy exact-``E_w`` comparator). It is not a
Z1 measurement and it does not call Thermalizers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from unsga3_extropic.loop import AnnealConfig
from unsga3_extropic.weights import simplex_weights

ProfileName = str
ProblemName = str

CODON_N_SPINS = 12
CODON_J = 1.0
CODON_FIELD_SEED = 0
CODON_GLOBAL_BIAS = 0.05
CODON_INSTANCE_LABEL = "n_spins=12, J=1, field_seed=0, global_bias=0.05"

# Deep Potts matches the CI smoke instance. The 8-site chain is default only.
POTTS_SMOKE_SITES = 6
POTTS_SMOKE_CATEGORIES = 3
POTTS_DEFAULT_SITES = 8
POTTS_DEFAULT_CATEGORIES = 3

# Torx PSWAP rate check. Not a search front. Smoke in torx_circuit is 20_000.
TORX_DEEP_N_SAMPLES = 100_000
TORX_DEEP_SEEDS = (0, 1, 2)
TORX_DEEP_P_SWAP = 0.3
# Binomial SE at p=0.3, n=100_000 is about 0.0014. 0.02 is a loose sanity bound.
TORX_DEEP_ABS_TOLERANCE = 0.02

_SMOKE_WEIGHTS = ((0.5, 0.5), (0.8, 0.2))
_SMOKE_WEIGHT_SOURCE = "fixed [[0.5, 0.5], [0.8, 0.2]]"
_DEFAULT_WEIGHTS = (
    (0.9, 0.1),
    (0.7, 0.3),
    (0.5, 0.5),
    (0.3, 0.7),
    (0.1, 0.9),
)
_DEFAULT_WEIGHT_SOURCE = "fixed [[0.9, 0.1], [0.7, 0.3], [0.5, 0.5], [0.3, 0.7], [0.1, 0.9]]"
_DEEP_N_WEIGHTS = 11
_DEEP_WEIGHT_SOURCE = f"simplex_weights({_DEEP_N_WEIGHTS}, 2)"


def _pairs(matrix: np.ndarray) -> tuple[tuple[float, float], ...]:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"weights must have shape (n, 2), got {arr.shape}")
    return tuple((float(row[0]), float(row[1])) for row in arr)


@dataclass(frozen=True)
class SearchSchedule:
    """One profile's weights, anneal, and replicate seeds."""

    profile: str
    betas: tuple[float, ...]
    n_warmup: int
    n_samples: int
    steps_per_sample: int
    base_seeds: tuple[int, ...]
    weights: tuple[tuple[float, float], ...]
    weight_source: str

    def n_weights(self) -> int:
        return len(self.weights)

    def weight_matrix(self) -> np.ndarray:
        return np.asarray(self.weights, dtype=np.float64)

    def anneal(self) -> AnnealConfig:
        return AnnealConfig(
            betas=self.betas,
            n_warmup=self.n_warmup,
            n_samples=self.n_samples,
            steps_per_sample=self.steps_per_sample,
        )

    def recorded_per_seed(self) -> int:
        """Objective rows one seed would record if every sample is kept."""
        return self.n_weights() * len(self.betas) * self.n_samples

    def recorded_total(self) -> int:
        return self.recorded_per_seed() * len(self.base_seeds)


def _schedule(
    profile: str,
    *,
    betas: tuple[float, ...],
    n_warmup: int,
    n_samples: int,
    steps_per_sample: int,
    base_seeds: tuple[int, ...],
    weights: tuple[tuple[float, float], ...],
    weight_source: str,
) -> SearchSchedule:
    if len(base_seeds) < 1:
        raise ValueError("base_seeds must be non-empty")
    if len(set(base_seeds)) != len(base_seeds):
        raise ValueError(f"duplicate base seeds: {base_seeds}")
    if any(seed < 0 for seed in base_seeds):
        raise ValueError("base seeds must be >= 0")
    return SearchSchedule(
        profile=profile,
        betas=betas,
        n_warmup=n_warmup,
        n_samples=n_samples,
        steps_per_sample=steps_per_sample,
        base_seeds=base_seeds,
        weights=weights,
        weight_source=weight_source,
    )


CODON_SMOKE = _schedule(
    "smoke",
    betas=(1.0, 4.0),
    n_warmup=4,
    n_samples=6,
    steps_per_sample=1,
    base_seeds=(7,),
    weights=_SMOKE_WEIGHTS,
    weight_source=_SMOKE_WEIGHT_SOURCE,
)
CODON_DEFAULT = _schedule(
    "default",
    betas=(0.5, 1.0, 2.0, 4.0),
    n_warmup=20,
    n_samples=24,
    steps_per_sample=2,
    base_seeds=(7,),
    weights=_DEFAULT_WEIGHTS,
    weight_source=_DEFAULT_WEIGHT_SOURCE,
)
POTTS_SMOKE = _schedule(
    "smoke",
    betas=(1.0, 4.0),
    n_warmup=2,
    n_samples=4,
    steps_per_sample=1,
    base_seeds=(7,),
    weights=_SMOKE_WEIGHTS,
    weight_source=_SMOKE_WEIGHT_SOURCE,
)
POTTS_DEFAULT = _schedule(
    "default",
    betas=(0.5, 1.0, 2.0, 4.0),
    n_warmup=8,
    n_samples=8,
    steps_per_sample=1,
    base_seeds=(7,),
    weights=_DEFAULT_WEIGHTS,
    weight_source=_DEFAULT_WEIGHT_SOURCE,
)
# Smoke and default domain-wall demos use base seed 11. Deep shares DEEP_SCHEDULE.
DOMAIN_WALL_SMOKE = replace(POTTS_SMOKE, base_seeds=(11,))
DOMAIN_WALL_DEFAULT = replace(POTTS_DEFAULT, base_seeds=(11,))
DEEP_SCHEDULE = _schedule(
    "deep",
    betas=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
    n_warmup=24,
    n_samples=24,
    steps_per_sample=2,
    base_seeds=(7, 11, 19),
    weights=_pairs(simplex_weights(_DEEP_N_WEIGHTS, 2)),
    weight_source=_DEEP_WEIGHT_SOURCE,
)


def _unhandled(kind: str, value: object) -> None:
    """Fail when a match over a closed set gains a variant."""
    raise AssertionError(f"unhandled {kind}: {value!r}")


def resolve_profile(*, smoke: bool, profile: str | None) -> str:
    """Map the demo ``--smoke`` flag and an optional profile name."""
    if profile is None:
        chosen = "smoke" if smoke else "default"
    else:
        chosen = profile
    if smoke and chosen != "smoke":
        raise ValueError("smoke=True cannot be combined with a non-smoke profile")
    match chosen:
        case "smoke":
            return "smoke"
        case "default":
            return "default"
        case "deep":
            return "deep"
        case _ as unreachable:
            _unhandled("profile", unreachable)
            raise AssertionError(unreachable)


def schedule_for(problem: str, profile: str) -> SearchSchedule:
    """Anneal budget for ``codon``, ``potts``, or ``domain_wall``."""
    match problem, profile:
        case "codon", "smoke":
            return CODON_SMOKE
        case "codon", "default":
            return CODON_DEFAULT
        case "codon", "deep":
            return DEEP_SCHEDULE
        case "potts", "smoke":
            return POTTS_SMOKE
        case "potts", "default":
            return POTTS_DEFAULT
        case "potts", "deep":
            return DEEP_SCHEDULE
        case "domain_wall", "smoke":
            return DOMAIN_WALL_SMOKE
        case "domain_wall", "default":
            return DOMAIN_WALL_DEFAULT
        case "domain_wall", "deep":
            return DEEP_SCHEDULE
        case _ as unreachable:
            _unhandled("problem/profile", unreachable)
            raise AssertionError(unreachable)


def potts_instance(profile: str) -> tuple[int, int]:
    """``(n_sites, n_categories)`` for a Potts or domain-wall profile."""
    match profile:
        case "smoke" | "deep":
            return (POTTS_SMOKE_SITES, POTTS_SMOKE_CATEGORIES)
        case "default":
            return (POTTS_DEFAULT_SITES, POTTS_DEFAULT_CATEGORIES)
        case _ as unreachable:
            _unhandled("profile", unreachable)
            raise AssertionError(unreachable)


def potts_instance_label(profile: str) -> str:
    n_sites, n_categories = potts_instance(profile)
    return f"n_sites={n_sites}, K={n_categories}"


def codon_problem_kwargs() -> dict[str, float | int]:
    """Constructor kwargs for the in-repo codon chain. Same at every profile."""
    return {
        "n_spins": CODON_N_SPINS,
        "J": CODON_J,
        "seed": CODON_FIELD_SEED,
        "global_bias": CODON_GLOBAL_BIAS,
    }


def profile_clause(schedule: SearchSchedule, base_seed: int) -> str:
    """Stable note prefix: profile, replicate, weight source, schedule product."""
    if base_seed not in schedule.base_seeds:
        raise ValueError(
            f"base_seed {base_seed} is not in {schedule.profile} seeds {schedule.base_seeds}"
        )
    index = schedule.base_seeds.index(base_seed) + 1
    return (
        f"profile={schedule.profile}. "
        f"base_seed={base_seed}. "
        f"replicate={index}/{len(schedule.base_seeds)}. "
        f"weights={schedule.weight_source}. "
        f"schedule_product={schedule.recorded_per_seed()}."
    )


def require_base_seed(schedule: SearchSchedule, base_seed: int | None) -> int:
    """Use the only seed, or require an explicit seed when there are several."""
    if base_seed is None:
        if len(schedule.base_seeds) != 1:
            raise ValueError(
                f"profile {schedule.profile} has seeds {list(schedule.base_seeds)}; "
                "pass base_seed"
            )
        return schedule.base_seeds[0]
    if base_seed not in schedule.base_seeds:
        raise ValueError(
            f"base_seed {base_seed} is not in {schedule.profile} seeds {schedule.base_seeds}"
        )
    return base_seed
