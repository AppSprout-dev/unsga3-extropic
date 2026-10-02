"""Domain-wall Ising sampling of the in-repo Potts chain.

The Potts weights from ``build_potts`` are compiled with
``compile_domain_wall`` (THRML example 03,
https://docs.thrml.ai/en/latest/03_codon_optimization.html). Sampling then
depends on the spin graph:

- 2-colored, and at least one edge: ``ThrmlIsingBackend``. Its free blocks
  are the even/odd checkerboard, so the spins are relabeled to that layout.
- Otherwise, including the default ``K >= 3`` chain (the between-site
  couplings plus the thermometer edges are not 2-colored): a
  ``FactorSamplingProgram`` of ``SpinEBMFactor``s. Free blocks are the
  example 03 4-coloring, ``(site parity, spin-index parity)``. One
  ``SpinGibbsConditional`` per block. ``SamplingSchedule`` has no beta;
  beta rescales the Ising weights, and the free-block state is carried
  across temperatures.

Decoded decisions are categorical states. A spin row that is not a
thermometer is counted in ``BackendResult.n_invalid`` and is not given a
Potts objective. This is a THRML simulation of the p-bit image. It does not
call a device and it does not vendor ``codon_opt``.

JAX and THRML are imported inside the colored sampler. They are optional at
import time, matching ``ThrmlPottsBackend``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

import numpy as np

from unsga3_extropic.backends.base import BackendResult
from unsga3_extropic.backends.thrml_ising import ThrmlIsingBackend
from unsga3_extropic.domain_wall import (
    DomainWallImage,
    checkerboard_permutation,
    compile_domain_wall,
)

# build_potts(w) -> (unary W[n, K], edges, pairwise W[n_edges, K, K])
PottsBuilder = Callable[
    [np.ndarray], tuple[np.ndarray, list[tuple[int, int]], np.ndarray]
]
# objective_fn(states[..., n_sites]) -> (..., n_obj) on valid categorical states.
ObjectiveFn = Callable[[np.ndarray], np.ndarray]


def _load_thrml() -> SimpleNamespace:
    """Import JAX and the spin-factor sampler. Optional at package import."""
    import jax
    import jax.numpy as jnp
    from thrml import (
        Block,
        BlockGibbsSpec,
        FactorSamplingProgram,
        SamplingSchedule,
        SpinNode,
        sample_states,
    )
    from thrml.models import SpinEBMFactor, SpinGibbsConditional

    return SimpleNamespace(
        jax=jax,
        jnp=jnp,
        Block=Block,
        BlockGibbsSpec=BlockGibbsSpec,
        FactorSamplingProgram=FactorSamplingProgram,
        SamplingSchedule=SamplingSchedule,
        SpinNode=SpinNode,
        sample_states=sample_states,
        SpinEBMFactor=SpinEBMFactor,
        SpinGibbsConditional=SpinGibbsConditional,
    )


def _validate_schedule(
    betas: tuple[float, ...],
    n_warmup: int,
    n_samples: int,
    steps_per_sample: int,
    seed: int,
    constraint_strength: float,
) -> None:
    if len(tuple(betas)) < 1:
        raise ValueError("betas must be non-empty")
    if n_warmup < 0 or n_samples < 1 or steps_per_sample < 1:
        raise ValueError(
            "require n_warmup >= 0, n_samples >= 1, steps_per_sample >= 1"
        )
    if int(seed) < 0:
        raise ValueError("seed must be >= 0")
    if float(constraint_strength) < 0.0:
        raise ValueError("constraint_strength must be >= 0")


def _finite_result(raw: BackendResult, n_sites: int) -> BackendResult:
    """Drop rows whose objectives are non-finite (invalid thermometers)."""
    objectives = np.asarray(raw.objectives, dtype=np.float64)
    if objectives.ndim != 2:
        raise RuntimeError(
            f"domain-wall objectives must be 2-D, got shape {objectives.shape}"
        )
    valid = np.isfinite(objectives).all(axis=-1)
    decisions = np.asarray(raw.decisions)
    if decisions.ndim != 2 or int(decisions.shape[-1]) != n_sites:
        raise RuntimeError(
            f"domain-wall decisions have shape {decisions.shape}, "
            f"expected (*, {n_sites})"
        )
    energies = np.asarray(raw.energies, dtype=np.float64)
    return BackendResult(
        decisions=np.asarray(decisions[valid], dtype=np.int64),
        objectives=objectives[valid],
        energies=energies[valid],
        n_evals=int(valid.sum()),
        n_invalid=int((~valid).sum()),
    )


@dataclass
class ThrmlDomainWallBackend:
    """Annealed domain-wall samples of one Potts scalarization at a time.

    ``build_potts(w)`` returns THRML weights of ``E_w`` (beta is applied by
    the sampler). ``objective_fn`` maps valid integer states to Potts
    objectives. ``constraint_strength`` is the thermometer penalty ``P``
    from example 03; beta scales it with the other Ising weights.
    """

    n_sites: int
    n_categories: int
    build_potts: PottsBuilder
    objective_fn: ObjectiveFn
    constraint_strength: float = 4.0

    def program_kind(self, w: np.ndarray) -> str:
        """``thrml_ising`` or ``spin_ebm``, without sampling."""
        image = self._compile(w)
        if _checkerboard(image) is None:
            return "spin_ebm"
        return "thrml_ising"

    def sample_weight(
        self,
        w: np.ndarray,
        *,
        seed: int,
        betas: tuple[float, ...] = (0.2, 0.5, 1.0, 2.0, 4.0, 8.0),
        n_warmup: int = 40,
        n_samples: int = 48,
        steps_per_sample: int = 2,
    ) -> BackendResult:
        w = np.asarray(w, dtype=np.float64)
        _validate_schedule(
            betas,
            n_warmup,
            n_samples,
            steps_per_sample,
            seed,
            self.constraint_strength,
        )
        image = self._compile(w)
        permutation = _checkerboard(image)
        if permutation is None:
            return self._sample_spin_program(
                w,
                image,
                seed=seed,
                betas=betas,
                n_warmup=n_warmup,
                n_samples=n_samples,
                steps_per_sample=steps_per_sample,
            )
        return self._sample_checkerboard(
            w,
            image,
            permutation,
            seed=seed,
            betas=betas,
            n_warmup=n_warmup,
            n_samples=n_samples,
            steps_per_sample=steps_per_sample,
        )

    def _compile(self, w: np.ndarray) -> DomainWallImage:
        unary, edges, pairwise = self.build_potts(np.asarray(w, dtype=np.float64))
        image = compile_domain_wall(unary, edges, pairwise)
        if image.n_sites != int(self.n_sites) or image.n_categories != int(
            self.n_categories
        ):
            raise ValueError(
                "build_potts produced "
                f"({image.n_sites}, {image.n_categories}); backend is "
                f"({self.n_sites}, {self.n_categories})"
            )
        return image

    def _sample_checkerboard(
        self,
        w: np.ndarray,
        image: DomainWallImage,
        permutation: np.ndarray,
        *,
        seed: int,
        betas: tuple[float, ...],
        n_warmup: int,
        n_samples: int,
        steps_per_sample: int,
    ) -> BackendResult:
        def build_ising(_w: np.ndarray):
            biases, edges, couplings = image.ising_terms(self.constraint_strength)
            ordered = np.empty_like(biases)
            ordered[permutation] = biases
            moved = [(int(permutation[i]), int(permutation[j])) for i, j in edges]
            return (
                np.asarray(ordered, dtype=np.float32),
                moved,
                np.asarray(couplings, dtype=np.float32),
            )

        def objective_fn(spins_pm1: np.ndarray) -> np.ndarray:
            native = np.asarray(spins_pm1, dtype=np.float64)[..., permutation]
            return image.objectives_with_nan(native, self.objective_fn)

        def spin_to_decision(spins_pm1: np.ndarray) -> np.ndarray:
            native = np.asarray(spins_pm1, dtype=np.float64)[..., permutation]
            return image.plus_counts(native)

        raw = ThrmlIsingBackend(
            n_spins=image.n_spins,
            build_ising=build_ising,
            objective_fn=objective_fn,
            spin_to_decision=spin_to_decision,
        ).sample_weight(
            w,
            seed=seed,
            betas=betas,
            n_warmup=n_warmup,
            n_samples=n_samples,
            steps_per_sample=steps_per_sample,
        )
        return _finite_result(raw, self.n_sites)

    def _sample_spin_program(
        self,
        w: np.ndarray,
        image: DomainWallImage,
        *,
        seed: int,
        betas: tuple[float, ...],
        n_warmup: int,
        n_samples: int,
        steps_per_sample: int,
    ) -> BackendResult:
        t = _load_thrml()
        nodes = [t.SpinNode() for _ in range(image.n_spins)]
        free_blocks = [
            t.Block([nodes[index] for index in block]) for block in image.color_blocks
        ]
        observe = t.Block(nodes)
        spec = t.BlockGibbsSpec(free_blocks, [])
        samplers = [t.SpinGibbsConditional() for _ in free_blocks]
        biases, _edges, couplings = image.ising_terms(self.constraint_strength)
        n_constraint = len(image.constraint_edges)
        bias_np = np.asarray(biases, dtype=np.float32)
        coupling_np = np.asarray(couplings, dtype=np.float32)

        key = t.jax.random.key(int(seed))
        key, key_init = t.jax.random.split(key, 2)
        init_states = np.asarray(
            t.jax.random.randint(
                key_init,
                (image.n_sites,),
                0,
                int(image.n_categories),
                dtype=t.jnp.uint8,
            )
        )
        init_bool = image.encode(init_states) > 0.0
        state = [
            t.jnp.asarray(init_bool[list(block)]) for block in image.color_blocks
        ]
        schedule = t.SamplingSchedule(
            n_warmup=n_warmup,
            n_samples=n_samples,
            steps_per_sample=steps_per_sample,
        )
        collected: list[np.ndarray] = []
        for beta in betas:
            scale = t.jnp.asarray(beta, dtype=t.jnp.float32)
            factors = [
                t.SpinEBMFactor(
                    [t.Block(nodes)],
                    t.jnp.asarray(bias_np) * scale,
                )
            ]
            if n_constraint:
                left = [nodes[i] for i, _j in image.constraint_edges]
                right = [nodes[j] for _i, j in image.constraint_edges]
                factors.append(
                    t.SpinEBMFactor(
                        [t.Block(left), t.Block(right)],
                        t.jnp.asarray(coupling_np[:n_constraint]) * scale,
                    )
                )
            if image.inter_edges:
                left = [nodes[i] for i, _j in image.inter_edges]
                right = [nodes[j] for _i, j in image.inter_edges]
                factors.append(
                    t.SpinEBMFactor(
                        [t.Block(left), t.Block(right)],
                        t.jnp.asarray(coupling_np[n_constraint:]) * scale,
                    )
                )
            program = t.FactorSamplingProgram(spec, samplers, factors, [])
            key, key_sample = t.jax.random.split(key, 2)
            samples = t.sample_states(
                key_sample, program, schedule, state, [], [observe]
            )
            if len(samples) != 1:
                raise RuntimeError(
                    f"sample_states returned {len(samples)} blocks; "
                    "expected one observation Block(nodes)"
                )
            block = np.asarray(samples[0])
            if block.ndim != 2 or int(block.shape[-1]) != image.n_spins:
                raise RuntimeError(
                    "sample_states returned shape "
                    f"{tuple(block.shape)}, expected (n_samples, {image.n_spins})"
                )
            state = [
                t.jnp.asarray(np.asarray(block[-1, list(idxs)], dtype=bool))
                for idxs in image.color_blocks
            ]
            collected.append(np.where(block.astype(bool), 1.0, -1.0))

        spins = np.concatenate(collected, axis=0)
        decisions, objectives, n_invalid = image.feasible(spins, self.objective_fn)
        energies = objectives @ w
        return BackendResult(
            decisions=np.asarray(decisions, dtype=np.int64),
            objectives=np.asarray(objectives, dtype=np.float64),
            energies=np.asarray(energies, dtype=np.float64),
            n_evals=int(len(decisions)),
            n_invalid=int(n_invalid),
        )


def _checkerboard(image: DomainWallImage) -> np.ndarray | None:
    """Checkerboard index map, or None when ``ThrmlIsingBackend`` cannot run.

    An edgeless graph is 2-colored, but ``IsingEBM`` still builds a pairwise
    factor. Those images use the spin-factor program instead.
    """
    edges = image.constraint_edges + image.inter_edges
    if len(edges) == 0:
        return None
    return checkerboard_permutation(image.n_spins, edges)
