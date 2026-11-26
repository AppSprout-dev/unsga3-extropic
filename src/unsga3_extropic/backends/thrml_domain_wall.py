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

from dataclasses import dataclass, field
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
    import equinox as eqx
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
        eqx=eqx,
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
    _spin_cache: dict = field(default_factory=dict, init=False, repr=False)
    _checkerboard_cache: dict = field(default_factory=dict, init=False, repr=False)

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

        cb_key = (image.n_spins, tuple(int(p) for p in permutation))
        ising_backend = self._checkerboard_cache.get(cb_key)
        if ising_backend is None:
            ising_backend = ThrmlIsingBackend(
                n_spins=image.n_spins,
                build_ising=build_ising,
                objective_fn=objective_fn,
                spin_to_decision=spin_to_decision,
            )
            self._checkerboard_cache[cb_key] = ising_backend
        else:
            ising_backend.build_ising = build_ising
            ising_backend.objective_fn = objective_fn
            ising_backend.spin_to_decision = spin_to_decision

        raw = ising_backend.sample_weight(
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
        biases, _edges, couplings = image.ising_terms(self.constraint_strength)
        n_constraint = len(image.constraint_edges)
        bias_jnp = t.jnp.asarray(np.asarray(biases, dtype=np.float32))
        coupling_c_jnp = (
            t.jnp.asarray(np.asarray(couplings[:n_constraint], dtype=np.float32))
            if n_constraint
            else None
        )
        coupling_i_jnp = (
            t.jnp.asarray(np.asarray(couplings[n_constraint:], dtype=np.float32))
            if image.inter_edges
            else None
        )

        cache_key = (
            int(image.n_spins),
            tuple(tuple(block) for block in image.color_blocks),
            tuple(image.constraint_edges),
            tuple(image.inter_edges),
            int(n_warmup),
            int(n_samples),
            int(steps_per_sample),
        )
        cached = self._spin_cache.get(cache_key)
        if cached is None:
            nodes = [t.SpinNode() for _ in range(image.n_spins)]
            node_block = t.Block(nodes)
            free_blocks = [
                t.Block([nodes[index] for index in block])
                for block in image.color_blocks
            ]
            observe = node_block
            spec = t.BlockGibbsSpec(free_blocks, [])
            samplers = [t.SpinGibbsConditional() for _ in free_blocks]
            schedule = t.SamplingSchedule(
                n_warmup=n_warmup,
                n_samples=n_samples,
                steps_per_sample=steps_per_sample,
            )
            scale0 = t.jnp.asarray(betas[0], dtype=t.jnp.float32)
            factors0 = [t.SpinEBMFactor([node_block], bias_jnp * scale0)]
            if n_constraint:
                c_left = t.Block([nodes[i] for i, _j in image.constraint_edges])
                c_right = t.Block([nodes[j] for _i, j in image.constraint_edges])
                factors0.append(
                    t.SpinEBMFactor([c_left, c_right], coupling_c_jnp * scale0)
                )
            else:
                c_left = None
                c_right = None
            if image.inter_edges:
                i_left = t.Block([nodes[i] for i, _j in image.inter_edges])
                i_right = t.Block([nodes[j] for _i, j in image.inter_edges])
                factors0.append(
                    t.SpinEBMFactor([i_left, i_right], coupling_i_jnp * scale0)
                )
            else:
                i_left = None
                i_right = None
            prog0 = t.FactorSamplingProgram(spec, samplers, factors0, [])
            _, static_prog = t.eqx.partition(prog0, t.eqx.is_array)

            @t.jax.jit
            def run_stage(k_samp, dynamic_prog, state_in):
                prog = t.eqx.combine(dynamic_prog, static_prog)
                return t.sample_states(k_samp, prog, schedule, state_in, [], [observe])

            cached = (
                node_block,
                c_left,
                c_right,
                i_left,
                i_right,
                spec,
                samplers,
                run_stage,
            )
            self._spin_cache[cache_key] = cached
        else:
            (
                node_block,
                c_left,
                c_right,
                i_left,
                i_right,
                spec,
                samplers,
                run_stage,
            ) = cached

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
        collected: list[np.ndarray] = []
        for beta in betas:
            scale = t.jnp.asarray(beta, dtype=t.jnp.float32)
            factors = [t.SpinEBMFactor([node_block], bias_jnp * scale)]
            if n_constraint:
                factors.append(
                    t.SpinEBMFactor([c_left, c_right], coupling_c_jnp * scale)
                )
            if image.inter_edges:
                factors.append(
                    t.SpinEBMFactor([i_left, i_right], coupling_i_jnp * scale)
                )
            program = t.FactorSamplingProgram(spec, samplers, factors, [])
            dynamic_prog, _ = t.eqx.partition(program, t.eqx.is_array)
            key, key_sample = t.jax.random.split(key, 2)
            samples = run_stage(key_sample, dynamic_prog, state)
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
