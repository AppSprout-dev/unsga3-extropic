"""THRML Potts backend (categorical block Gibbs + weight-rescaled anneal).

Sampling follows THRML 0.1.4 example 00 and the Potts half of example 03
(https://docs.thrml.ai):

    CategoricalNode, Block, BlockGibbsSpec,
    CategoricalEBMFactor, CategoricalGibbsConditional,
    FactorSamplingProgram, FactorizedEBM,
    SamplingSchedule, sample_states

``CategoricalNode()`` takes no category count. ``K`` is
``CategoricalGibbsConditional(n_categories).n_categories``. Sampled states
are integers in ``[0, K)``.

``DiscreteEBMFactor.energy`` in THRML 0.1.4 returns ``-sum W[state]``
(high weight, low energy). ``build_potts`` therefore returns those weights,
already scaled by the scalarization ``w``. ``SamplingSchedule`` has no beta.
Annealing multiplies the weights by beta and rebuilds the program, which is
what example 03 does, and carries the free-block state forward.

Free blocks are an explicit coloring. The default is the even/odd chain
coloring. An edge whose endpoints share a color is rejected before any
sample and before JAX is imported.

JAX and THRML are imported inside ``_load_thrml``. They are optional at
package import time: archive code and ``ExactEwMetropolisBackend`` run
without them. Domain-wall Ising is a different encoding and is not built
here.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

import numpy as np

from unsga3_extropic.backends.base import BackendResult

# build_potts(w) -> (unary W[n, K], edges, pairwise W[n_edges, K, K])
# W are THRML weights: FactorizedEBM energy is -sum W[state], not +W.
PottsBuilder = Callable[
    [np.ndarray], tuple[np.ndarray, list[tuple[int, int]], np.ndarray]
]
# objective_fn(states[..., n]) -> (..., n_obj). states are integers in [0, K).
ObjectiveFn = Callable[[np.ndarray], np.ndarray]


def _load_thrml() -> SimpleNamespace:
    """Import JAX and THRML. Optional at package import time."""
    import jax
    import jax.numpy as jnp
    from thrml import (
        Block,
        BlockGibbsSpec,
        CategoricalNode,
        FactorSamplingProgram,
        SamplingSchedule,
        sample_states,
    )
    from thrml.models import (
        CategoricalEBMFactor,
        CategoricalGibbsConditional,
        FactorizedEBM,
    )

    return SimpleNamespace(
        jax=jax,
        jnp=jnp,
        Block=Block,
        BlockGibbsSpec=BlockGibbsSpec,
        CategoricalNode=CategoricalNode,
        FactorSamplingProgram=FactorSamplingProgram,
        SamplingSchedule=SamplingSchedule,
        sample_states=sample_states,
        CategoricalEBMFactor=CategoricalEBMFactor,
        CategoricalGibbsConditional=CategoricalGibbsConditional,
        FactorizedEBM=FactorizedEBM,
    )


def _validate_schedule(
    betas: tuple[float, ...],
    n_warmup: int,
    n_samples: int,
    steps_per_sample: int,
    seed: int,
) -> None:
    if len(tuple(betas)) < 1:
        raise ValueError("betas must be non-empty")
    if n_warmup < 0 or n_samples < 1 or steps_per_sample < 1:
        raise ValueError(
            "require n_warmup >= 0, n_samples >= 1, steps_per_sample >= 1"
        )
    if int(seed) < 0:
        raise ValueError("seed must be >= 0")


def even_odd_coloring(n_sites: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Even/odd free blocks for a path (example 00 / example 03 chain)."""
    if n_sites < 2:
        raise ValueError("even/odd coloring needs n_sites >= 2")
    return tuple(range(0, n_sites, 2)), tuple(range(1, n_sites, 2))


def validate_potts_instance(
    n_sites: int,
    n_categories: int,
    unary: np.ndarray,
    edges,
    pairwise: np.ndarray,
    coloring: tuple[tuple[int, ...], ...] | None,
) -> tuple[list[tuple[int, int]], tuple[tuple[int, ...], ...]]:
    """Check shapes and that every edge crosses free blocks.

    ``coloring`` is a partition of ``range(n_sites)`` into independent sets.
    ``None`` selects :func:`even_odd_coloring`.
    """
    if n_sites < 1:
        raise ValueError("n_sites must be >= 1")
    if not (1 <= int(n_categories) <= 256):
        raise ValueError("n_categories must lie in 1..256 (uint8 states)")
    k = int(n_categories)
    if unary.shape != (n_sites, k):
        raise ValueError(f"unary shape {unary.shape} != ({n_sites}, {k})")
    if coloring is None:
        resolved = even_odd_coloring(n_sites)
    else:
        resolved = tuple(tuple(int(i) for i in block) for block in coloring)
    if len(resolved) < 1:
        raise ValueError("coloring must contain at least one free block")
    color = [-1] * n_sites
    for c, block in enumerate(resolved):
        if len(block) < 1:
            raise ValueError(f"free block {c} is empty")
        for i in block:
            if not (0 <= i < n_sites):
                raise ValueError(f"coloring index {i} out of range for n_sites={n_sites}")
            if color[i] != -1:
                raise ValueError(f"node {i} appears in more than one free block")
            color[i] = c
    missing = [i for i, c in enumerate(color) if c < 0]
    if missing:
        raise ValueError(f"coloring does not cover nodes {missing}")

    clean_edges: list[tuple[int, int]] = []
    for edge in edges:
        if len(edge) != 2:
            raise ValueError(f"edge {edge!r} must be a pair of node indices")
        i, j = int(edge[0]), int(edge[1])
        if not (0 <= i < n_sites and 0 <= j < n_sites):
            raise ValueError(f"edge {(i, j)} out of range for n_sites={n_sites}")
        if color[i] == color[j]:
            raise ValueError(
                f"edge {(i, j)} lies inside one free block (color {color[i]}); "
                "neighbors must not share a free block"
            )
        clean_edges.append((i, j))
    if pairwise.shape != (len(clean_edges), k, k):
        raise ValueError(
            f"pairwise shape {pairwise.shape} != ({len(clean_edges)}, {k}, {k})"
        )
    return clean_edges, resolved


def factorized_potts_energy(
    n_sites: int,
    unary_weights: np.ndarray,
    edges: list[tuple[int, int]],
    pairwise_weights: np.ndarray,
    states: np.ndarray,
) -> np.ndarray:
    """``FactorizedEBM`` energy of categorical states at the given weights.

    Energy is ``-sum W[state]`` (THRML 0.1.4). ``states`` has shape
    ``(..., n_sites)``. This is the beta = 1 energy when ``unary_weights``
    and ``pairwise_weights`` are the arrays returned by ``build_potts``.
    Multiplying those weights by beta multiplies this energy by beta.
    """
    t = _load_thrml()
    states_np = np.asarray(states)
    if states_np.shape[-1] != n_sites:
        raise ValueError(
            f"states shape {states_np.shape} does not end with n_sites={n_sites}"
        )
    nodes = [t.CategoricalNode() for _ in range(n_sites)]
    unary = t.jnp.asarray(np.asarray(unary_weights, dtype=np.float32))
    factors = [t.CategoricalEBMFactor([t.Block(nodes)], unary)]
    if len(edges):
        left = [nodes[i] for i, _j in edges]
        right = [nodes[j] for _i, j in edges]
        pair = t.jnp.asarray(np.asarray(pairwise_weights, dtype=np.float32))
        factors.append(t.CategoricalEBMFactor([t.Block(left), t.Block(right)], pair))
    ebm = t.FactorizedEBM(factors)
    observe = [t.Block(nodes)]
    flat = np.ascontiguousarray(states_np, dtype=np.uint8).reshape(-1, n_sites)

    def one(state):
        return ebm.energy([state], observe)

    energies = t.jax.vmap(one)(t.jnp.asarray(flat))
    out = np.asarray(energies, dtype=np.float64).reshape(states_np.shape[:-1])
    return out


@dataclass
class ThrmlPottsBackend:
    """Annealed THRML categorical sampling for one weight vector at a time.

    ``build_potts(w)`` returns THRML weights of ``E_w`` (beta applied
    separately by rescaling those weights). ``objective_fn`` maps integer
    states shaped ``(..., n_sites)`` to objectives ``(..., M)``.
    """

    n_sites: int
    n_categories: int
    build_potts: PottsBuilder
    objective_fn: ObjectiveFn
    coloring: tuple[tuple[int, ...], ...] | None = None

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
        _validate_schedule(betas, n_warmup, n_samples, steps_per_sample, seed)
        unary_np, edge_idx, pair_np = self.build_potts(w)
        unary_np = np.asarray(unary_np, dtype=np.float32)
        pair_np = np.asarray(pair_np, dtype=np.float32)
        edges, coloring = validate_potts_instance(
            self.n_sites,
            self.n_categories,
            unary_np,
            edge_idx,
            pair_np,
            self.coloring,
        )

        t = _load_thrml()
        nodes = [t.CategoricalNode() for _ in range(self.n_sites)]
        free_blocks = [t.Block([nodes[i] for i in idxs]) for idxs in coloring]
        observe = t.Block(nodes)
        spec = t.BlockGibbsSpec(free_blocks, [])
        # K lives on the conditional. One conditional per free block, as in
        # example 03. Reused across temperatures; only the weights change.
        samplers = [
            t.CategoricalGibbsConditional(int(self.n_categories)) for _ in free_blocks
        ]
        if any(int(s.n_categories) != int(self.n_categories) for s in samplers):
            raise RuntimeError("CategoricalGibbsConditional.n_categories != K")

        key = t.jax.random.key(int(seed))
        state = []
        for idxs in coloring:
            key, k_init = t.jax.random.split(key, 2)
            state.append(
                t.jax.random.randint(
                    k_init,
                    (len(idxs),),
                    0,
                    int(self.n_categories),
                    dtype=t.jnp.uint8,
                )
            )

        collected: list[np.ndarray] = []
        schedule = t.SamplingSchedule(
            n_warmup=n_warmup,
            n_samples=n_samples,
            steps_per_sample=steps_per_sample,
        )
        for beta in betas:
            scale = t.jnp.asarray(beta, dtype=t.jnp.float32)
            unary_b = t.jnp.asarray(unary_np) * scale
            factors = [t.CategoricalEBMFactor([t.Block(nodes)], unary_b)]
            if edges:
                left = [nodes[i] for i, _j in edges]
                right = [nodes[j] for _i, j in edges]
                pair_b = t.jnp.asarray(pair_np) * scale
                factors.append(
                    t.CategoricalEBMFactor([t.Block(left), t.Block(right)], pair_b)
                )
            program = t.FactorSamplingProgram(spec, samplers, factors, [])
            key, k_samp = t.jax.random.split(key, 2)
            samples = t.sample_states(
                k_samp, program, schedule, state, [], [observe]
            )
            if len(samples) != 1:
                raise RuntimeError(
                    f"sample_states returned {len(samples)} blocks; "
                    "expected one observation Block(nodes)"
                )
            block = np.asarray(samples[0])
            if block.ndim != 2 or int(block.shape[-1]) != self.n_sites:
                raise RuntimeError(
                    "sample_states returned shape "
                    f"{tuple(block.shape)}, expected (n_samples, {self.n_sites})"
                )
            if int(block.min()) < 0 or int(block.max()) >= int(self.n_categories):
                raise RuntimeError(
                    f"sample states outside [0, {self.n_categories})"
                )
            state = [
                t.jnp.asarray(block[-1, list(idxs)], dtype=t.jnp.uint8)
                for idxs in coloring
            ]
            collected.append(block)

        all_states = np.concatenate(collected, axis=0).astype(np.int64)
        objectives = np.asarray(self.objective_fn(all_states), dtype=np.float64)
        energies = factorized_potts_energy(
            self.n_sites, unary_np, edges, pair_np, all_states
        )
        return BackendResult(
            decisions=all_states,
            objectives=objectives,
            energies=np.asarray(energies, dtype=np.float64),
            n_evals=len(all_states),
        )
