"""THRML Ising backend (block Gibbs + temperature anneal).

Sampling matches the THRML 0.1.4 quickstart on
https://docs.thrml.ai and the extropic-ai/thrml README:

    SpinNode, Block, IsingEBM, IsingSamplingProgram,
    SamplingSchedule, sample_states, hinton_init, jax.random.key

``IsingEBM`` energy, with bool states ``True -> +1`` and ``False -> -1``::

    E(s) = -beta * (b · s + sum_{ij} J_ij s_i s_j)

``SamplingSchedule`` has no temperature field. Annealing rebuilds
``IsingEBM`` and ``IsingSamplingProgram`` at each beta and carries the
free-block state forward. Free blocks are the documented checkerboard
``nodes[::2]``, ``nodes[1::2]``, so every edge must join opposite parities.

JAX and THRML are imported inside ``sample_weight``. They are optional at
import time: archive code and ``ExactEwMetropolisBackend`` run without them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from unsga3_extropic.backends.base import BackendResult

# build_ising(w) -> (biases[n], edge_list[(i,j),...], j_weights[n_edges])
IsingBuilder = Callable[
    [np.ndarray], tuple[np.ndarray, list[tuple[int, int]], np.ndarray]
]
# objectives(spins_pm1) -> (..., n_obj)
ObjectiveFn = Callable[[np.ndarray], np.ndarray]


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


def _validate_ising_params(
    n_spins: int,
    biases: np.ndarray,
    edge_idx,
    weights: np.ndarray,
) -> list[tuple[int, int]]:
    """Check shapes and the even/odd block coloring used by this backend."""
    if n_spins < 1:
        raise ValueError("n_spins must be >= 1")
    if biases.shape != (n_spins,):
        raise ValueError(f"biases shape {biases.shape} != ({n_spins},)")
    edges: list[tuple[int, int]] = []
    for edge in edge_idx:
        if len(edge) != 2:
            raise ValueError(f"edge {edge!r} must be a pair of node indices")
        i, j = int(edge[0]), int(edge[1])
        if not (0 <= i < n_spins and 0 <= j < n_spins):
            raise ValueError(f"edge {(i, j)} out of range for n_spins={n_spins}")
        if (i % 2) == (j % 2):
            raise ValueError(
                f"edge {(i, j)} lies inside one checkerboard block; "
                "free blocks are even and odd indices, as in the THRML "
                "two-color chain (https://docs.thrml.ai)"
            )
        edges.append((i, j))
    if weights.shape != (len(edges),):
        raise ValueError(f"weights shape {weights.shape} != ({len(edges)},)")
    return edges


@dataclass
class ThrmlIsingBackend:
    """Annealed THRML ``IsingEBM`` sampling for one weight vector at a time.

    ``build_ising(w)`` returns Ising parameters of ``E_w`` in the THRML
    convention above (beta is applied separately by the anneal).
    ``objective_fn`` maps ±1 spin arrays shaped ``(..., n)`` to objectives
    ``(..., M)``.
    """

    n_spins: int
    build_ising: IsingBuilder
    objective_fn: ObjectiveFn
    spin_to_decision: Callable[[np.ndarray], np.ndarray] | None = None

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
        biases_np, edge_idx, j_np = self.build_ising(w)
        biases_np = np.asarray(biases_np, dtype=np.float32)
        j_np = np.asarray(j_np, dtype=np.float32)
        edges_ij = _validate_ising_params(self.n_spins, biases_np, edge_idx, j_np)

        # Optional dependency: see module docstring.
        import jax
        import jax.numpy as jnp
        from thrml import Block, SamplingSchedule, SpinNode, sample_states
        from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

        nodes = [SpinNode() for _ in range(self.n_spins)]
        edges = [(nodes[i], nodes[j]) for i, j in edges_ij]
        free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
        observe = Block(nodes)

        biases = jnp.asarray(biases_np)
        couplings = jnp.asarray(j_np)

        key = jax.random.key(int(seed))
        key, k_init = jax.random.split(key, 2)
        ebm0 = IsingEBM(
            nodes,
            edges,
            biases,
            couplings,
            jnp.asarray(betas[0], dtype=jnp.float32),
        )
        state = hinton_init(k_init, ebm0, free_blocks, ())

        collected: list[np.ndarray] = []
        for beta in betas:
            ebm = IsingEBM(
                nodes,
                edges,
                biases,
                couplings,
                jnp.asarray(beta, dtype=jnp.float32),
            )
            program = IsingSamplingProgram(ebm, free_blocks, clamped_blocks=[])
            schedule = SamplingSchedule(
                n_warmup=n_warmup,
                n_samples=n_samples,
                steps_per_sample=steps_per_sample,
            )
            key, k_samp = jax.random.split(key, 2)
            samples = sample_states(
                k_samp, program, schedule, state, [], [observe]
            )
            if len(samples) != 1:
                raise RuntimeError(
                    f"sample_states returned {len(samples)} blocks; "
                    "expected one observation Block(nodes)"
                )
            block = samples[0]
            if getattr(block, "ndim", None) != 2 or int(block.shape[-1]) != self.n_spins:
                raise RuntimeError(
                    "sample_states returned shape "
                    f"{tuple(getattr(block, 'shape', ()))}, expected "
                    f"(n_samples, {self.n_spins})"
                )
            spins_bool = np.asarray(block, dtype=bool)
            # Carry the terminal observation, split into the same free blocks.
            state = [
                jnp.asarray(spins_bool[-1, ::2]),
                jnp.asarray(spins_bool[-1, 1::2]),
            ]
            collected.append(spins_bool)

        all_bool = np.concatenate(collected, axis=0)
        spins_pm1 = np.where(all_bool, 1.0, -1.0).astype(np.float64)
        if self.spin_to_decision is not None:
            decisions = self.spin_to_decision(spins_pm1)
        else:
            decisions = ((spins_pm1 + 1.0) / 2.0).astype(np.float64)

        objs = np.asarray(self.objective_fn(spins_pm1), dtype=np.float64)
        e_w = (objs * w.reshape(1, -1)).sum(axis=-1)
        return BackendResult(
            decisions=decisions,
            objectives=objs,
            energies=e_w,
            n_evals=len(decisions),
        )
