"""THRML Ising/Potts-expressible backend (block Gibbs + anneal).

Requires energies that remain Ising (or Potts) after weight-linear
combination: E_w = sum_k w_k E_k where each E_k is a bias+pairwise form.
The problem supplies biases/weights as linear functions of w.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from unsga3_extropic.backends.base import BackendResult

# Types for problem hooks
# build_ising(w) -> (biases[n], edge_list[(i,j),...], j_weights[n_edges])
IsingBuilder = Callable[
    [np.ndarray], tuple[np.ndarray, list[tuple[int, int]], np.ndarray]
]
# objectives(spins_pm1 or bits) -> (n, n_obj)
ObjectiveFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class ThrmlIsingBackend:
    """Annealed THRML ``IsingEBM`` sampling for one weight vector at a time.

    ``build_ising(w)`` must return Ising parameters for E_w.
    ``objective_fn`` maps ±1 spin arrays shaped (..., n) → objectives (..., M).
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
        import jax
        import jax.numpy as jnp
        from thrml import Block, SamplingSchedule, SpinNode, sample_states
        from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

        w = np.asarray(w, dtype=np.float64)
        biases_np, edge_idx, j_np = self.build_ising(w)
        biases_np = np.asarray(biases_np, dtype=np.float32)
        j_np = np.asarray(j_np, dtype=np.float32)

        nodes = [SpinNode() for _ in range(self.n_spins)]
        edges = [(nodes[i], nodes[j]) for i, j in edge_idx]
        free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
        observe = Block(nodes)

        biases = jnp.asarray(biases_np)
        weights = jnp.asarray(j_np)

        key = jax.random.PRNGKey(seed)
        key, k_init = jax.random.split(key)
        ebm0 = IsingEBM(
            nodes, edges, biases, weights, jnp.asarray(betas[0], dtype=jnp.float32)
        )
        state = hinton_init(k_init, ebm0, free_blocks, ())

        collected: list[np.ndarray] = []
        for beta in betas:
            ebm = IsingEBM(
                nodes, edges, biases, weights, jnp.asarray(beta, dtype=jnp.float32)
            )
            program = IsingSamplingProgram(ebm, free_blocks, clamped_blocks=[])
            schedule = SamplingSchedule(
                n_warmup=n_warmup,
                n_samples=n_samples,
                steps_per_sample=steps_per_sample,
            )
            key, k_samp = jax.random.split(key)
            samples = sample_states(k_samp, program, schedule, state, [], [observe])
            spins_bool = np.asarray(samples[0])  # (n_samples, n) bool; True=+1
            last = spins_bool[-1]
            state = [jnp.asarray(last[::2]), jnp.asarray(last[1::2])]
            collected.append(spins_bool)

        all_bool = np.concatenate(collected, axis=0)
        # THRML SpinNode: True -> +1, False -> -1
        spins_pm1 = np.where(all_bool, 1.0, -1.0).astype(np.float64)
        if self.spin_to_decision is not None:
            decisions = self.spin_to_decision(spins_pm1)
        else:
            decisions = ((spins_pm1 + 1.0) / 2.0).astype(np.float64)  # bits in {0,1}

        objs = np.asarray(self.objective_fn(spins_pm1), dtype=np.float64)
        e_w = (objs * w.reshape(1, -1)).sum(axis=-1)
        return BackendResult(
            decisions=decisions,
            objectives=objs,
            energies=e_w,
            n_evals=len(decisions),
        )
