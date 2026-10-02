"""Optional Torx two-pbit circuit. Not a search backend.

Documented entry point, re-checked 2026-10-02 against
https://docs.torx.ai/en/latest/ and
https://docs.torx.ai/en/latest/getting-started.html
(``extro-torx`` 0.0.2, Python >= 3.11):

    DiscretePCircuit([PSWAP([0, 1])])
    thetas = [jnp.array([jnp.log(0.3 / 0.7)])]
    BranchingSimulator(num_samples=20_000)
    sim.build_circuit(circuit, thetas)
    sim.sample(compiled, state, jax.random.key(seed))

``PSWAP`` swaps two pbits with probability ``p = sigmoid(theta)``.
``theta`` is the logit of ``p``. Gates are structure; parameters are a
separate list. From ``|10)``, about ``1 - p`` of the samples stay
``|10)`` and about ``p`` become ``|01)``.

This module samples that circuit and stops. It is not a
``SamplingBackend``, and ``WeightSweepLoop`` does not call it. ``torx``
is imported inside ``TorxPswapCircuit.sample`` so a missing extra does
not break import of the core package on Python 3.10.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

# Docs quickstart: p(swap) = 0.3, 20_000 samples, seed 0, start at |10).
DOCUMENTED_P_SWAP = 0.3
DOCUMENTED_N_SAMPLES = 20_000
DOCUMENTED_SEED = 0
DOCUMENTED_INITIAL_STATE = (1, 0)
# Loose absolute tolerance on the empirical rate at DOCUMENTED_N_SAMPLES.
# Binomial SE at p=0.3 is about 0.003; 0.05 is the smoke bound.
SMOKE_ABS_TOLERANCE = 0.05


@dataclass(frozen=True)
class PswapDraw:
    """Samples from one ``PSWAP`` circuit, plus the empirical rates."""

    samples: np.ndarray  # (n_samples, 2) pbits in {0, 1}
    stay_rate: float
    swap_rate: float
    p_swap: float
    n_samples: int
    seed: int
    initial_state: tuple[int, int]


def _load_torx():
    """Import Torx on demand.

    The ``torx`` extra requires Python >= 3.11 because ``extro-torx``
    does. Core ``unsga3_extropic`` stays importable on 3.10 without it.
    """
    if sys.version_info < (3, 11):
        raise ImportError(
            "the torx extra requires Python >= 3.11 (extro-torx). "
            "unsga3_extropic itself stays importable on Python 3.10."
        )
    try:
        # Optional dependency: see module docstring.
        import jax
        import jax.numpy as jnp
        from torx.psc import BranchingSimulator, DiscretePCircuit, PSWAP
    except ImportError as exc:
        raise ImportError(
            "extro-torx is not installed. On Python >= 3.11: "
            'pip install -e ".[torx]"'
        ) from exc
    return jax, jnp, DiscretePCircuit, BranchingSimulator, PSWAP


def _as_pbit_pair(initial_state) -> tuple[int, int]:
    try:
        bits = tuple(int(bit) for bit in initial_state)
    except TypeError as exc:
        raise ValueError("initial_state must be two pbits, each 0 or 1") from exc
    if len(bits) != 2 or any(bit not in (0, 1) for bit in bits):
        raise ValueError("initial_state must be two pbits, each 0 or 1")
    return bits[0], bits[1]


class TorxPswapCircuit:
    """Two-site ``PSWAP`` sampled with ``BranchingSimulator``.

    Not a ``SamplingBackend``. Installing the ``torx`` extra does not
    change ``WeightSweepLoop``.
    """

    def sample(
        self,
        *,
        p_swap: float = DOCUMENTED_P_SWAP,
        n_samples: int = DOCUMENTED_N_SAMPLES,
        seed: int = DOCUMENTED_SEED,
        initial_state: tuple[int, int] = DOCUMENTED_INITIAL_STATE,
    ) -> PswapDraw:
        """Draw ``n_samples`` outputs of the documented two-pbit circuit."""
        if not 0.0 < float(p_swap) < 1.0:
            raise ValueError("p_swap must lie in (0, 1); theta is logit(p)")
        if int(n_samples) < 1:
            raise ValueError("n_samples must be >= 1")
        if int(seed) < 0:
            raise ValueError("seed must be >= 0")
        state_pair = _as_pbit_pair(initial_state)
        p_swap = float(p_swap)
        n_samples = int(n_samples)
        seed = int(seed)

        jax, jnp, DiscretePCircuit, BranchingSimulator, PSWAP = _load_torx()
        circuit = DiscretePCircuit([PSWAP([0, 1])])
        # Same logit leaf as the docs quickstart: theta = log(p / (1 - p)).
        thetas = [jnp.array([jnp.log(p_swap / (1.0 - p_swap))])]
        sim = BranchingSimulator(num_samples=n_samples)
        compiled = sim.build_circuit(circuit, thetas)
        state = jnp.array(state_pair, dtype=jnp.int32)
        samples = np.asarray(
            sim.sample(compiled, state, jax.random.key(seed))
        )
        if samples.shape != (n_samples, 2):
            raise RuntimeError(
                f"BranchingSimulator.sample returned shape {samples.shape}, "
                f"expected ({n_samples}, 2)"
            )

        initial = np.array(state_pair, dtype=np.int32)
        swapped = np.array((state_pair[1], state_pair[0]), dtype=np.int32)
        stay_rate = float(np.mean(np.all(samples == initial, axis=1)))
        swap_rate = float(np.mean(np.all(samples == swapped, axis=1)))
        return PswapDraw(
            samples=samples,
            stay_rate=stay_rate,
            swap_rate=swap_rate,
            p_swap=p_swap,
            n_samples=n_samples,
            seed=seed,
            initial_state=state_pair,
        )
