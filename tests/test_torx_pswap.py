"""Statistical smoke for the documented Torx PSWAP circuit.

Skipped when ``extro-torx`` is not installed. The core suite does not
need this extra.
"""

from __future__ import annotations

import numpy as np
import pytest

from unsga3_extropic.torx_circuit import (
    DOCUMENTED_INITIAL_STATE,
    DOCUMENTED_N_SAMPLES,
    DOCUMENTED_P_SWAP,
    DOCUMENTED_SEED,
    SMOKE_ABS_TOLERANCE,
    TorxPswapCircuit,
)

pytestmark = pytest.mark.torx


def test_pswap_stay_and_swap_rates():
    """Docs quickstart: PSWAP on |10) with p(swap) = 0.3, 20_000 samples."""
    pytest.importorskip("torx")
    draw = TorxPswapCircuit().sample(
        p_swap=DOCUMENTED_P_SWAP,
        n_samples=DOCUMENTED_N_SAMPLES,
        seed=DOCUMENTED_SEED,
        initial_state=DOCUMENTED_INITIAL_STATE,
    )
    samples = np.asarray(draw.samples)
    assert samples.shape == (DOCUMENTED_N_SAMPLES, 2)
    assert set(np.unique(samples)).issubset({0, 1})

    initial = np.array(DOCUMENTED_INITIAL_STATE, dtype=np.int32)
    swapped = np.array([0, 1], dtype=np.int32)
    stay = float(np.mean(np.all(samples == initial, axis=1)))
    swap = float(np.mean(np.all(samples == swapped, axis=1)))
    assert stay == pytest.approx(draw.stay_rate)
    assert swap == pytest.approx(draw.swap_rate)
    assert abs(stay - (1.0 - DOCUMENTED_P_SWAP)) < SMOKE_ABS_TOLERANCE
    assert abs(swap - DOCUMENTED_P_SWAP) < SMOKE_ABS_TOLERANCE
    assert abs(stay + swap - 1.0) < 1e-9
