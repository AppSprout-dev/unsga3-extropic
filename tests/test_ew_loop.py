"""Exact E_w Metropolis and the weight-sweep loop (no THRML)."""

from __future__ import annotations

import json

import numpy as np

from unsga3_extropic import AnnealConfig, WeightSweepLoop
from unsga3_extropic.backends import ExactEwMetropolisBackend


def _objective(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.float64)
    ones = bits.sum(axis=-1)
    n = bits.shape[-1]
    return np.stack([ones, n - ones], axis=-1)


def test_ew_energy_matches_weights_and_is_reproducible():
    backend = ExactEwMetropolisBackend(n_bits=4, objective_fn=_objective)
    w = np.array([0.2, 0.8])
    kwargs = dict(
        seed=5,
        betas=(0.5, 2.0),
        n_warmup=3,
        n_samples=4,
        steps_per_sample=1,
    )
    first = backend.sample_weight(w, **kwargs)
    second = backend.sample_weight(w, **kwargs)
    assert first.decisions.shape == (8, 4)
    assert first.objectives.shape == (8, 2)
    assert np.allclose(first.energies, first.objectives @ w)
    assert np.allclose(first.decisions, second.decisions)
    assert set(np.unique(first.decisions)).issubset({0.0, 1.0})
    # init + each proposal. Recording f does not add another counted eval.
    assert first.n_evals == 1 + 2 * (3 + 4 * 1)


def test_weight_sweep_summary_is_json_safe():
    backend = ExactEwMetropolisBackend(n_bits=3, objective_fn=_objective)
    loop = WeightSweepLoop(
        backend=backend,
        n_weights=3,
        n_obj=2,
        anneal=AnnealConfig(betas=(1.0,), n_warmup=2, n_samples=3, steps_per_sample=1),
        seed=1,
    )
    result = loop.run()
    assert result.weights.shape == (3, 2)
    assert np.allclose(result.weights.sum(axis=1), 1.0)
    assert result.total_evals == 3 * result.per_weight[0].n_evals
    decisions, front = result.nondominated_front()
    assert len(decisions) == len(front) >= 1
    summary = result.summary()
    assert summary["n_weights"] == 3
    assert "hv_2d_data_ref" in summary
    json.dumps(summary)
