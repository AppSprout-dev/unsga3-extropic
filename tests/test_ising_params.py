"""Ising parameter checks that do not sample (no JAX required)."""

from __future__ import annotations

import numpy as np
import pytest

from unsga3_extropic.backends.thrml_ising import ThrmlIsingBackend


def _backend(
    n: int,
    edges: list[tuple[int, int]],
    *,
    biases: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> ThrmlIsingBackend:
    def build(_w: np.ndarray):
        b = (
            np.zeros(n, np.float32)
            if biases is None
            else np.asarray(biases, dtype=np.float32)
        )
        ww = (
            np.ones(len(list(edges)), np.float32)
            if weights is None
            else np.asarray(weights, dtype=np.float32)
        )
        return b, edges, ww

    return ThrmlIsingBackend(
        n_spins=n,
        build_ising=build,
        objective_fn=lambda spins: np.zeros(spins.shape[:-1] + (1,)),
    )


def test_rejects_same_parity_edge():
    backend = _backend(4, [(0, 2)])
    with pytest.raises(ValueError, match="checkerboard"):
        backend.sample_weight(np.ones(1), seed=0, betas=(1.0,), n_warmup=1, n_samples=1, steps_per_sample=1)


def test_rejects_bias_shape():
    backend = _backend(3, [(0, 1)], biases=np.zeros(2, np.float32))
    with pytest.raises(ValueError, match="biases"):
        backend.sample_weight(np.ones(1), seed=0, betas=(1.0,), n_warmup=0, n_samples=1, steps_per_sample=1)


def test_rejects_empty_betas_and_negative_seed():
    backend = _backend(2, [(0, 1)])
    with pytest.raises(ValueError, match="betas"):
        backend.sample_weight(np.ones(1), seed=0, betas=(), n_warmup=0, n_samples=1, steps_per_sample=1)
    with pytest.raises(ValueError, match="seed"):
        backend.sample_weight(np.ones(1), seed=-1, betas=(1.0,), n_warmup=0, n_samples=1, steps_per_sample=1)
