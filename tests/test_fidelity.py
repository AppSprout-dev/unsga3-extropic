"""Front comparison harness (NumPy). One Ising loop score is marked thrml."""

from __future__ import annotations

import numpy as np
import pytest

from unsga3_extropic import AnnealConfig, WeightSweepLoop
from unsga3_extropic.archive import hypervolume_2d
from unsga3_extropic.fidelity import (
    compare_fronts,
    coverage,
    generational_distance,
    load_front,
)
from unsga3_extropic.problems import CodonIsingProblem


def test_hypervolume_matches_the_known_two_point_front():
    front = np.array([[0.0, 1.0], [1.0, 0.0]])
    # Slabs against ref (2, 2): width 1 at height (2-0) plus width 1 at height (2-1).
    assert hypervolume_2d(front, ref=(2.0, 2.0)) == pytest.approx(3.0)
    assert hypervolume_2d(np.array([[0.0, 0.0]]), ref=(1.0, 1.0)) == pytest.approx(1.0)
    assert compare_fronts(front)["hypervolume_2d"] == pytest.approx(
        hypervolume_2d(front, ref=tuple(compare_fronts(front)["hv_ref"]))
    )


def test_generational_distance_and_coverage():
    reference = np.array([[0.0, 0.0], [1.0, 1.0]])
    front = np.array([[0.0, 1.0]])
    assert generational_distance(front, reference) == pytest.approx(1.0)
    assert generational_distance(np.zeros((0, 2)), reference) == 0.0
    better = np.array([[0.0, 0.0]])
    worse = np.array([[1.0, 1.0], [0.5, 2.0]])
    assert coverage(better, worse) == pytest.approx(1.0)
    assert coverage(worse, better) == pytest.approx(0.0)
    assert coverage(worse, better, minimize=False) == pytest.approx(1.0)
    assert coverage(worse, worse) == pytest.approx(1.0)
    assert coverage(np.zeros((0, 2)), worse) == pytest.approx(0.0)


def test_codon_enumerate_front_scores_itself():
    problem = CodonIsingProblem(n_spins=6, J=1.0, seed=3, global_bias=0.05)
    _bits, front = problem.enumerate_front()
    assert len(front) >= 1
    ref = (
        float(front[:, 0].max() + 1.0),
        float(front[:, 1].max() + 1.0),
    )
    assert hypervolume_2d(front, ref=ref) > 0.0
    assert generational_distance(front, front) == pytest.approx(0.0)
    assert coverage(front, front) == pytest.approx(1.0)
    scores = compare_fronts(front, front)
    assert scores["nd_count"] == len(front)
    assert scores["generational_distance"] == pytest.approx(0.0)
    assert scores["coverage"] == pytest.approx(1.0)


def test_load_front_npy_and_npz(tmp_path):
    front = np.array([[0.0, 1.0], [1.0, 0.0], [0.4, 0.4]])
    npy = tmp_path / "front.npy"
    np.save(npy, front)
    assert np.allclose(load_front(npy, n_obj=2), front)
    npz = tmp_path / "front.npz"
    np.savez_compressed(npz, front=front)
    assert np.allclose(load_front(npz, n_obj=2), front)
    only = tmp_path / "only.npz"
    np.savez(only, anything=front)
    assert np.allclose(load_front(only, n_obj=2), front)
    both = tmp_path / "both.npz"
    np.savez(both, a=front, b=front)
    with pytest.raises(ValueError, match="front"):
        load_front(both, n_obj=2)
    with pytest.raises(ValueError, match="shape"):
        load_front(npy, n_obj=3)
    flat = tmp_path / "flat.npy"
    np.save(flat, np.zeros(4))
    with pytest.raises(ValueError, match="shape"):
        load_front(flat, n_obj=2)
    with pytest.raises(ValueError, match=".npy or .npz"):
        load_front(tmp_path / "front.txt", n_obj=2)


@pytest.mark.thrml
def test_score_ising_loop_against_enumerated_front():
    pytest.importorskip("jax")
    pytest.importorskip("thrml")

    problem = CodonIsingProblem(n_spins=4, J=1.0, seed=1)
    loop = WeightSweepLoop(
        backend=problem.make_backend(),
        anneal=AnnealConfig(betas=(1.0, 4.0), n_warmup=1, n_samples=3, steps_per_sample=1),
        seed=0,
        weights=np.array([[0.5, 0.5], [1.0, 0.0]]),
        n_obj=2,
    )
    result = loop.run()
    _decisions, front = result.nondominated_front()
    _bits, exact = problem.enumerate_front()
    scores = compare_fronts(front, exact)
    assert scores["nd_count"] == len(front) >= 1
    assert scores["hypervolume_2d"] is not None
    assert scores["generational_distance"] >= 0.0
    assert 0.0 <= scores["coverage"] <= 1.0
