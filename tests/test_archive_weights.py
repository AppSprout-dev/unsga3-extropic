"""Archive and weight-vector tests (no THRML sampling)."""

from __future__ import annotations

import numpy as np
import pytest

from unsga3_extropic.archive import (
    Archive,
    hypervolume_2d,
    nondominated_mask,
    unique_rows,
)
from unsga3_extropic.weights import simplex_weights


def test_nondominated_2d():
    objs = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [0.6, 0.6]])
    mask = nondominated_mask(objs)
    assert mask.tolist() == [True, True, True, False]


def test_archive_unique():
    a = Archive()
    a.add(np.array([[0], [1], [0]]), np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]))
    x, f = a.nondominated()
    assert len(f) == 2


def test_simplex_weights_2d():
    w = simplex_weights(5, 2)
    assert w.shape == (5, 2)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert np.all(w >= 0)


def test_unique_rows():
    a = np.array([[1.0, 2.0], [1.0000001, 2.0], [3.0, 4.0]])
    u = unique_rows(a, decimals=5)
    assert len(u) == 2


def test_nondominated_3d_and_maximize():
    objs = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    assert nondominated_mask(objs).tolist() == [True, True, True, False]
    max_objs = np.array([[0.0, 0.0], [1.0, 1.0], [0.2, 0.2]])
    assert nondominated_mask(max_objs, minimize=False).tolist() == [
        False,
        True,
        False,
    ]


def test_archive_weight_ids_and_mismatch():
    archive = Archive()
    archive.add(
        np.array([[0.0], [1.0]]),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        weight_id=3,
    )
    _x, _f, ids = archive.as_arrays()
    assert ids.tolist() == [3, 3]
    with pytest.raises(ValueError, match="length mismatch"):
        archive.add(np.zeros((2, 1)), np.zeros((1, 2)))


def test_empty_archive_and_hypervolume():
    x, f = Archive().nondominated()
    assert len(x) == 0 and len(f) == 0
    assert hypervolume_2d(np.zeros((0, 2))) == 0.0
    front = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert hypervolume_2d(front, ref=(1.1, 1.1)) == pytest.approx(0.21)


def test_simplex_weights_more_objectives():
    single = simplex_weights(1, 2)
    assert single.shape == (1, 2)
    assert np.allclose(single, [[0.5, 0.5]])
    grid = simplex_weights(3, 2)
    assert np.allclose(grid, [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    w3 = simplex_weights(3, 3)
    assert w3.shape == (3, 3)
    assert np.allclose(w3.sum(axis=1), 1.0)
    assert np.all(w3 >= 0)
    with pytest.raises(ValueError):
        simplex_weights(0, 2)
    with pytest.raises(ValueError):
        simplex_weights(3, 1)
