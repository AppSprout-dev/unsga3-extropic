"""Unit tests that do not require THRML sampling."""

from __future__ import annotations

import numpy as np

from unsga3_extropic.archive import Archive, nondominated_mask, unique_rows
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
