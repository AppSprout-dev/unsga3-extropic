"""Optional nondominated backends match the default loop exactly."""

from __future__ import annotations

import numpy as np
import pytest

from unsga3_extropic.archive import nondominated_mask
from unsga3_extropic.kernels.nondominated import (
    BACKENDS,
    cuda_available,
    jit_available,
    list_backends,
    nondominated_mask_backend,
    resolve_backend,
)


def _cases():
    rng = np.random.default_rng(7)
    front_t = rng.random(80)
    return [
        np.zeros((0, 2)),
        np.array([[1.0, 2.0]]),
        np.ones((5, 2)),
        np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [0.6, 0.6]]),
        np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]),
        rng.random((64, 2)),
        rng.random((200, 3)),
        np.column_stack([front_t, 1.0 - np.sqrt(front_t)]),
        rng.random((1, 4)),
    ]


ALWAYS = ("loop", "tiled")


def _optional():
    extra = []
    if jit_available():
        extra.extend(["jit", "jit_parallel"])
    if cuda_available():
        extra.append("cuda")
    return extra


@pytest.mark.parametrize("backend", ALWAYS)
@pytest.mark.parametrize("minimize", [True, False])
def test_always_available_matches_default(backend, minimize):
    for objs in _cases():
        expected = nondominated_mask(objs, minimize=minimize, backend="loop")
        got = nondominated_mask(objs, minimize=minimize, backend=backend)
        assert got.dtype == bool
        assert got.shape == (len(objs),)
        assert np.array_equal(got, expected)


@pytest.mark.skipif(not jit_available(), reason="numba not installed")
@pytest.mark.parametrize("backend", ["jit", "jit_parallel"])
@pytest.mark.parametrize("minimize", [True, False])
def test_jit_matches_default(backend, minimize):
    rng = np.random.default_rng(11)
    cases = _cases() + [rng.random((400, 2)), rng.random((300, 3))]
    for objs in cases:
        expected = nondominated_mask(objs, minimize=minimize, backend="loop")
        got = nondominated_mask(objs, minimize=minimize, backend=backend)
        assert np.array_equal(got, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not cuda_available(), reason="no CUDA device")
@pytest.mark.parametrize("minimize", [True, False])
def test_cuda_matches_default(minimize):
    rng = np.random.default_rng(19)
    cases = _cases() + [rng.random((512, 2)), rng.random((256, 3))]
    for objs in cases:
        expected = nondominated_mask(objs, minimize=minimize, backend="loop")
        got = nondominated_mask(objs, minimize=minimize, backend="cuda")
        assert np.array_equal(got, expected)


def test_default_path_is_loop(monkeypatch):
    monkeypatch.delenv("UNSGA3_ND_BACKEND", raising=False)
    objs = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [0.6, 0.6]])
    assert np.array_equal(nondominated_mask(objs), nondominated_mask(objs, backend="loop"))
    assert resolve_backend(None) == "loop"


def test_env_backend(monkeypatch):
    objs = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [0.6, 0.6]])
    monkeypatch.setenv("UNSGA3_ND_BACKEND", "tiled")
    assert np.array_equal(nondominated_mask(objs), nondominated_mask(objs, backend="tiled"))
    monkeypatch.setenv("UNSGA3_ND_BACKEND", "loop")
    assert np.array_equal(nondominated_mask(objs), nondominated_mask(objs, backend="loop"))


def test_explicit_backend_overrides_env(monkeypatch):
    objs = np.array([[0.0, 1.0], [0.6, 0.6], [1.0, 0.0]])
    monkeypatch.setenv("UNSGA3_ND_BACKEND", "tiled")
    got = nondominated_mask(objs, backend="loop")
    assert np.array_equal(got, nondominated_mask(objs, backend="loop"))


def test_unknown_backend():
    with pytest.raises(ValueError, match="unknown nondominated backend"):
        nondominated_mask(np.zeros((2, 2)), backend="not-a-backend")


def test_rejects_1d():
    with pytest.raises(ValueError, match="2-D"):
        nondominated_mask(np.array([1.0, 2.0]), backend="loop")
    with pytest.raises(ValueError, match="2-D"):
        nondominated_mask_backend(np.array([1.0, 2.0]), backend="tiled")


def test_list_backends_includes_all_names():
    available = list_backends()
    assert tuple(available) == BACKENDS
    assert available["loop"] is True
    assert available["tiled"] is True
    assert available["cuda"] is cuda_available()
    assert available["jit"] is jit_available()

def _nonfinite_cases():
    """Clouds where IEEE NaN/inf comparisons decide dominance."""
    return [
        np.array([[np.nan, 0.1], [1.0, 1.0]]),
        np.array([[0.0, 0.0], [np.nan, 1.0]]),
        np.array([[np.nan, np.nan], [np.nan, 0.0], [0.0, 0.0]]),
        np.array([[0.0, 1.0], [np.nan, 0.0], [1.0, 0.0], [0.6, 0.6]]),
        np.array([[0.0, 1.0], [np.inf, 0.0], [1.0, 0.0]]),
        np.array([[np.inf, np.inf], [0.0, 1.0], [1.0, 0.0]]),
        np.array([[-np.inf, 1.0], [0.0, 0.0], [1.0, -np.inf]]),
        np.array([[np.nan, np.inf], [0.5, 0.5], [np.inf, np.nan], [0.0, 1.0]]),
        np.array([[0.0, 0.0, np.nan], [1.0, 1.0, 1.0], [0.1, 0.1, 0.1]]),
    ]


def test_nonfinite_matches_loop():
    """Every available backend matches the loop on NaN and inf rows.

    The default scan uses NumPy ``<=`` / ``<``. A NaN coordinate fails
    those tests, so a NaN row does not dominate a finite row and a
    finite row does not dominate a NaN row. Compiled backends used to
    treat ``not (a > b + eps)`` as ``a <= b + eps``, which is false for
    NaN and let a NaN row dominate.
    """
    backends = ["loop", "tiled", *_optional()]
    # The reported bug: NaN row must not dominate the finite row.
    reported = np.array([[np.nan, 0.1], [1.0, 1.0]])
    assert nondominated_mask(reported, backend="loop").tolist() == [True, True]
    for objs in _nonfinite_cases():
        for minimize in (True, False):
            expected = nondominated_mask(objs, minimize=minimize, backend="loop")
            for backend in backends:
                got = nondominated_mask(objs, minimize=minimize, backend=backend)
                assert np.array_equal(got, expected), (
                    backend,
                    minimize,
                    objs,
                    expected,
                    got,
                )


def test_loop_backend_is_archive_scan():
    """``backend='loop'`` must call the archive scan, not a second copy."""
    import inspect

    from unsga3_extropic.kernels.nondominated import nondominated_mask_loop

    source = inspect.getsource(nondominated_mask_loop)
    assert "_nondominated_mask_loop" in source
    assert "np.all" not in source
