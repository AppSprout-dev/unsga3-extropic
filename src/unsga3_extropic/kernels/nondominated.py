"""Optional backends for the pairwise non-dominated filter.

The default ranking path in ``archive.nondominated_mask`` is a per-row
NumPy scan (``loop``). That scan is the ranking hot spot on the
weight-sweep archive: each of *n* rows launches an O(*n M*) comparison,
so cost grows as *n*^2. A broadcasted NumPy comparison (``tiled``) uses
the same arithmetic and does not beat ``loop`` at realistic archive
sizes, because it materializes large temporaries and cannot stop early
once a dominator is found.

``jit`` compiles the same pairwise test with Numba and exits the inner
loop as soon as any point dominates the candidate. That is the honest
CPU win. ``jit_parallel`` splits candidates across cores; it helps when
almost every point is non-dominated (no early exit) and costs extra
thread overhead on typical mixed archives. ``cuda`` is the same test as
a GPU kernel (Numba CUDA, or CuPy ``RawKernel`` using
``ND_MASK_CUDA_SOURCE``). It is skipped when no device is present.

Dominance matches ``archive.nondominated_mask`` exactly, including the
``1e-12`` slack and IEEE NaN/inf behavior of the NumPy comparisons
(a non-finite coordinate fails ``<=`` / ``<``, so a NaN row does not
dominate a finite row, and a finite row does not dominate a NaN row).
No backend is used unless the caller passes ``backend`` or sets
``UNSGA3_ND_BACKEND``. The ``loop`` backend calls the canonical scan in
``archive``; it does not keep a second copy of that scan.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np

BACKENDS = ("loop", "tiled", "jit", "jit_parallel", "cuda")
DEFAULT_BACKEND = "loop"
BACKEND_ENV = "UNSGA3_ND_BACKEND"
_EPS = 1e-12
_TILE = 256

ND_MASK_CUDA_SOURCE = r"""
extern "C" __global__
void nd_mask(const double* objs, unsigned char* keep, int n, int m, int minimize) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const double sign = minimize ? 1.0 : -1.0;
    const double* fi = objs + ((size_t)i * (size_t)m);
    for (int j = 0; j < n; ++j) {
        const double* fj = objs + ((size_t)j * (size_t)m);
        bool all_le = true;
        bool any_lt = false;
        for (int k = 0; k < m; ++k) {
            const double a = sign * fj[k];
            const double b = sign * fi[k];
            /* Match NumPy a <= b+eps. a > b+eps is wrong for NaN. */
            if (!(a <= b + 1e-12)) {
                all_le = false;
                break;
            }
            if (a < b - 1e-12) {
                any_lt = true;
            }
        }
        if (all_le && any_lt) {
            keep[i] = 0;
            return;
        }
    }
    keep[i] = 1;
}
"""

_serial_impl: Callable[..., np.ndarray] | None = None
_parallel_impl: Callable[..., np.ndarray] | None = None
_cuda_kernel = None
_cuda_kind: str | None = None


def resolve_backend(backend: str | None = None) -> str:
    """Return a known backend name. ``None`` reads ``UNSGA3_ND_BACKEND``."""
    if backend is None:
        raw = os.environ.get(BACKEND_ENV, DEFAULT_BACKEND)
    else:
        raw = backend
    name = str(raw).strip().lower()
    if not name:
        name = DEFAULT_BACKEND
    if name not in BACKENDS:
        known = ", ".join(BACKENDS)
        raise ValueError(f"unknown nondominated backend {name!r}; expected one of: {known}")
    return name


def jit_available() -> bool:
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def cuda_available() -> bool:
    try:
        from numba import cuda
        if cuda.is_available():
            return True
    except Exception:
        pass
    try:
        import cupy as cp
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def list_backends() -> dict[str, bool]:
    """Whether each backend can run in this process right now."""
    return {
        "loop": True,
        "tiled": True,
        "jit": jit_available(),
        "jit_parallel": jit_available(),
        "cuda": cuda_available(),
    }


def _require_2d(objs: np.ndarray) -> np.ndarray:
    rows = np.asarray(objs, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError(f"objs must be 2-D, got shape {rows.shape}")
    return np.ascontiguousarray(rows)


def nondominated_mask_loop(objs: np.ndarray, *, minimize: bool = True) -> np.ndarray:
    """Canonical per-row scan from ``archive._nondominated_mask_loop``.

    This wrapper only validates shape. It does not reimplement dominance,
    so the default path and ``backend='loop'`` cannot drift.
    """
    from unsga3_extropic.archive import _nondominated_mask_loop

    return _nondominated_mask_loop(_require_2d(objs), minimize=minimize)


def nondominated_mask_tiled(
    objs: np.ndarray, *, minimize: bool = True, tile: int = _TILE
) -> np.ndarray:
    """Chunked broadcast comparison. Same answers as ``loop``; not faster at scale."""
    rows = _require_2d(objs)
    n = len(rows)
    keep = np.ones(n, dtype=bool)
    if n == 0:
        return keep
    signed = rows if minimize else -rows
    width = max(int(tile), 1)
    for start in range(0, n, width):
        end = min(start + width, n)
        block = signed[start:end]
        less_equal = np.all(signed[None, :, :] <= block[:, None, :] + _EPS, axis=2)
        strictly = np.any(signed[None, :, :] < block[:, None, :] - _EPS, axis=2)
        keep[start:end] = ~np.any(less_equal & strictly, axis=1)
    return keep


def _numba_kernels():
    global _serial_impl, _parallel_impl
    if _serial_impl is not None and _parallel_impl is not None:
        return _serial_impl, _parallel_impl
    try:
        from numba import njit, prange
    except ImportError as exc:
        raise RuntimeError(
            "backend 'jit' requires numba; install the optional extra "
            "`pip install 'unsga3-extropic[kernels]'`"
        ) from exc

    @njit
    def _serial(objs, minimize):
        n, m = objs.shape
        keep = np.ones(n, dtype=np.bool_)
        sign = 1.0 if minimize else -1.0
        for i in range(n):
            dominated = False
            for j in range(n):
                all_le = True
                any_lt = False
                for k in range(m):
                    a = sign * objs[j, k]
                    b = sign * objs[i, k]
                    # Match NumPy ``a <= b + eps``. ``a > b + eps`` is not
                    # equivalent when a or b is NaN: NaN > x is False, so the
                    # old test left all_le True and a NaN row could dominate
                    # a finite row. ``not (a <= b + eps)`` is True for NaN.
                    if not (a <= b + _EPS):
                        all_le = False
                        break
                    if a < b - _EPS:
                        any_lt = True
                if all_le and any_lt:
                    dominated = True
                    break
            keep[i] = not dominated
        return keep

    @njit(parallel=True)
    def _parallel(objs, minimize):
        n, m = objs.shape
        keep = np.ones(n, dtype=np.bool_)
        sign = 1.0 if minimize else -1.0
        for i in prange(n):
            dominated = False
            for j in range(n):
                all_le = True
                any_lt = False
                for k in range(m):
                    a = sign * objs[j, k]
                    b = sign * objs[i, k]
                    # Match NumPy ``a <= b + eps``. ``a > b + eps`` is not
                    # equivalent when a or b is NaN: NaN > x is False, so the
                    # old test left all_le True and a NaN row could dominate
                    # a finite row. ``not (a <= b + eps)`` is True for NaN.
                    if not (a <= b + _EPS):
                        all_le = False
                        break
                    if a < b - _EPS:
                        any_lt = True
                if all_le and any_lt:
                    dominated = True
                    break
            keep[i] = not dominated
        return keep

    _serial_impl = _serial
    _parallel_impl = _parallel
    return _serial_impl, _parallel_impl


def nondominated_mask_jit(
    objs: np.ndarray, *, minimize: bool = True, parallel: bool = False
) -> np.ndarray:
    rows = _require_2d(objs)
    serial, par = _numba_kernels()
    impl = par if parallel else serial
    return np.asarray(impl(rows, bool(minimize)))


def _load_cuda_kernel():
    global _cuda_kernel, _cuda_kind
    if _cuda_kernel is not None:
        return _cuda_kernel, _cuda_kind
    try:
        from numba import cuda

        if cuda.is_available():
            @cuda.jit
            def _kernel(objs, keep, minimize):
                i = cuda.grid(1)
                n, m = objs.shape
                if i >= n:
                    return
                sign = 1.0 if minimize else -1.0
                for j in range(n):
                    all_le = True
                    any_lt = False
                    for k in range(m):
                        a = sign * objs[j, k]
                        b = sign * objs[i, k]
                        if not (a <= b + _EPS):
                            all_le = False
                            break
                        if a < b - _EPS:
                            any_lt = True
                    if all_le and any_lt:
                        keep[i] = 0
                        return
                keep[i] = 1

            _cuda_kernel = _kernel
            _cuda_kind = "numba"
            return _cuda_kernel, _cuda_kind
    except Exception:
        pass
    try:
        import cupy as cp

        if int(cp.cuda.runtime.getDeviceCount()) > 0:
            _cuda_kernel = cp.RawKernel(ND_MASK_CUDA_SOURCE, "nd_mask")
            _cuda_kind = "cupy"
            return _cuda_kernel, _cuda_kind
    except Exception:
        pass
    raise RuntimeError(
        "backend 'cuda' needs a CUDA device and either numba.cuda or cupy"
    )


def nondominated_mask_cuda(objs: np.ndarray, *, minimize: bool = True) -> np.ndarray:
    rows = _require_2d(objs)
    n = len(rows)
    if n == 0:
        return np.ones(0, dtype=bool)
    kernel, kind = _load_cuda_kernel()
    if kind == "numba":
        from numba import cuda

        threads = 128
        blocks = (n + threads - 1) // threads
        device_objs = cuda.to_device(rows)
        device_keep = cuda.device_array(n, dtype=np.uint8)
        kernel[blocks, threads](device_objs, device_keep, 1 if minimize else 0)
        return np.asarray(device_keep.copy_to_host(), dtype=bool)
    import cupy as cp

    threads = 128
    blocks = (n + threads - 1) // threads
    device_objs = cp.asarray(rows)
    device_keep = cp.empty(n, dtype=cp.uint8)
    kernel((blocks,), (threads,), (device_objs, device_keep, np.int32(n), np.int32(rows.shape[1]), np.int32(1 if minimize else 0)))
    return np.asarray(cp.asnumpy(device_keep), dtype=bool)


def nondominated_mask_backend(
    objs: np.ndarray, *, minimize: bool = True, backend: str | None = None
) -> np.ndarray:
    """Dispatch to a named backend. ``loop`` is the original scan."""
    chosen = resolve_backend(backend)
    if chosen == "loop":
        return nondominated_mask_loop(objs, minimize=minimize)
    if chosen == "tiled":
        return nondominated_mask_tiled(objs, minimize=minimize)
    if chosen == "jit":
        return nondominated_mask_jit(objs, minimize=minimize, parallel=False)
    if chosen == "jit_parallel":
        return nondominated_mask_jit(objs, minimize=minimize, parallel=True)
    return nondominated_mask_cuda(objs, minimize=minimize)
