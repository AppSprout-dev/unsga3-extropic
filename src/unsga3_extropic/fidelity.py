"""Compare minimization fronts inside this repository.

Inputs are ``ndarray`` fronts, not samplers. 2-D hypervolume reuses
``archive.hypervolume_2d``. Generational distance is the Van Veldhuizen
mean nearest-reference distance. Coverage is the fraction of one front
weakly dominated by the other.

``.npy`` and ``.npz`` loaders do not start another process. Bend, C#, and
ZDT1 stay outside the repo; a front they emit can be dropped in as a file.
The functions do not search over betas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from unsga3_extropic.archive import hypervolume_2d

_TOL = 1e-12


def generational_distance(
    front: np.ndarray,
    reference: np.ndarray,
    *,
    p: float = 2.0,
) -> float:
    """Mean distance from each row of ``front`` to the nearest reference row.

    ``(mean_i d_i**p) ** (1/p)`` with Euclidean ``d_i``. An empty ``front``
    returns ``0.0``. An empty ``reference`` raises.
    """
    got = np.asarray(front, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if got.ndim != 2 or ref.ndim != 2:
        raise ValueError(
            f"front and reference must be 2-D, got {got.shape} and {ref.shape}"
        )
    if got.shape[1] != ref.shape[1]:
        raise ValueError(
            f"objective counts differ: front {got.shape[1]} vs reference {ref.shape[1]}"
        )
    if p <= 0:
        raise ValueError("p must be > 0")
    if len(ref) == 0:
        raise ValueError("reference front is empty")
    if len(got) == 0:
        return 0.0
    diff = got[:, None, :] - ref[None, :, :]
    nearest = np.linalg.norm(diff, axis=-1).min(axis=1)
    return float(np.mean(nearest**p) ** (1.0 / p))


def coverage(
    front_a: np.ndarray,
    front_b: np.ndarray,
    *,
    minimize: bool = True,
) -> float:
    """Fraction of rows of ``front_b`` weakly dominated by some row of ``front_a``.

    Minimization: ``a`` weakly dominates ``b`` when ``a <= b`` on every
    objective. ``coverage(F, F) == 1`` for a non-empty finite front. An
    empty ``front_b`` returns ``1.0``.
    """
    a = np.asarray(front_a, dtype=np.float64)
    b = np.asarray(front_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"fronts must be 2-D, got {a.shape} and {b.shape}")
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"objective counts differ: {a.shape[1]} vs {b.shape[1]}"
        )
    if len(b) == 0:
        return 1.0
    if len(a) == 0:
        return 0.0
    if minimize:
        weak = np.all(a[:, None, :] <= b[None, :, :] + _TOL, axis=-1)
    else:
        weak = np.all(a[:, None, :] >= b[None, :, :] - _TOL, axis=-1)
    return float(weak.any(axis=0).mean())


def load_front(path: str | Path, *, n_obj: int) -> np.ndarray:
    """Load a minimization front of shape ``(n, n_obj)``.

    ``.npy`` is the array itself. ``.npz`` uses the array named ``front``,
    or the only array in the archive. Any other rank or objective count
    raises ``ValueError``.
    """
    if int(n_obj) < 1:
        raise ValueError("n_obj must be >= 1")
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(file_path)
    elif suffix == ".npz":
        with np.load(file_path) as archive:
            names = list(archive.files)
            if "front" in names:
                arr = np.array(archive["front"])
            elif len(names) == 1:
                arr = np.array(archive[names[0]])
            else:
                raise ValueError(
                    "npz front must contain an array named 'front' or a single "
                    f"array; found {names}"
                )
    else:
        raise ValueError(f"front file must be .npy or .npz, got {suffix!r}")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or int(arr.shape[1]) != int(n_obj):
        raise ValueError(f"front shape {arr.shape} does not match (n, {n_obj})")
    return arr


def compare_fronts(
    obtained: np.ndarray,
    reference: np.ndarray | None = None,
) -> dict:
    """Score a minimization front.

    ``hypervolume_2d`` is set when the front has two objectives, using a
    reference point just beyond the max of the obtained and reference rows.
    ``generational_distance`` and ``coverage`` are set when both fronts are
    non-empty. ``coverage`` is the fraction of ``reference`` weakly dominated
    by ``obtained``.
    """
    got = np.asarray(obtained, dtype=np.float64)
    if got.ndim != 2:
        raise ValueError(f"obtained front must be 2-D, got {got.shape}")
    out: dict = {
        "nd_count": int(len(got)),
        "hypervolume_2d": None,
        "hv_ref": None,
        "generational_distance": None,
        "coverage": None,
    }
    ref = None if reference is None else np.asarray(reference, dtype=np.float64)
    if got.shape[1] == 2 and len(got):
        span = got
        if ref is not None and ref.ndim == 2 and ref.shape[1] == 2 and len(ref):
            span = np.vstack([got, ref])
        hv_ref = (
            float(span[:, 0].max() + 0.1 * max(abs(float(span[:, 0].max())), 1.0)),
            float(span[:, 1].max() + 0.1 * max(abs(float(span[:, 1].max())), 1.0)),
        )
        out["hypervolume_2d"] = hypervolume_2d(got, ref=hv_ref)
        out["hv_ref"] = [hv_ref[0], hv_ref[1]]
    elif got.shape[1] == 2 and len(got) == 0:
        out["hypervolume_2d"] = 0.0
    if ref is not None and ref.ndim == 2 and len(ref) and len(got):
        out["generational_distance"] = generational_distance(got, ref)
        out["coverage"] = coverage(got, ref)
    return out
