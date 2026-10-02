"""Offline non-dominated archive and reference-direction niching.

Ranking is the non-dominated filter. Niche survival is a second set on
those rows: ideal–nadir normalization, then perpendicular distance to the
caller-supplied reference directions. Search does not happen here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def nondominated_mask(objs: np.ndarray, *, minimize: bool = True) -> np.ndarray:
    """Boolean mask of non-dominated rows (Pareto front of the finite set)."""
    objs = np.asarray(objs, dtype=np.float64)
    if objs.ndim != 2:
        raise ValueError(f"objs must be 2-D, got shape {objs.shape}")
    n = len(objs)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        if minimize:
            dom = np.all(objs <= objs[i] + 1e-12, axis=1) & np.any(
                objs < objs[i] - 1e-12, axis=1
            )
        else:
            dom = np.all(objs >= objs[i] - 1e-12, axis=1) & np.any(
                objs > objs[i] + 1e-12, axis=1
            )
        if np.any(dom):
            keep[i] = False
    return keep


def unique_rows(a: np.ndarray, decimals: int = 6) -> np.ndarray:
    """Drop near-duplicate rows (rounded uniqueness, original values kept)."""
    a = np.asarray(a)
    if len(a) == 0:
        return a
    r = np.round(a, decimals=decimals)
    _, idx = np.unique(r, axis=0, return_index=True)
    return a[np.sort(idx)]


@dataclass
class Archive:
    """Accumulates decision vectors + objective vectors; exposes ND front."""

    decisions: list[np.ndarray] = field(default_factory=list)
    objectives: list[np.ndarray] = field(default_factory=list)
    weight_ids: list[int] = field(default_factory=list)

    def add(
        self,
        decisions: np.ndarray,
        objectives: np.ndarray,
        *,
        weight_id: int = -1,
    ) -> None:
        decisions = np.asarray(decisions)
        objectives = np.asarray(objectives, dtype=np.float64)
        if len(decisions) != len(objectives):
            raise ValueError("decisions and objectives length mismatch")
        for d, o in zip(decisions, objectives):
            self.decisions.append(np.asarray(d))
            self.objectives.append(np.asarray(o, dtype=np.float64))
            self.weight_ids.append(weight_id)

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.objectives:
            return (
                np.zeros((0, 0)),
                np.zeros((0, 0)),
                np.zeros((0,), dtype=int),
            )
        return (
            np.stack(self.decisions, axis=0),
            np.stack(self.objectives, axis=0),
            np.asarray(self.weight_ids, dtype=int),
        )

    def nondominated(
        self, *, decimals: int = 6, minimize: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        x, f, _ = self.as_arrays()
        if len(f) == 0:
            return x, f
        mask = nondominated_mask(f, minimize=minimize)
        x_nd, f_nd = x[mask], f[mask]
        # unique on objectives
        f_u = unique_rows(f_nd, decimals=decimals)
        # keep one decision per unique objective row
        kept_x = []
        kept_f = []
        seen = set()
        for xi, fi in zip(x_nd, f_nd):
            key = tuple(np.round(fi, decimals=decimals).tolist())
            if key in seen:
                continue
            seen.add(key)
            kept_x.append(xi)
            kept_f.append(fi)
        return np.asarray(kept_x), np.asarray(kept_f, dtype=np.float64)

    def associate(
        self,
        directions: np.ndarray,
        *,
        decimals: int = 6,
        minimize: bool = True,
    ) -> "AssociatedRows":
        """Associate unique non-dominated rows to ``directions``.

        Dominated rows are dropped before normalization, so they do not
        set the ideal or the nadir and they do not occupy a direction.
        """
        decisions, objectives, _ids = self.as_arrays()
        return associate_objectives(
            objectives,
            directions,
            decisions=decisions,
            decimals=decimals,
            minimize=minimize,
        )

    def niche_survival(
        self,
        directions: np.ndarray,
        *,
        quota: int = 1,
        decimals: int = 6,
        minimize: bool = True,
    ) -> "NicheResult":
        """Closer occupants of ``directions``. Does not replace ``nondominated``."""
        decisions, objectives, _ids = self.as_arrays()
        return niche_survival_rows(
            objectives,
            directions,
            decisions=decisions,
            quota=quota,
            decimals=decimals,
            minimize=minimize,
        )


_SCALE_EPS = 1e-12


def normalize_objectives(
    objectives: np.ndarray,
    *,
    minimize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ideal–nadir normalization of a minimization front.

    On the rows passed in (already the non-dominated set when called from
    ``associate_objectives``):

    - ``ideal_j = min_i f_{i,j}`` and ``nadir_j = max_i f_{i,j}``
    - ``scale_j = nadir_j - ideal_j``
    - if ``scale_j <= 1e-12`` the objective is constant on this set; that
      entry is replaced by ``1`` so the translated coordinate is unchanged
      and a zero span does not explode
    - ``f_norm = (f - ideal) / scale``

    The normalized ideal is the origin. Reference directions are rays from
    that origin. Maximization negates the rows first; the returned ideal
    and scale are in that minimization frame.

    An empty input returns empty normalized rows and zero ideal/scale
    vectors with one entry per objective. A 1-D array raises ``ValueError``.
    """
    rows = np.asarray(objectives, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError(f"objectives must be 2-D, got shape {rows.shape}")
    n_obj = rows.shape[1]
    if n_obj < 1:
        raise ValueError("objectives must have at least one column")
    if len(rows) == 0:
        zeros = np.zeros(n_obj, dtype=np.float64)
        return np.zeros((0, n_obj), dtype=np.float64), zeros, zeros.copy()
    signed = rows if minimize else -rows
    ideal = signed.min(axis=0)
    span = signed.max(axis=0) - ideal
    scale = np.where(span <= _SCALE_EPS, 1.0, span)
    return (signed - ideal) / scale, ideal, scale


def perpendicular_distances(
    points: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    """Perpendicular distance from each point to each direction ray.

    ``d(s, w) = || s - (s · u) u ||`` with ``u = w / ||w||``. ``points``
    has shape ``(n, M)`` and ``directions`` has shape ``(H, M)``. The
    result has shape ``(n, H)``. A zero-length direction raises
    ``ValueError``. Distance is unchanged if a direction is scaled, so a
    simplex weight and a positive multiple of that weight are the same ray.
    """
    pts = np.asarray(points, dtype=np.float64)
    dirs = np.asarray(directions, dtype=np.float64)
    if pts.ndim != 2 or dirs.ndim != 2:
        raise ValueError(
            f"points and directions must be 2-D, got {pts.shape} and {dirs.shape}"
        )
    if pts.shape[1] != dirs.shape[1]:
        raise ValueError(
            f"objective count {pts.shape[1]} does not match directions {dirs.shape[1]}"
        )
    if len(dirs) == 0:
        raise ValueError("directions must contain at least one ray")
    norms = np.linalg.norm(dirs, axis=1)
    if np.any(norms <= _SCALE_EPS):
        raise ValueError("reference directions must be non-zero")
    unit = dirs / norms[:, None]
    proj_len = pts @ unit.T
    residual = pts[:, None, :] - proj_len[:, :, None] * unit[None, :, :]
    return np.linalg.norm(residual, axis=-1)


@dataclass(frozen=True)
class AssociatedRows:
    """Unique non-dominated rows and their nearest reference direction.

    ``direction_index`` is the argmin of perpendicular distance. Equal
    distances take the smallest direction index. Row order matches
    ``Archive.nondominated``.
    """

    decisions: np.ndarray
    objectives: np.ndarray
    direction_index: np.ndarray
    distance: np.ndarray
    normalized: np.ndarray
    ideal: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class NicheResult:
    """Quota-limited survivors, one closer occupant per filled direction.

    Ordered by direction index, then perpendicular distance, then the
    original non-dominated row order. This is not the non-dominated set.
    """

    decisions: np.ndarray
    objectives: np.ndarray
    direction_index: np.ndarray
    distance: np.ndarray


def _empty_association(n_var: int, n_obj: int) -> AssociatedRows:
    return AssociatedRows(
        decisions=np.zeros((0, n_var), dtype=np.float64),
        objectives=np.zeros((0, n_obj), dtype=np.float64),
        direction_index=np.zeros(0, dtype=int),
        distance=np.zeros(0, dtype=np.float64),
        normalized=np.zeros((0, n_obj), dtype=np.float64),
        ideal=np.zeros(n_obj, dtype=np.float64),
        scale=np.zeros(n_obj, dtype=np.float64),
    )


def _unique_nondominated(
    decisions: np.ndarray,
    objectives: np.ndarray,
    *,
    decimals: int,
    minimize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Rows ``Archive.nondominated`` keeps, in the same order."""
    mask = nondominated_mask(objectives, minimize=minimize)
    x_nd = decisions[mask]
    f_nd = objectives[mask]
    kept_x: list[np.ndarray] = []
    kept_f: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for xi, fi in zip(x_nd, f_nd):
        key = tuple(np.round(fi, decimals=decimals).tolist())
        if key in seen:
            continue
        seen.add(key)
        kept_x.append(np.asarray(xi))
        kept_f.append(np.asarray(fi, dtype=np.float64))
    n_var = decisions.shape[1]
    n_obj = objectives.shape[1]
    if not kept_f:
        return (
            np.zeros((0, n_var), dtype=np.float64),
            np.zeros((0, n_obj), dtype=np.float64),
        )
    return np.stack(kept_x, axis=0), np.stack(kept_f, axis=0)


def _check_quota(quota: int) -> int:
    if isinstance(quota, bool) or not isinstance(quota, (int, np.integer)):
        raise ValueError("quota must be an integer >= 1")
    quota_int = int(quota)
    if quota_int < 1:
        raise ValueError("quota must be an integer >= 1")
    return quota_int


def associate_objectives(
    objectives: np.ndarray,
    directions: np.ndarray,
    *,
    decisions: np.ndarray | None = None,
    decimals: int = 6,
    minimize: bool = True,
) -> AssociatedRows:
    """Associate pooled objective rows to reference directions.

    Rows are the pooled sample (every weight's candidates together).
    Dominated rows are removed first. The remaining rows are ideal–nadir
    normalized (``normalize_objectives``) and each is assigned to the
    direction with the smallest perpendicular distance
    (``perpendicular_distances``). ``directions`` are used as given.
    They are not replaced by a new simplex sample. A positive scale factor
    on a direction does not change the ray.

    ``decisions`` aligns with ``objectives``. When it is omitted, the
    decision column is the original row index.
    """
    objs = np.asarray(objectives, dtype=np.float64)
    dirs = np.asarray(directions, dtype=np.float64)
    if objs.ndim != 2:
        raise ValueError(f"objectives must be 2-D, got shape {objs.shape}")
    if dirs.ndim != 2 or dirs.shape[0] < 1:
        raise ValueError(f"directions must be 2-D and non-empty, got shape {dirs.shape}")
    if len(objs) == 0:
        n_var = 0
        if decisions is not None:
            decisions_arr = np.asarray(decisions)
            if decisions_arr.ndim != 2:
                raise ValueError(
                    f"decisions must be 2-D, got shape {decisions_arr.shape}"
                )
            n_var = decisions_arr.shape[1]
        return _empty_association(n_var, int(dirs.shape[1]))
    if objs.shape[1] != dirs.shape[1]:
        raise ValueError(
            f"objective count {objs.shape[1]} does not match directions {dirs.shape[1]}"
        )
    if decisions is None:
        decisions_arr = np.arange(len(objs), dtype=np.float64)[:, None]
    else:
        decisions_arr = np.asarray(decisions)
        if decisions_arr.ndim != 2 or len(decisions_arr) != len(objs):
            raise ValueError(
                "decisions must have shape (n, n_var) aligned with objectives, "
                f"got {decisions_arr.shape} vs {objs.shape}"
            )
    kept_x, kept_f = _unique_nondominated(
        decisions_arr, objs, decimals=decimals, minimize=minimize
    )
    n_obj = objs.shape[1]
    if len(kept_f) == 0:
        return _empty_association(kept_x.shape[1], n_obj)
    normalized, ideal, scale = normalize_objectives(kept_f, minimize=minimize)
    dists = perpendicular_distances(normalized, dirs)
    direction_index = np.argmin(dists, axis=1).astype(int)
    distance = dists[np.arange(len(kept_f)), direction_index]
    return AssociatedRows(
        decisions=kept_x,
        objectives=kept_f,
        direction_index=direction_index,
        distance=distance,
        normalized=normalized,
        ideal=ideal,
        scale=scale,
    )


def niche_survival_rows(
    objectives: np.ndarray,
    directions: np.ndarray,
    *,
    decisions: np.ndarray | None = None,
    quota: int = 1,
    decimals: int = 6,
    minimize: bool = True,
) -> NicheResult:
    """Keep closer occupants, preferring an empty direction over a duplicate.

    Association is ``associate_objectives``. Seats are then filled one
    at a time. Among directions that still have an unselected associate and
    have taken fewer than ``quota`` rows, the next seat goes to the
    direction with the fewest selected occupants (ties: smaller direction
    index). The row taken from that direction is the closest remaining one
    (ties: earlier non-dominated order).

    With ``quota == 1`` each occupied direction keeps its closest row and
    drops the rest. A lone occupant of an otherwise empty direction is kept
    even when its perpendicular distance is larger than that of a dropped
    duplicate in a crowded direction. Dominated rows never fill a seat.
    """
    quota_int = _check_quota(quota)
    associated = associate_objectives(
        objectives,
        directions,
        decisions=decisions,
        decimals=decimals,
        minimize=minimize,
    )
    n = len(associated.objectives)
    if n == 0:
        return NicheResult(
            decisions=associated.decisions,
            objectives=associated.objectives,
            direction_index=associated.direction_index,
            distance=associated.distance,
        )
    groups: dict[int, list[int]] = {}
    for row_index, direction in enumerate(associated.direction_index):
        groups.setdefault(int(direction), []).append(row_index)
    for members in groups.values():
        members.sort(
            key=lambda row_index: (float(associated.distance[row_index]), row_index)
        )
    taken = {direction: 0 for direction in groups}
    selected: list[int] = []
    while True:
        eligible = [
            direction
            for direction, members in groups.items()
            if taken[direction] < quota_int and taken[direction] < len(members)
        ]
        if not eligible:
            break
        direction = min(eligible, key=lambda item: (taken[item], item))
        selected.append(groups[direction][taken[direction]])
        taken[direction] += 1
    order = sorted(
        selected,
        key=lambda row_index: (
            int(associated.direction_index[row_index]),
            float(associated.distance[row_index]),
            row_index,
        ),
    )
    idx = np.asarray(order, dtype=int)
    return NicheResult(
        decisions=associated.decisions[idx],
        objectives=associated.objectives[idx],
        direction_index=associated.direction_index[idx],
        distance=associated.distance[idx],
    )


def hypervolume_2d(
    front: np.ndarray, ref: tuple[float, float] = (1.1, 1.1)
) -> float:
    """2-D hypervolume for minimization vs a dominated reference point."""
    if len(front) == 0:
        return 0.0
    pts = unique_rows(front[nondominated_mask(front)])
    pts = pts[(pts[:, 0] < ref[0]) & (pts[:, 1] < ref[1])]
    if len(pts) == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    prev_f1 = ref[0]
    for f1, f2 in pts[::-1]:
        hv += (prev_f1 - f1) * (ref[1] - f2)
        prev_f1 = f1
    return float(hv)
