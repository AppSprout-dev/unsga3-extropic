"""Offline non-dominated archive (U-NSGA-III ranking surrogate)."""

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
