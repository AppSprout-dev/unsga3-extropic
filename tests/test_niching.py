"""Reference-direction niching on pooled objectives (no THRML)."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from unsga3_extropic.archive import (
    Archive,
    associate_objectives,
    niche_survival_rows,
    nondominated_mask,
    normalize_objectives,
    perpendicular_distances,
)
from unsga3_extropic.backends.base import BackendResult
from unsga3_extropic.loop import AnnealConfig, WeightSweepLoop
from unsga3_extropic.results import build_measured_record
from unsga3_extropic.weights import simplex_weights

ROOT = Path(__file__).resolve().parents[1]

# Pooled minimization rows. A and B cluster on one reference direction.
# C is a worse scalarization and the only occupant of another direction,
# with a larger perpendicular distance than B. D anchors the opposite
# corner. S lies on C's ray but is dominated by C, so it must not take
# that niche even though its perpendicular distance is smaller.
CLUSTER = np.array(
    [
        [0.00, 1.00],
        [0.12, 0.97],
        [0.70, 0.42],
        [1.00, 0.00],
        [0.80, 0.80],
    ],
    dtype=np.float64,
)
DIRECTIONS = np.array(
    [
        [0.0, 1.0],
        [0.5, 0.5],
        [1.0, 0.0],
    ],
    dtype=np.float64,
)


def _spec_normalize(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ideal–nadir map written out beside the fixture, not via the library."""
    ideal = rows.min(axis=0)
    span = rows.max(axis=0) - ideal
    scale = np.where(span <= 1e-12, 1.0, span)
    return (rows - ideal) / scale, ideal, scale


def _spec_perp(points: np.ndarray, directions: np.ndarray) -> np.ndarray:
    unit = directions / np.linalg.norm(directions, axis=1)[:, None]
    proj_len = points @ unit.T
    residual = points[:, None, :] - proj_len[:, :, None] * unit[None, :, :]
    return np.linalg.norm(residual, axis=-1)


def _has_row(rows: np.ndarray, target: np.ndarray) -> bool:
    if len(rows) == 0:
        return False
    return bool(np.any(np.all(np.isclose(rows, target), axis=1)))


class ScriptedBackend:
    """Returns fixed objective batches. The only sampler method is sample_weight."""

    def __init__(self, batches: list[np.ndarray]) -> None:
        self.batches = batches
        self.seen: list[np.ndarray] = []

    def sample_weight(
        self,
        w: np.ndarray,
        *,
        seed: int,
        betas: tuple[float, ...],
        n_warmup: int,
        n_samples: int,
        steps_per_sample: int,
    ) -> BackendResult:
        batch = np.asarray(self.batches[len(self.seen)], dtype=np.float64)
        self.seen.append(np.asarray(w, dtype=np.float64).copy())
        row = np.arange(len(batch), dtype=np.float64)
        decisions = np.column_stack(
            [np.full(len(batch), float(len(self.seen) - 1)), row]
        )
        energies = batch @ np.asarray(w, dtype=np.float64)
        return BackendResult(
            decisions=decisions,
            objectives=batch,
            energies=energies,
            n_evals=len(batch),
        )


def test_normalization_and_perpendicular_distance_match_the_spec():
    mask = nondominated_mask(CLUSTER)
    assert mask.tolist() == [True, True, True, True, False]
    nd = CLUSTER[mask]
    spec_norm, spec_ideal, spec_scale = _spec_normalize(nd)
    got_norm, got_ideal, got_scale = normalize_objectives(nd)
    assert np.allclose(got_ideal, spec_ideal)
    assert np.allclose(got_scale, spec_scale)
    assert np.allclose(got_ideal, [0.0, 0.0])
    assert np.allclose(got_scale, [1.0, 1.0])
    assert np.allclose(got_norm, spec_norm)
    assert np.allclose(got_norm, nd)

    spec_dist = _spec_perp(spec_norm, DIRECTIONS)
    assert np.allclose(perpendicular_distances(got_norm, DIRECTIONS), spec_dist)
    # Axis point (0, 1) sits on the first ray and one unit off the last.
    assert spec_dist[0, 0] == pytest.approx(0.0)
    assert spec_dist[0, 1] == pytest.approx(np.sqrt(0.5))
    assert spec_dist[0, 2] == pytest.approx(1.0)

    associated = associate_objectives(CLUSTER, DIRECTIONS)
    assert np.allclose(associated.normalized, spec_norm)
    assert np.array_equal(associated.direction_index, spec_dist.argmin(axis=1))
    assert np.allclose(
        associated.distance, spec_dist[np.arange(len(nd)), associated.direction_index]
    )
    # Dominated row on the middle ray is not associated.
    assert not _has_row(associated.objectives, CLUSTER[4])


def test_lone_occupant_beats_a_crowded_duplicate():
    """ND keeps the cluster. Niching also keeps the empty direction's point."""
    archive = Archive()
    archive.add(np.arange(len(CLUSTER))[:, None], CLUSTER, weight_id=0)
    _x_nd, front = archive.nondominated()
    assert len(front) == 4
    assert _has_row(front, CLUSTER[0])
    assert _has_row(front, CLUSTER[1])
    assert _has_row(front, CLUSTER[2])
    assert not _has_row(front, CLUSTER[4])

    associated = archive.associate(DIRECTIONS)
    # A and B share a direction. C is alone on a different one, and farther
    # from its ray than B is from the crowded ray.
    assert associated.direction_index[0] == associated.direction_index[1]
    assert associated.direction_index[2] != associated.direction_index[0]
    assert np.count_nonzero(associated.direction_index == associated.direction_index[2]) == 1
    assert associated.distance[0] < associated.distance[1]
    assert associated.distance[2] > associated.distance[1]
    scalar = CLUSTER[:4] @ np.array([0.5, 0.5])
    assert scalar[2] > scalar[0]
    assert scalar[2] > scalar[1]

    niche = archive.niche_survival(DIRECTIONS, quota=1)
    assert _has_row(niche.objectives, CLUSTER[0])
    assert _has_row(niche.objectives, CLUSTER[2])
    assert _has_row(niche.objectives, CLUSTER[3])
    assert not _has_row(niche.objectives, CLUSTER[1])
    assert not _has_row(niche.objectives, CLUSTER[4])
    # Both sets stay available.
    _x_again, front_again = archive.nondominated()
    assert len(front_again) == 4
    assert len(niche.objectives) == 3


def test_closer_of_two_occupants_is_kept_at_quota_one():
    rows = np.array(
        [
            [0.00, 1.00],
            [0.20, 0.90],
            [1.00, 0.00],
        ],
        dtype=np.float64,
    )
    associated = associate_objectives(rows, DIRECTIONS)
    assert associated.direction_index[0] == associated.direction_index[1]
    assert associated.direction_index[0] != associated.direction_index[2]
    assert associated.distance[0] < associated.distance[1]
    assert associated.distance[0] == pytest.approx(0.0)
    assert associated.distance[1] == pytest.approx(0.2)

    niche = niche_survival_rows(rows, DIRECTIONS, quota=1)
    assert _has_row(niche.objectives, rows[0])
    assert not _has_row(niche.objectives, rows[1])
    assert _has_row(niche.objectives, rows[2])

    both = niche_survival_rows(CLUSTER, DIRECTIONS, quota=2)
    assert _has_row(both.objectives, CLUSTER[0])
    assert _has_row(both.objectives, CLUSTER[1])


def test_dominated_point_does_not_reenter_a_niche():
    # S is exactly on the middle ray (distance 0) and would beat C if
    # dominated rows were allowed to compete.
    on_ray = CLUSTER[4]
    assert np.allclose(on_ray, [0.8, 0.8])
    dist = _spec_perp(on_ray.reshape(1, 2), DIRECTIONS)
    assert dist[0, 1] == pytest.approx(0.0)
    niche = niche_survival_rows(CLUSTER, DIRECTIONS, quota=1)
    assert not _has_row(niche.objectives, on_ray)
    assert _has_row(niche.objectives, CLUSTER[2])


def test_constant_objective_scale_and_equal_distance_tie():
    repeated = np.array([[2.0, 3.0], [2.0, 3.0]], dtype=np.float64)
    associated = associate_objectives(repeated, np.eye(2))
    assert len(associated.objectives) == 1
    assert np.allclose(associated.scale, [1.0, 1.0])
    assert np.allclose(associated.normalized, [[0.0, 0.0]])

    tied = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.5],
        ],
        dtype=np.float64,
    )
    dirs = np.eye(2)
    associated = associate_objectives(tied, dirs)
    middle = np.flatnonzero(np.all(np.isclose(associated.objectives, [0.5, 0.5]), axis=1))
    assert associated.direction_index[middle[0]] == 0


def test_three_objectives_keep_one_axis_occupant_each():
    rows = np.eye(3)
    niche = niche_survival_rows(rows, np.eye(3), quota=1)
    assert len(niche.objectives) == 3
    assert np.allclose(np.sort(niche.direction_index), [0, 1, 2])


def test_loop_uses_caller_directions_and_only_sample_weight(monkeypatch):
    def _resampled(*_args, **_kwargs):
        raise AssertionError("simplex_weights was called")

    monkeypatch.setattr("unsga3_extropic.loop.simplex_weights", _resampled)
    weights = simplex_weights(3, 2)
    backend = ScriptedBackend([CLUSTER[:3], CLUSTER[3:4], CLUSTER[4:5]])
    loop = WeightSweepLoop(
        backend=backend,
        anneal=AnnealConfig(betas=(1.0,), n_warmup=0, n_samples=1, steps_per_sample=1),
        seed=7,
        weights=weights,
        n_obj=2,
    )
    result = loop.run()
    assert len(backend.seen) == 3
    assert np.allclose(np.stack(backend.seen), weights)
    assert np.allclose(result.weights, weights)

    source = inspect.getsource(WeightSweepLoop.run)
    assert "sample_weight" in source
    for banned in ("crossover", "sbx", "mutate", "polynomial"):
        assert banned not in source.lower()

    niche = result.niche_front()
    direct = result.archive.niche_survival(weights)
    assert np.allclose(niche.objectives, direct.objectives)
    # C was sampled under weight index 0 and still joins direction 1.
    came_from_first_weight = niche.decisions[:, 0] == 0.0
    c_rows = niche.decisions[came_from_first_weight]
    assert np.any(np.isclose(c_rows[:, 1], 2.0))
    c_slot = np.flatnonzero(
        (niche.decisions[:, 0] == 0.0) & np.isclose(niche.decisions[:, 1], 2.0)
    )
    assert niche.direction_index[c_slot[0]] == 1
    dropped = (niche.decisions[:, 0] == 0.0) & np.isclose(niche.decisions[:, 1], 1.0)
    assert not np.any(dropped)
    _x_nd, front = result.nondominated_front()
    assert len(front) == 4
    assert len(niche.objectives) == 3


def test_scaled_rays_match_and_a_new_simplex_would_not():
    rows = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    caller = np.array([[2.0, 0.0], [0.0, 5.0]], dtype=np.float64)
    associated = associate_objectives(rows, caller)
    assert associated.direction_index.tolist() == [1, 0]
    resampled = simplex_weights(2, 2)
    other = associate_objectives(rows, resampled)
    assert other.direction_index.tolist() == [0, 1]
    assert not np.array_equal(associated.direction_index, other.direction_index)


def test_default_simplex_is_built_once(monkeypatch):
    calls: list[tuple[int, int]] = []
    real = simplex_weights

    def _wrapped(n_weights: int, n_obj: int = 2) -> np.ndarray:
        calls.append((n_weights, n_obj))
        return real(n_weights, n_obj)

    monkeypatch.setattr("unsga3_extropic.loop.simplex_weights", _wrapped)
    backend = ScriptedBackend([CLUSTER[:2], CLUSTER[2:4]])
    loop = WeightSweepLoop(
        backend=backend,
        n_weights=2,
        n_obj=2,
        anneal=AnnealConfig(betas=(1.0,), n_warmup=0, n_samples=1, steps_per_sample=1),
        seed=1,
        weights=None,
    )
    result = loop.run()
    assert calls == [(2, 2)]
    assert np.allclose(result.weights, real(2, 2))
    assert len(result.niche_front().objectives) >= 1


def test_measured_record_appends_the_niche_count(tmp_path):
    weights = simplex_weights(3, 2)
    backend = ScriptedBackend([CLUSTER[:3], CLUSTER[3:4], CLUSTER[4:5]])
    result = WeightSweepLoop(
        backend=backend,
        anneal=AnnealConfig(betas=(1.0,), n_warmup=0, n_samples=1, steps_per_sample=1),
        seed=7,
        weights=weights,
    ).run()
    record = build_measured_record(
        result,
        anneal=AnnealConfig(betas=(1.0,), n_warmup=0, n_samples=1, steps_per_sample=1),
        base_seed=7,
        issue="5",
        phase="2",
        problem="fixture",
        backend="scripted",
        front_path=tmp_path / "front.npz",
        notes="pooled fixture",
    )
    assert record["metrics"]["nd_count"] == 4
    assert "Closer-in-niche survival kept 3 of 4 non-dominated rows" in record["notes"]
    assert "metrics.nd_count is the non-dominated archive" in record["notes"]


def test_readme_mapping_describes_niching():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    mapping = text.split("## Algorithm mapping", 1)[1].split("##", 1)[0]
    assert "perpendicular" in mapping
    assert "niche_survival" in mapping
    assert "nondominated" in mapping
    assert "sample_weight" in mapping
    assert "weight niches + offline ND" not in text
    assert "not implemented yet" not in mapping


def test_rejects_bad_quota_and_shape():
    with pytest.raises(ValueError, match="quota"):
        niche_survival_rows(CLUSTER[:2], DIRECTIONS, quota=0)
    with pytest.raises(ValueError, match="non-zero"):
        perpendicular_distances(CLUSTER[:1, :], np.zeros((1, 2)))
    with pytest.raises(ValueError, match="does not match"):
        associate_objectives(CLUSTER, np.ones((2, 3)))
