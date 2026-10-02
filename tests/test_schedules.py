"""Deep profile budgets stay larger than the CI smoke. No sampling."""

from __future__ import annotations

import pytest

from unsga3_extropic.problems.codon_ising import CodonIsingProblem
from unsga3_extropic.problems.potts_chain import PottsChainProblem
from unsga3_extropic.schedules import (
    CODON_INSTANCE_LABEL,
    DEEP_SCHEDULE,
    TORX_DEEP_N_SAMPLES,
    TORX_DEEP_P_SWAP,
    TORX_DEEP_SEEDS,
    codon_problem_kwargs,
    potts_instance,
    potts_instance_label,
    profile_clause,
    require_base_seed,
    resolve_profile,
    schedule_for,
)
from unsga3_extropic.torx_circuit import DOCUMENTED_N_SAMPLES, DOCUMENTED_P_SWAP


def test_deep_budget_is_far_above_smoke():
    deep = schedule_for("codon", "deep")
    codon_smoke = schedule_for("codon", "smoke")
    potts_smoke = schedule_for("potts", "smoke")
    wall_smoke = schedule_for("domain_wall", "smoke")
    assert deep is DEEP_SCHEDULE
    assert schedule_for("potts", "deep") is DEEP_SCHEDULE
    assert schedule_for("domain_wall", "deep") is DEEP_SCHEDULE
    assert deep.recorded_per_seed() == 11 * 6 * 24
    assert deep.recorded_per_seed() >= 50 * codon_smoke.recorded_per_seed()
    assert deep.recorded_total() >= 150 * potts_smoke.recorded_per_seed()
    assert codon_smoke.recorded_per_seed() == 24
    assert potts_smoke.recorded_per_seed() == 16
    assert wall_smoke.recorded_per_seed() == 16
    assert wall_smoke.base_seeds == (11,)
    assert codon_smoke.base_seeds == (7,)
    assert len(deep.base_seeds) >= 3
    assert len(set(deep.base_seeds)) == len(deep.base_seeds)
    assert deep.n_weights() > codon_smoke.n_weights()
    assert len(deep.betas) > len(codon_smoke.betas)
    assert min(deep.betas) < min(codon_smoke.betas)
    assert max(deep.betas) > max(codon_smoke.betas)
    assert deep.n_warmup > codon_smoke.n_warmup
    assert deep.n_samples > codon_smoke.n_samples
    assert deep.steps_per_sample > codon_smoke.steps_per_sample
    assert deep.n_samples > potts_smoke.n_samples
    assert deep.n_warmup > potts_smoke.n_warmup


def test_deep_potts_instance_matches_smoke_not_default():
    assert potts_instance("deep") == potts_instance("smoke") == (6, 3)
    assert potts_instance("default") == (8, 3)
    assert potts_instance_label("deep") == "n_sites=6, K=3"
    problem = PottsChainProblem(*potts_instance("deep"))
    assert problem.n_sites == 6
    codon = CodonIsingProblem(**codon_problem_kwargs())
    assert codon.n_spins == 12
    assert CODON_INSTANCE_LABEL.startswith("n_spins=12")


def test_deep_weights_lie_on_the_simplex_and_include_endpoints():
    weights = DEEP_SCHEDULE.weight_matrix()
    assert weights.shape == (11, 2)
    assert abs(float(weights.sum(axis=1).min()) - 1.0) < 1e-12
    assert abs(float(weights.sum(axis=1).max()) - 1.0) < 1e-12
    assert (0.0, 1.0) in DEEP_SCHEDULE.weights or _close_pair(weights, 0.0, 1.0)
    assert _close_pair(weights, 1.0, 0.0)
    assert "simplex_weights(11, 2)" in DEEP_SCHEDULE.weight_source


def _close_pair(weights, first: float, second: float) -> bool:
    return any(
        abs(float(row[0]) - first) < 1e-12 and abs(float(row[1]) - second) < 1e-12
        for row in weights
    )


def test_profile_clause_marks_deep_replicate():
    schedule = schedule_for("domain_wall", "deep")
    text = profile_clause(schedule, 11)
    assert text.startswith("profile=deep.")
    assert "base_seed=11." in text
    assert "replicate=2/3." in text
    assert "schedule_product=1584." in text
    smoke = profile_clause(schedule_for("codon", "smoke"), 7)
    assert smoke.startswith("profile=smoke.")
    assert "replicate=1/1." in smoke


def test_require_base_seed_rejects_a_partial_deep_call():
    schedule = schedule_for("potts", "deep")
    with pytest.raises(ValueError, match="base_seed"):
        require_base_seed(schedule, None)
    assert require_base_seed(schedule_for("potts", "smoke"), None) == 7
    with pytest.raises(ValueError, match="not in"):
        require_base_seed(schedule, 3)


def test_resolve_profile_keeps_smoke_and_default_flags():
    assert resolve_profile(smoke=True, profile=None) == "smoke"
    assert resolve_profile(smoke=False, profile=None) == "default"
    assert resolve_profile(smoke=False, profile="deep") == "deep"
    with pytest.raises(ValueError, match="smoke=True"):
        resolve_profile(smoke=True, profile="deep")
    with pytest.raises(AssertionError):
        resolve_profile(smoke=False, profile="wider")


def test_unknown_problem_is_rejected():
    with pytest.raises(AssertionError):
        schedule_for("zdt1", "deep")


def test_torx_deep_rate_check_is_larger_than_smoke():
    assert TORX_DEEP_N_SAMPLES >= 5 * DOCUMENTED_N_SAMPLES
    assert TORX_DEEP_P_SWAP == DOCUMENTED_P_SWAP
    assert len(TORX_DEEP_SEEDS) >= 3
    assert 0 in TORX_DEEP_SEEDS
