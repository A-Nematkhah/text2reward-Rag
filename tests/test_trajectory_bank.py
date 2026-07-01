"""Trajectory bank metric aggregation and lite/full Stage B banks."""

from collections import Counter

from txt2reward.config.validation import BANK_MAX_VIOLATION_RATE, BANK_MIN_FITNESS_GAP
from txt2reward.llm.prompts import DEFAULT_BOOTSTRAP_REWARD_BODY
from txt2reward.sandbox.sandbox import compile_reward_function
from txt2reward.trajectory.bank import (
    _aggregate_trajectory_metrics,
    build_trajectory_bank,
    build_trajectory_bank_lite,
    evaluate_consistency,
    get_trajectory_bank,
    measure_gate_stats,
)

_LITE_CATEGORIES = (
    "safe_steady",
    "safe_fast",
    "stationary_farming",
    "reckless_crash",
    "tailgating_no_crash",
    "oscillating_lanes",
    "jerk_accel_spam",
    "legitimate_overtaking",
)


def test_trajectory_metrics_include_robust_ttc():
    states = [
        {"speed_ms": 20.0, "collided": False, "overtook": False, "long_jerk": 0.0, "ttc": 30.0},
        {"speed_ms": 20.0, "collided": False, "overtook": False, "long_jerk": 0.0, "ttc": 1.0},
    ]
    m = _aggregate_trajectory_metrics(states)
    assert m["min_ttc"] == 1.0
    assert "p10_ttc" in m


def test_lite_bank_has_sixteen_trajectories_and_all_categories():
    bank = build_trajectory_bank_lite()
    assert len(bank) == 16
    cats = Counter(s.category for s in bank)
    assert set(cats) == set(_LITE_CATEGORIES)
    assert all(cats[c] == 2 for c in _LITE_CATEGORIES)


def test_lite_bank_has_fewer_decisive_pairs_than_full():
    fn = compile_reward_function(DEFAULT_BOOTSTRAP_REWARD_BODY)
    lite_stats = measure_gate_stats(fn, bank=build_trajectory_bank_lite(), min_fitness_gap=BANK_MIN_FITNESS_GAP)
    full_stats = measure_gate_stats(fn, bank=build_trajectory_bank(), min_fitness_gap=BANK_MIN_FITNESS_GAP)
    assert lite_stats.n_trajectories == 16
    assert full_stats.n_trajectories == 40
    assert lite_stats.soft_decisive_pairs < full_stats.soft_decisive_pairs
    assert lite_stats.soft_decisive_pairs < 150


def test_bootstrap_passes_lite_bank_stage_b():
    from txt2reward.config.validation import LITE_BANK_MAX_SOFT_VIOLATIONS

    fn = compile_reward_function(DEFAULT_BOOTSTRAP_REWARD_BODY)
    bank = build_trajectory_bank_lite()
    stats = measure_gate_stats(fn, bank=bank, min_fitness_gap=BANK_MIN_FITNESS_GAP)
    ok, _, console = evaluate_consistency(
        fn,
        bank=bank,
        max_violation_rate=BANK_MAX_VIOLATION_RATE,
        min_fitness_gap=BANK_MIN_FITNESS_GAP,
        max_soft_violations=LITE_BANK_MAX_SOFT_VIOLATIONS,
    )
    assert stats.passive_violations == 0
    assert stats.hard_violations == 0
    assert stats.soft_violations <= LITE_BANK_MAX_SOFT_VIOLATIONS
    assert stats.soft_violation_rate <= BANK_MAX_VIOLATION_RATE
    assert ok
    assert console == "PASS"


def test_get_trajectory_bank_defaults_to_lite(monkeypatch):
    import txt2reward.config.validation as val_cfg

    monkeypatch.setattr(val_cfg, "TRAJECTORY_BANK_MODE", "lite")
    assert len(get_trajectory_bank()) == 16


def test_get_trajectory_bank_full_mode(monkeypatch):
    import txt2reward.config.validation as val_cfg

    monkeypatch.setattr(val_cfg, "TRAJECTORY_BANK_MODE", "full")
    assert len(get_trajectory_bank()) == 40
