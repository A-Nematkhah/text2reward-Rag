"""Fitness scoring: v6/v7 ablations, v8 ranking, and behavioral penalties."""

import pytest
from txt2reward.archive.archive import (
    _passive_driving_gate,
    _safety_penalty_v7,
    _survival_score_v8,
    compute_fitness,
    compute_fitness_v6,
    compute_fitness_v7,
    compute_fitness_v8,
    enrich_fitness_metrics,
    infer_curriculum_phase,
    is_passive_driving,
    near_miss_rate,
    safe_overtake_ratio,
)

from tests.helpers import base_metrics, fitness_metrics

# ── Legacy / ablation (v6–v7) ─────────────────────────────────────────────────


def test_collision_reduces_fitness():
    metrics = {
        "crash_rate": 1.0,
        "mean_speed": 0.0,
        "mean_overtakes": 0.0,
        "mean_long_jerk": 1.0,
        "mean_ttc": 1.0,
        "completion_rate": 0.0,
    }
    assert compute_fitness(metrics) < 0.1


def test_passive_safe_scores_lower_than_active_safe():
    passive = {
        "mean_speed": 20.0,
        "crash_rate": 0.0,
        "mean_overtakes": 0.0,
        "mean_long_jerk": 0.6,
        "mean_ttc": 2.4,
        "p10_ttc": 1.4,
        "min_ttc": 0.3,
        "completion_rate": 1.0,
    }
    active = {
        "mean_speed": 25.0,
        "crash_rate": 0.05,
        "mean_overtakes": 3.0,
        "mean_long_jerk": 1.0,
        "mean_ttc": 5.0,
        "p10_ttc": 4.0,
        "min_ttc": 2.0,
        "completion_rate": 0.95,
    }
    assert is_passive_driving(passive)
    assert not is_passive_driving(active)
    assert compute_fitness(passive) < compute_fitness(active) * 0.6


def test_passive_gate_inactive_while_still_crashing():
    assert _passive_driving_gate(20.0, 0.0, 0.5) == 1.0
    assert _passive_driving_gate(20.0, 0.0, 0.0) < 1.0
    assert not is_passive_driving({"crash_rate": 0.5, "mean_speed": 20.0, "mean_overtakes": 0.0})


def test_v7_transition_no_longer_beats_active():
    transition = {
        "mean_speed": 21.2,
        "crash_rate": 0.35,
        "mean_overtakes": 0.25,
        "mean_long_jerk": 7.0,
        "mean_ttc": 1.47,
        "p10_ttc": 0.49,
        "min_ttc": 0.0001,
        "total_lane_changes": 401,
        "n_episodes": 40,
        "total_overtakes": 10,
    }
    ideal = {
        "mean_speed": 27.0,
        "crash_rate": 0.08,
        "mean_overtakes": 2.5,
        "mean_long_jerk": 2.0,
        "mean_ttc": 3.5,
        "p10_ttc": 2.5,
        "min_ttc": 1.0,
        "total_lane_changes": 12,
        "n_episodes": 40,
        "total_overtakes": 100,
    }
    assert compute_fitness_v7(transition, generation=4) < compute_fitness_v7(ideal, generation=5)
    assert compute_fitness_v6(transition) > compute_fitness_v6(ideal) * 0.15


def test_v7_slow_to_survive_trend_penalised():
    prev = {
        "mean_speed": 23.0,
        "crash_rate": 0.20,
        "mean_overtakes": 0.4,
        "mean_long_jerk": 5.0,
        "mean_ttc": 2.0,
        "p10_ttc": 1.0,
        "min_ttc": 0.5,
    }
    current = {
        "mean_speed": 20.5,
        "crash_rate": 0.10,
        "mean_overtakes": 0.1,
        "mean_long_jerk": 1.0,
        "mean_ttc": 2.5,
        "p10_ttc": 1.5,
        "min_ttc": 0.8,
    }
    with_trend = compute_fitness_v7(current, generation=5, prev_metrics=prev)
    without = compute_fitness_v7(current, generation=5, prev_metrics=None)
    assert with_trend < without


# ── v8 survival ranking ───────────────────────────────────────────────────────


def test_survival_score_monotonic_above_fifty_percent():
    scores = [_survival_score_v8(cr) for cr in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]]
    assert scores == sorted(scores)
    assert scores[0] == pytest.approx(0.002, abs=0.001)
    assert scores[-1] > scores[0]


def test_high_crash_regime_preserves_ranking_not_flatline():
    fast = {"mean_speed": 29.0, "mean_overtakes": 1.0, "mean_long_jerk": 8.0, "mean_accel": 2.5}
    f100 = compute_fitness_v8(fitness_metrics(crash_rate=1.0, **fast))
    f90 = compute_fitness_v8(fitness_metrics(crash_rate=0.9, **fast))
    f80 = compute_fitness_v8(fitness_metrics(crash_rate=0.8, **fast))
    f70 = compute_fitness_v8(fitness_metrics(crash_rate=0.7, **fast))
    f60 = compute_fitness_v8(fitness_metrics(crash_rate=0.6, **fast))
    assert f100 < f90 < f80 < f70 < f60
    assert f100 != f90 != f80
    assert f100 >= 0.002


def test_v7_flatline_removed_in_v8():
    m = fitness_metrics(crash_rate=1.0, mean_speed=29.0, mean_overtakes=1.0)
    assert compute_fitness_v7(m) == 0.01
    assert compute_fitness_v8(m) != 0.01
    m2 = fitness_metrics(crash_rate=0.6, mean_speed=29.0, mean_overtakes=1.0)
    assert compute_fitness_v8(m2) > compute_fitness_v8(m)


def test_lower_crash_rate_always_higher_fitness_v8():
    assert compute_fitness_v8(fitness_metrics(crash_rate=0.1)) > compute_fitness_v8(fitness_metrics(crash_rate=0.5))


def test_same_crash_better_behavior_scores_higher():
    m_slow = fitness_metrics(crash_rate=1.0, mean_speed=20.0, mean_overtakes=0.0)
    m_fast = fitness_metrics(crash_rate=1.0, mean_speed=29.0, mean_overtakes=1.2)
    assert compute_fitness_v8(m_fast) > compute_fitness_v8(m_slow)


def test_safe_agent_beats_crash_farming_agent():
    reckless = fitness_metrics(crash_rate=1.0, mean_speed=29.5, mean_overtakes=1.0, mean_long_jerk=9.0)
    improving = fitness_metrics(crash_rate=0.25, mean_speed=26.0, mean_overtakes=1.5, mean_long_jerk=2.0)
    assert compute_fitness_v8(improving) > compute_fitness_v8(reckless)


def test_compute_fitness_defaults_to_v8():
    m = fitness_metrics(crash_rate=0.8)
    assert compute_fitness(m) == compute_fitness_v8(m)


# ── Behavioral penalties & derived metrics ────────────────────────────────────


def test_safety_penalty_increases_with_crash_rate():
    assert _safety_penalty_v7(0.05) < _safety_penalty_v7(0.3)
    assert _safety_penalty_v7(0.3) < _safety_penalty_v7(0.8)


def test_collision_dominates_fitness_at_high_crash():
    reckless = fitness_metrics(crash_rate=1.0, mean_speed=29.5, mean_long_jerk=9.0)
    safer = fitness_metrics(crash_rate=0.25, mean_speed=26.0, mean_long_jerk=2.0)
    assert compute_fitness_v8(safer) > compute_fitness_v8(reckless)


def test_passive_driving_detected_at_low_speed():
    m = {"crash_rate": 0.05, "mean_speed": 18.0, "mean_overtakes": 0.0}
    assert is_passive_driving(m)


def test_stationary_penalised_in_v8():
    assert compute_fitness_v8(fitness_metrics(mean_speed=3.0, crash_rate=0.0)) < compute_fitness_v8(
        fitness_metrics(mean_speed=25.0)
    )


def test_slow_to_survive_trend_penalised_in_v8():
    prev = base_metrics(mean_speed=26.0, crash_rate=0.25, mean_overtakes=2.0)
    current = base_metrics(mean_speed=23.0, crash_rate=0.15, mean_overtakes=1.0)
    assert compute_fitness_v8(current, prev_metrics=prev) < compute_fitness_v8(current, prev_metrics=None)


def test_jerk_spam_penalised_in_v8():
    calm = fitness_metrics(
        crash_rate=0.15,
        mean_long_jerk=0.8,
        mean_accel=0.8,
        mean_speed=27.0,
        mean_ttc=4.0,
        min_ttc=2.0,
    )
    spam = fitness_metrics(
        crash_rate=0.15,
        mean_long_jerk=8.0,
        mean_accel=5.0,
        mean_speed=22.0,
        mean_ttc=2.0,
        min_ttc=0.5,
    )
    assert compute_fitness_v8(spam) < compute_fitness_v8(calm)


def test_tailgating_penalised_vs_safe_driving():
    safe = fitness_metrics(crash_rate=0.05, min_ttc=2.5, p10_ttc=3.0, mean_ttc=5.0)
    tailgate = fitness_metrics(crash_rate=0.05, min_ttc=0.8, p10_ttc=1.2, mean_ttc=1.5)
    assert compute_fitness_v8(tailgate) < compute_fitness_v8(safe)


def test_lane_thrashing_penalised():
    efficient = fitness_metrics(crash_rate=0.08, total_lane_changes=12, total_overtakes=10, n_episodes=40)
    thrash = fitness_metrics(crash_rate=0.08, total_lane_changes=120, total_overtakes=4, n_episodes=40)
    assert compute_fitness_v8(thrash) < compute_fitness_v8(efficient)


def test_passive_cruising_beats_at_same_crash_when_active():
    passive = fitness_metrics(crash_rate=0.08, mean_speed=20.0, mean_overtakes=0.1)
    active = fitness_metrics(crash_rate=0.08, mean_speed=27.0, mean_overtakes=2.5)
    assert compute_fitness_v8(active) > compute_fitness_v8(passive)


def test_safe_overtake_ratio():
    assert safe_overtake_ratio({"total_lane_changes": 10, "total_overtakes": 5}) == 0.5
    assert safe_overtake_ratio({"total_lane_changes": 0, "total_overtakes": 0}) == 0.0


def test_near_miss_rate_from_ttc_vals():
    assert near_miss_rate({"ttc_vals": [5.0, 4.0, 1.5, 1.0, 3.0]}) == 0.4


def test_enrich_fitness_metrics_adds_derived_fields():
    raw = {
        "crash_rate": 0.1,
        "mean_speed": 26.0,
        "mean_overtakes": 1.5,
        "total_lane_changes": 20,
        "total_overtakes": 12,
        "n_episodes": 10,
        "min_ttc": 1.0,
        "p10_ttc": 2.5,
    }
    enriched = enrich_fitness_metrics(raw)
    assert enriched["safe_overtake_ratio"] == 0.6
    assert enriched["lane_change_rate"] == 2.0
    assert "near_miss_rate" in enriched
    assert enriched["curriculum_phase"] in {"survive", "speed", "overtake", "refine"}


@pytest.mark.parametrize(
    "metrics,expected",
    [
        ({"crash_rate": 0.5, "mean_speed": 28.0, "mean_overtakes": 1.0}, "survive"),
        ({"crash_rate": 0.2, "mean_speed": 20.0, "mean_overtakes": 0.5}, "speed"),
        ({"crash_rate": 0.05, "mean_speed": 26.0, "mean_overtakes": 0.2}, "overtake"),
        ({"crash_rate": 0.05, "mean_speed": 27.0, "mean_overtakes": 2.0}, "refine"),
    ],
)
def test_infer_curriculum_phase_from_metrics(metrics, expected):
    assert infer_curriculum_phase(metrics) == expected


def test_sample_confidence_penalty_shrinks_small_n_episodes():
    small_n = fitness_metrics(crash_rate=0.10, n_episodes=10)
    large_n = fitness_metrics(crash_rate=0.10, n_episodes=100)
    assert compute_fitness_v8(small_n) < compute_fitness_v8(large_n)


def test_sample_confidence_penalty_inactive_above_threshold():
    at_threshold = fitness_metrics(crash_rate=0.10, n_episodes=30)
    above_threshold = fitness_metrics(crash_rate=0.10, n_episodes=300)
    assert compute_fitness_v8(at_threshold) == pytest.approx(compute_fitness_v8(above_threshold), abs=1e-4)
