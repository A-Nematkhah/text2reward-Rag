"""Tests for fitness v8 — survival ranking, curriculum, archive retrieval."""

import pytest

from reward_archive import (
    RewardArchive,
    _survival_score_v8,
    compute_fitness,
    compute_fitness_v7,
    compute_fitness_v8,
    enrich_fitness_metrics,
    infer_curriculum_phase,
    lane_change_rate,
    near_miss_rate,
    safe_overtake_ratio,
)


def _base_metrics(**overrides):
    m = {
        "mean_speed": 27.0,
        "crash_rate": 0.0,
        "mean_overtakes": 2.0,
        "mean_long_jerk": 1.5,
        "mean_accel": 1.0,
        "mean_ttc": 4.0,
        "p10_ttc": 3.0,
        "min_ttc": 1.5,
        "total_lane_changes": 8,
        "total_overtakes": 80,
        "n_episodes": 40,
    }
    m.update(overrides)
    return m


def test_survival_score_monotonic_above_fifty_percent():
    scores = [_survival_score_v8(cr) for cr in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]]
    assert scores == sorted(scores)
    assert scores[0] == pytest.approx(0.02, abs=0.001)
    assert scores[-1] > scores[0]


def test_high_crash_regime_preserves_ranking_not_flatline():
    """100% crash must score lower than 90%, which scores lower than 80%."""
    fast = {"mean_speed": 29.0, "mean_overtakes": 1.0, "mean_long_jerk": 8.0, "mean_accel": 2.5}
    f100 = compute_fitness_v8(_base_metrics(crash_rate=1.0, **fast))
    f90 = compute_fitness_v8(_base_metrics(crash_rate=0.9, **fast))
    f80 = compute_fitness_v8(_base_metrics(crash_rate=0.8, **fast))
    f70 = compute_fitness_v8(_base_metrics(crash_rate=0.7, **fast))
    f60 = compute_fitness_v8(_base_metrics(crash_rate=0.6, **fast))
    assert f100 < f90 < f80 < f70 < f60
    assert f100 != f90 != f80
    assert f100 >= 0.015


def test_v7_flatline_removed_in_v8():
    """v7 returned 0.01 for all crash>50%; v8 differentiates."""
    m = _base_metrics(crash_rate=1.0, mean_speed=29.0, mean_overtakes=1.0)
    assert compute_fitness_v7(m) == 0.01
    assert compute_fitness_v8(m) != 0.01
    m2 = _base_metrics(crash_rate=0.6, mean_speed=29.0, mean_overtakes=1.0)
    assert compute_fitness_v8(m2) > compute_fitness_v8(m)


def test_same_crash_better_behavior_scores_higher():
    m_slow = _base_metrics(crash_rate=1.0, mean_speed=20.0, mean_overtakes=0.0)
    m_fast = _base_metrics(crash_rate=1.0, mean_speed=29.0, mean_overtakes=1.2)
    assert compute_fitness_v8(m_fast) > compute_fitness_v8(m_slow)


def test_safe_agent_beats_crash_farming_agent():
    reckless = _base_metrics(crash_rate=1.0, mean_speed=29.5, mean_overtakes=1.0, mean_long_jerk=9.0)
    improving = _base_metrics(crash_rate=0.25, mean_speed=26.0, mean_overtakes=1.5, mean_long_jerk=2.0)
    assert compute_fitness_v8(improving) > compute_fitness_v8(reckless)


def test_infer_curriculum_phase_metrics_driven():
    assert infer_curriculum_phase({"crash_rate": 0.5, "mean_speed": 28.0, "mean_overtakes": 1.0}) == "survive"
    assert infer_curriculum_phase({"crash_rate": 0.2, "mean_speed": 20.0, "mean_overtakes": 0.5}) == "speed"
    assert infer_curriculum_phase({"crash_rate": 0.05, "mean_speed": 26.0, "mean_overtakes": 0.2}) == "overtake"
    assert infer_curriculum_phase({"crash_rate": 0.05, "mean_speed": 27.0, "mean_overtakes": 2.0}) == "refine"


def test_safe_overtake_ratio():
    assert safe_overtake_ratio({"total_lane_changes": 10, "total_overtakes": 5}) == 0.5
    assert safe_overtake_ratio({"total_lane_changes": 0, "total_overtakes": 0}) == 0.0


def test_compute_fitness_defaults_to_v8():
    m = _base_metrics(crash_rate=0.8)
    assert compute_fitness(m) == compute_fitness_v8(m)


def test_archive_deduplicates_identical_reward_code(tmp_path):
    path = tmp_path / "archive.json"
    archive = RewardArchive(str(path))
    code = "def compute_reward(state):\n    return 1.0\n"
    for i in range(5):
        archive.entries.append(
            {
                "generation": i,
                "reward_code": code,
                "metrics": {"crash_rate": 1.0 - i * 0.1, "mean_speed": 25.0, "mean_overtakes": 1.0},
                "fitness": 0.05 + i * 0.01,
                "critique": "",
                "critique_meta": {"failure_modes": [], "strengths": [], "summary": ""},
            }
        )
    top = archive.get_top_k(3)
    assert len(top) == 1


def test_jerk_spam_penalised_in_v8():
    calm = _base_metrics(
        crash_rate=0.15,
        mean_long_jerk=0.8,
        mean_accel=0.8,
        mean_speed=27.0,
        mean_ttc=4.0,
        min_ttc=2.0,
    )
    spam = _base_metrics(
        crash_rate=0.15,
        mean_long_jerk=8.0,
        mean_accel=5.0,
        mean_speed=22.0,
        mean_ttc=2.0,
        min_ttc=0.5,
    )
    assert compute_fitness_v8(spam) < compute_fitness_v8(calm)


def test_near_miss_rate_from_ttc_vals():
    m = {"ttc_vals": [5.0, 4.0, 1.5, 1.0, 3.0]}
    assert near_miss_rate(m) == 0.4


def test_tailgating_penalised_vs_safe_driving():
    safe = _base_metrics(crash_rate=0.05, min_ttc=2.5, p10_ttc=3.0, mean_ttc=5.0)
    tailgate = _base_metrics(crash_rate=0.05, min_ttc=0.8, p10_ttc=1.2, mean_ttc=1.5)
    assert compute_fitness_v8(tailgate) < compute_fitness_v8(safe)


def test_lane_thrashing_penalised():
    efficient = _base_metrics(
        crash_rate=0.08,
        total_lane_changes=12,
        total_overtakes=10,
        n_episodes=40,
    )
    thrash = _base_metrics(
        crash_rate=0.08,
        total_lane_changes=120,
        total_overtakes=4,
        n_episodes=40,
    )
    assert compute_fitness_v8(thrash) < compute_fitness_v8(efficient)


def test_passive_cruising_beats_at_same_crash_when_active():
    passive = _base_metrics(crash_rate=0.08, mean_speed=20.0, mean_overtakes=0.1)
    active = _base_metrics(crash_rate=0.08, mean_speed=27.0, mean_overtakes=2.5)
    assert compute_fitness_v8(active) > compute_fitness_v8(passive)


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
    assert "safe_overtake_ratio" in enriched
    assert "lane_change_rate" in enriched
    assert "near_miss_rate" in enriched
    assert enriched["curriculum_phase"] in {"survive", "speed", "overtake", "refine"}
    assert enriched["safe_overtake_ratio"] == 0.6
    assert enriched["lane_change_rate"] == 2.0
