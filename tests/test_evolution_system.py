"""
Task 7 — consolidated evolution-system tests.

Covers the six areas required by the implementation plan:
  1. Fitness ranking
  2. Crash penalties
  3. Activity penalties
  4. Trajectory-bank consistency
  5. Archive retrieval
  6. Curriculum transitions

Plus an end-to-end metrics → fitness → archive → LLM-context pipeline check.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from reward_archive import (
    RewardArchive,
    _safety_penalty_v7,
    _survival_score_v8,
    compute_fitness,
    compute_fitness_v8,
    effective_fitness,
    infer_curriculum_phase,
    is_crash_farming,
    is_passive_driving,
)
from reward_designer import (
    DEFAULT_BOOTSTRAP_REWARD_BODY,
    RewardDesigner,
    _full_validation_pipeline,
    _smoke_test_reward_code,
)
from reward_sandbox import compile_reward_function
from trajectory_bank import (
    BANK_MAX_VIOLATION_RATE,
    build_trajectory_bank,
    evaluate_consistency,
    measure_gate_stats,
)


def _base_metrics(**overrides):
    m = {
        "mean_speed": 27.0,
        "crash_rate": 0.08,
        "mean_overtakes": 2.0,
        "mean_long_jerk": 1.5,
        "mean_accel": 1.0,
        "mean_ttc": 4.0,
        "p10_ttc": 3.0,
        "min_ttc": 2.0,
        "total_lane_changes": 12,
        "total_overtakes": 10,
        "n_episodes": 40,
        "completion_rate": 0.92,
    }
    m.update(overrides)
    return m


# ── 1. Fitness ranking ────────────────────────────────────────────────────────


class TestFitnessRanking:
    def test_lower_crash_rate_always_higher_fitness_v8(self):
        low = _base_metrics(crash_rate=0.1)
        high = _base_metrics(crash_rate=0.5)
        assert compute_fitness_v8(low) > compute_fitness_v8(high)

    def test_survival_score_monotonic_above_fifty_percent_crash(self):
        scores = [_survival_score_v8(cr) for cr in [1.0, 0.8, 0.6, 0.5]]
        assert scores == sorted(scores)
        assert scores[0] < scores[-1]

    def test_same_crash_active_beats_passive(self):
        passive = _base_metrics(crash_rate=0.1, mean_speed=20.0, mean_overtakes=0.1)
        active = _base_metrics(crash_rate=0.1, mean_speed=27.0, mean_overtakes=2.5)
        assert compute_fitness_v8(active) > compute_fitness_v8(passive)

    def test_v8_default_is_not_v7_flatline(self):
        m = _base_metrics(crash_rate=1.0, mean_speed=29.0, mean_overtakes=1.0)
        assert compute_fitness(m) > 0.015
        assert compute_fitness(m) != 0.01


# ── 2. Crash penalties ────────────────────────────────────────────────────────


class TestCrashPenalties:
    def test_safety_penalty_increases_with_crash_rate(self):
        assert _safety_penalty_v7(0.05) < _safety_penalty_v7(0.3)
        assert _safety_penalty_v7(0.3) < _safety_penalty_v7(0.8)

    def test_collision_dominates_fitness_at_high_crash(self):
        reckless = _base_metrics(crash_rate=1.0, mean_speed=29.5, mean_long_jerk=9.0)
        safer = _base_metrics(crash_rate=0.25, mean_speed=26.0, mean_long_jerk=2.0)
        assert compute_fitness_v8(safer) > compute_fitness_v8(reckless)

    def test_weak_collision_rejected_by_stage_a(self):
        code = (
            "def compute_reward(state):\n"
            "    if state['collided']:\n"
            "        return -30.0\n"
            "    return clip(state['speed_ms'] * 0.15, 0.0, 5.0)\n"
        )
        ok, err = _smoke_test_reward_code(code)
        assert not ok
        assert "collision" in err.lower() or "crash" in err.lower() or "farming" in err.lower()

    def test_bootstrap_passes_full_validation(self):
        ok, err, _ = _full_validation_pipeline(DEFAULT_BOOTSTRAP_REWARD_BODY)
        assert ok, err


# ── 3. Activity penalties ─────────────────────────────────────────────────────


class TestActivityPenalties:
    def test_passive_driving_detected_at_low_speed(self):
        m = {"crash_rate": 0.05, "mean_speed": 18.0, "mean_overtakes": 0.0}
        assert is_passive_driving(m)

    def test_stationary_penalised_in_v8(self):
        moving = _base_metrics(mean_speed=25.0)
        stationary = _base_metrics(mean_speed=3.0, crash_rate=0.0)
        assert compute_fitness_v8(stationary) < compute_fitness_v8(moving)

    def test_slow_to_survive_trend_penalised_in_v8(self):
        prev = _base_metrics(mean_speed=24.0, crash_rate=0.20, mean_overtakes=0.5)
        current = _base_metrics(mean_speed=20.0, crash_rate=0.10, mean_overtakes=0.1)
        with_trend = compute_fitness_v8(current, prev_metrics=prev)
        without = compute_fitness_v8(current, prev_metrics=None)
        assert with_trend < without

    def test_jerk_spam_scores_below_calm_driving(self):
        calm = _base_metrics(mean_long_jerk=0.8, mean_accel=0.8, min_ttc=2.5)
        spam = _base_metrics(mean_long_jerk=8.0, mean_accel=5.0, min_ttc=0.5, mean_speed=22.0)
        assert compute_fitness_v8(spam) < compute_fitness_v8(calm)


# ── 4. Trajectory-bank consistency ────────────────────────────────────────────


class TestTrajectoryBankConsistency:
    def test_bootstrap_soft_rate_below_threshold(self):
        bank = build_trajectory_bank()
        fn = compile_reward_function(DEFAULT_BOOTSTRAP_REWARD_BODY)
        stats = measure_gate_stats(fn, bank=bank)
        assert stats.passive_violations == 0
        assert stats.hard_violations == 0
        assert stats.soft_violation_rate <= BANK_MAX_VIOLATION_RATE

    def test_crash_rewarding_function_fails_hard_check(self):
        bank = build_trajectory_bank()

        def bad(state):
            return 100.0 if state["collided"] else 0.0

        ok, report, _ = evaluate_consistency(bad, bank=bank, max_violation_rate=1.0)
        assert not ok
        assert "hard safety violations" in report

    def test_passive_safe_gap_fails_passive_hard_check(self):
        code = """
def compute_reward(state):
    if state["collided"]:
        return -80.0
    r = 0.05 * state["speed_ms"]
    if state["front_dist"] > 40:
        r += 2.0
    return r
"""
        bank = build_trajectory_bank()
        fn = compile_reward_function(code)
        stats = measure_gate_stats(fn, bank=bank)
        ok, _, _ = evaluate_consistency(fn, bank=bank, max_violation_rate=1.0)
        assert stats.passive_violations > 0 or not ok


# ── 5. Archive retrieval ──────────────────────────────────────────────────────


class TestArchiveRetrieval:
    def test_top_k_dedupes_identical_code(self, tmp_path):
        archive = RewardArchive(str(tmp_path / "a.json"))
        code = "def compute_reward(state):\n    return 1.0\n"
        for i in range(4):
            archive.entries.append(
                {
                    "generation": i,
                    "reward_code": code,
                    "metrics": _base_metrics(crash_rate=0.1 + i * 0.05),
                    "fitness": 0.5 + i * 0.05,
                    "fitness_version": 8,
                    "critique": "",
                    "critique_meta": {"failure_modes": [], "strengths": [], "summary": ""},
                }
            )
        assert len(archive.get_top_k(3)) == 1

    def test_top_k_excludes_crash_farming_when_safer_exists(self, tmp_path):
        archive = RewardArchive(str(tmp_path / "b.json"))
        archive.entries = [
            {
                "generation": 0,
                "reward_code": "def compute_reward(state):\n    return 1.0\n",
                "metrics": _base_metrics(crash_rate=0.15),
                "fitness": 0.55,
                "fitness_version": 8,
                "critique": "",
                "critique_meta": {"failure_modes": [], "strengths": [], "summary": ""},
            },
            {
                "generation": 1,
                "reward_code": "def compute_reward(state):\n    return 2.0\n",
                "metrics": _base_metrics(crash_rate=1.0, mean_speed=29.0, mean_overtakes=1.0),
                "fitness": 0.58,
                "fitness_version": 8,
                "critique": "",
                "critique_meta": {"failure_modes": [], "strengths": [], "summary": ""},
            },
        ]
        top = archive.get_top_k(2)
        assert len(top) == 1
        assert top[0]["generation"] == 0
        assert is_crash_farming(archive.entries[1]["metrics"])

    def test_failed_rewards_include_crash_farming(self, tmp_path):
        archive = RewardArchive(str(tmp_path / "c.json"))
        archive.entries = [
            {
                "generation": 0,
                "reward_code": "def compute_reward(state):\n    return 1.0\n",
                "metrics": _base_metrics(crash_rate=1.0, mean_speed=29.0, mean_overtakes=1.0),
                "fitness": 0.12,
                "fitness_version": 8,
                "critique": "",
                "critique_meta": {"failure_modes": [], "strengths": [], "summary": ""},
            },
        ]
        failed = archive.get_failed_rewards(k=2, max_fitness=0.08)
        assert any(e["generation"] == 0 for e in failed)

    def test_effective_fitness_uses_current_metrics_not_stale_score(self, tmp_path):
        archive = RewardArchive(str(tmp_path / "d.json"))
        entry = {
            "generation": 0,
            "reward_code": "def compute_reward(state):\n    return 1.0\n",
            "metrics": _base_metrics(crash_rate=1.0, mean_speed=29.0),
            "fitness": 0.99,
            "fitness_version": 8,
            "critique": "",
            "critique_meta": {"failure_modes": [], "strengths": [], "summary": ""},
        }
        assert effective_fitness(entry) < 0.2


# ── 6. Curriculum transitions ───────────────────────────────────────────────


class TestCurriculumTransitions:
    @pytest.mark.parametrize(
        "metrics,expected",
        [
            ({"crash_rate": 0.5, "mean_speed": 28.0, "mean_overtakes": 1.0}, "survive"),
            ({"crash_rate": 0.2, "mean_speed": 20.0, "mean_overtakes": 0.5}, "speed"),
            ({"crash_rate": 0.05, "mean_speed": 26.0, "mean_overtakes": 0.2}, "overtake"),
            ({"crash_rate": 0.05, "mean_speed": 27.0, "mean_overtakes": 2.0}, "refine"),
        ],
    )
    def test_phase_from_metrics_not_generation(self, metrics, expected):
        assert infer_curriculum_phase(metrics) == expected
        assert infer_curriculum_phase(metrics) == infer_curriculum_phase(metrics)

    def test_format_for_llm_includes_curriculum_header(self, tmp_path):
        archive = RewardArchive(str(tmp_path / "e.json"))
        archive.add_entry(
            "def compute_reward(state):\n    return 1.0\n",
            _base_metrics(),
        )
        text = archive.format_for_llm(k=1, curriculum_phase="speed")
        assert "CURRENT CURRICULUM PHASE: speed" in text


# ── 7. End-to-end pipeline ────────────────────────────────────────────────────


class TestEndToEndPipeline:
    def test_aggregate_metrics_to_archive_to_top_k(self):
        episode_stats = [
            {
                "mean_speed": 26.0,
                "collisions": 0,
                "steps": 120,
                "total_overtakes": 2,
                "total_lane_changes": 3,
                "mean_ttc": 5.0,
                "p10_ttc": 4.0,
                "min_ttc": 3.0,
                "mean_long_jerk": 1.0,
                "mean_accel": 0.8,
            }
            for _ in range(40)
        ]
        metrics = RewardDesigner._aggregate_metrics(episode_stats)
        assert metrics["curriculum_phase"] in {"survive", "speed", "overtake", "refine"}
        assert metrics["crash_rate"] == 0.0

        workdir = tempfile.mkdtemp()
        archive = RewardArchive(os.path.join(workdir, "archive.json"))
        entry = archive.add_entry(
            "def compute_reward(state):\n    return 1.0\n",
            metrics,
        )
        assert entry["metrics"]["curriculum_phase"] == metrics["curriculum_phase"]
        assert entry["fitness"] == pytest.approx(compute_fitness(metrics), rel=1e-4)

        top = archive.get_top_k(1)
        assert len(top) == 1
        assert top[0]["generation"] == 0

    def test_evolve_archives_before_generating(self, monkeypatch):
        workdir = tempfile.mkdtemp()
        archive_path = os.path.join(workdir, "reward_archive.json")
        reward_path = os.path.join(workdir, "reward_program.py")
        with open(reward_path, "w", encoding="utf-8") as f:
            f.write("def compute_reward(state):\n    return 1.0\n")

        designer = RewardDesigner(archive_path=archive_path, reward_path=reward_path, verbose=False)
        monkeypatch.setattr(designer, "_call_generate_with_repair", lambda *a, **k: None)
        monkeypatch.setattr(designer, "_call_critique", lambda *a, **k: "")

        designer._episode_stats = [
            {
                "mean_speed": 25.0,
                "collisions": 1,
                "steps": 80,
                "total_overtakes": 1,
                "total_lane_changes": 2,
                "mean_ttc": 3.0,
                "p10_ttc": 2.0,
                "min_ttc": 1.0,
                "mean_long_jerk": 2.0,
                "mean_accel": 1.0,
            }
            for _ in range(40)
        ]
        designer._episode_count = designer.warmup_episodes
        designer._evolve()

        assert len(designer.archive.entries) == 1
        assert designer.get_last_evolution_metrics() is not None
        assert "curriculum_phase" in designer.get_last_evolution_metrics()
