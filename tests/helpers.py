"""Shared test helpers (importable from test modules)."""

from __future__ import annotations

from txt2reward.core.types import FitnessMetrics
from txt2reward.llm.prompts import DEFAULT_BOOTSTRAP_REWARD_BODY


def passing_reward_code() -> str:
    return DEFAULT_BOOTSTRAP_REWARD_BODY.strip()


def base_metrics(**overrides) -> FitnessMetrics:
    """Default metrics for archive / evolution integration tests."""
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


def fitness_metrics(**overrides) -> FitnessMetrics:
    """Default metrics for fitness v8 unit tests (matches legacy test_fitness_v8)."""
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


def archive_entry(gen: int, code: str, metrics: dict, fitness: float, *, fitness_version: int = 8) -> dict:
    return {
        "generation": gen,
        "reward_code": code,
        "metrics": metrics,
        "fitness": fitness,
        "fitness_version": fitness_version,
        "critique": "",
        "critique_meta": {"failure_modes": [], "strengths": [], "summary": ""},
    }
