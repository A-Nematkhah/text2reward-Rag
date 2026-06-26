"""Shared TypedDicts, Literals, and Protocols for cross-module data contracts."""

from __future__ import annotations

from typing import Callable, Literal, TypedDict

# ── Curriculum ────────────────────────────────────────────────────────────────

CurriculumPhase = Literal["survive", "speed", "overtake", "refine"]

CURRICULUM_PHASES: tuple[CurriculumPhase, ...] = (
    "survive",
    "speed",
    "overtake",
    "refine",
)

# ── Reward sandbox state (passed to compute_reward) ───────────────────────────


class RewardState(TypedDict):
    """Canonical per-step state dict for LLM-generated reward functions."""

    speed_ms: float
    front_dist: float
    ttc: float
    rel_vel_ms: float
    lane: int
    overtook: bool
    lane_changed: bool
    collided: bool
    nearby_vehicles: int
    accel_ms2: float
    long_jerk: float
    lat_jerk: float


RewardFn = Callable[[RewardState], float]

# ── Trajectory / episode observability ────────────────────────────────────────


class TrajectorySample(TypedDict, total=False):
    """Single step snapshot stored in episode_stats['trajectory_samples']."""

    speed_ms: float
    lane: int
    front_dist: float
    ttc: float
    rel_vel_ms: float
    accel_ms2: float
    nearby_vehicles: int
    overtook: bool
    collided: bool


class EpisodeStats(TypedDict, total=False):
    """Per-episode summary emitted by LLMRewardWrapper in info['episode_stats']."""

    total_env_reward: float
    total_shaped_reward: float
    mean_speed: float
    mean_front_dist: float
    collisions: int
    steps: int
    mean_ttc: float
    min_ttc: float
    p10_ttc: float
    ttc_vals: list[float]
    mean_rel_vel: float
    mean_long_jerk: float
    mean_lat_jerk: float
    mean_accel: float
    mean_density: float
    total_overtakes: int
    total_lane_changes: int
    trajectory_samples: list[TrajectorySample]


class EvalEpisodeResult(TypedDict):
    """Single-episode result from evaluation.evaluate.run_episode."""

    total_reward: float
    steps: int
    crashed: bool
    mean_speed: float
    overtakes: int
    lane_changes: int
    mean_ttc: float
    p10_ttc: float
    min_ttc: float
    ttc_vals: list[float]
    mean_long_jerk: float
    mean_accel: float


# ── Fitness & archive ─────────────────────────────────────────────────────────


class CoreMetrics(TypedDict):
    """Minimum metric fields used by curriculum inference and smoke tests."""

    crash_rate: float
    mean_speed: float
    mean_overtakes: float


class FitnessMetrics(CoreMetrics, total=False):
    """Aggregated driving metrics for fitness scoring and archive storage."""

    mean_steps: float
    completion_rate: float
    mean_ttc: float
    p10_ttc: float
    min_ttc: float
    mean_long_jerk: float
    mean_lat_jerk: float
    mean_accel: float
    mean_rel_vel: float
    max_steps: int
    n_episodes: int
    total_overtakes: int
    total_lane_changes: int
    ttc_vals: list[float]
    curriculum_phase: CurriculumPhase
    near_miss_rate: float
    safe_overtake_ratio: float
    lane_change_rate: float


class CritiqueMeta(TypedDict):
    failure_modes: list[str]
    strengths: list[str]
    summary: str


class ArchiveEntry(TypedDict):
    generation: int
    reward_code: str
    metrics: FitnessMetrics
    fitness: float
    fitness_version: int
    critique: str
    critique_meta: CritiqueMeta
    timestamp: str


class ArchiveFile(TypedDict):
    meta: dict[str, object]
    entries: list[ArchiveEntry]
