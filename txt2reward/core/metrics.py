"""Derived driving metrics and episode-level TTC pooling."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

from txt2reward.core.constants import HIGHWAY_MAX_STEPS, HIGHWAY_SPEED_SCALE
from txt2reward.core.types import EpisodeStats, EvalEpisodeResult, FitnessMetrics

NEAR_MISS_TTC_THRESHOLD = 2.0

# near_miss_rate references threshold constant below


def safe_overtake_ratio(metrics: FitnessMetrics | Mapping[str, Any]) -> float:
    """Fraction of lane changes that produced an overtake (0–1)."""
    lane_changes = max(int(metrics.get("total_lane_changes", 0)), 0)
    if lane_changes <= 0:
        return 0.0
    total_overtakes = metrics.get("total_overtakes")
    if total_overtakes is None:
        n_eps = max(int(metrics.get("n_episodes", 1)), 1)
        total_overtakes = float(metrics.get("mean_overtakes", 0.0)) * n_eps
    return float(min(1.0, float(total_overtakes) / lane_changes))


def percentile(values: list[float], pct: int, *, default: float = 30.0) -> float:
    """Linear-interpolation percentile (0–100). Returns ``default`` when empty."""
    if not values:
        return default
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def collect_episode_ttc_values(
    episode_stats: Sequence[EpisodeStats | EvalEpisodeResult | Mapping[str, Any]],
    *,
    use_trajectory_samples: bool = False,
) -> list[float]:
    """Gather per-step TTC observations from a batch of episode stat dicts."""
    all_ttc: list[float] = []
    for stats in episode_stats:
        if stats.get("ttc_vals"):
            all_ttc.extend(float(v) for v in stats["ttc_vals"])
        elif use_trajectory_samples:
            samples = stats.get("trajectory_samples", [])
            if isinstance(samples, list):
                for sample in samples:
                    if isinstance(sample, Mapping):
                        all_ttc.append(float(sample.get("ttc", 30.0)))
        else:
            all_ttc.append(float(stats.get("min_ttc", 30.0)))
    return all_ttc


def denormalize_speed(vx_raw: float) -> float:
    """Convert highway-env vx observation to m/s."""
    if abs(vx_raw) <= 1.5:
        return max(0.0, vx_raw * HIGHWAY_SPEED_SCALE)
    return max(0.0, vx_raw)


def pool_ttc_p10_min(
    episode_stats: Sequence[EpisodeStats | EvalEpisodeResult | Mapping[str, Any]],
    *,
    use_trajectory_samples: bool = False,
) -> tuple[float, float]:
    """Return pooled ``(p10_ttc, min_ttc)`` across episodes."""
    p10_ttc, min_ttc, _ = pool_episode_ttc(episode_stats, use_trajectory_samples=use_trajectory_samples)
    return p10_ttc, min_ttc


def pool_episode_ttc(
    episode_stats: Sequence[EpisodeStats | EvalEpisodeResult | Mapping[str, Any]],
    *,
    use_trajectory_samples: bool = False,
) -> tuple[float, float, list[float]]:
    """
    Pool step-level TTC across episodes.

    Returns ``(p10_ttc, min_ttc, all_ttc_values)``. When no step-level values
    exist, falls back to per-episode ``p10_ttc`` / ``min_ttc`` aggregates.
    """
    all_ttc = collect_episode_ttc_values(episode_stats, use_trajectory_samples=use_trajectory_samples)
    n = max(len(episode_stats), 1)
    if all_ttc:
        return percentile(all_ttc, 10), min(all_ttc), all_ttc
    p10_fallback = sum(float(s.get("p10_ttc", 30.0)) for s in episode_stats) / n
    min_fallback = min((float(s.get("min_ttc", 30.0)) for s in episode_stats), default=30.0)
    return p10_fallback, min_fallback, []


def lane_change_rate(metrics: FitnessMetrics | Mapping[str, Any]) -> float:
    """Mean lane changes per episode."""
    n_eps = max(int(metrics.get("n_episodes", 1)), 1)
    return float(metrics.get("total_lane_changes", 0)) / n_eps


def near_miss_rate(metrics: FitnessMetrics | Mapping[str, Any], *, threshold: float = NEAR_MISS_TTC_THRESHOLD) -> float:
    """
    Fraction of steps with TTC below ``threshold`` (default 2 s).

    Uses per-step ``ttc_vals`` when present (aggregated from episode_stats).
    Falls back to ``min_ttc`` / ``p10_ttc`` for legacy archive entries.
    """
    vals = metrics.get("ttc_vals")
    if isinstance(vals, list) and vals:
        n = len(vals)
        return float(sum(1 for v in vals if float(v) < threshold) / n)

    min_ttc = float(metrics.get("min_ttc", -1.0))
    p10_ttc = float(metrics.get("p10_ttc", -1.0))
    if min_ttc < 0 and p10_ttc < 0:
        return 0.0
    effective_min = min_ttc if min_ttc >= 0 else p10_ttc
    if effective_min < threshold:
        return float(min(1.0, 0.4 + 0.6 * (1.0 - effective_min / threshold)))
    if p10_ttc >= 0 and p10_ttc < threshold:
        return 0.20
    return 0.0


def enrich_fitness_metrics(metrics: FitnessMetrics | Mapping[str, Any]) -> FitnessMetrics:
    """Attach derived v8 metrics used by fitness and LLM critique."""
    from txt2reward.archive.curriculum import infer_curriculum_phase

    out = dict(metrics)
    if "near_miss_rate" not in out:
        out["near_miss_rate"] = near_miss_rate(out)
    out["safe_overtake_ratio"] = safe_overtake_ratio(out)
    out["lane_change_rate"] = lane_change_rate(out)
    out["curriculum_phase"] = infer_curriculum_phase(out)
    return cast(FitnessMetrics, out)


def aggregate_single_trajectory(states: list[Mapping[str, Any]]) -> dict[str, float]:
    """Reduce per-step trajectory states to a single-episode metric dict."""
    n = max(len(states), 1)
    crashed = any(s.get("collided") for s in states)
    overtakes = sum(1 for s in states if s.get("overtook"))
    lane_changes = sum(1 for s in states if s.get("lane_changed"))
    speed_sum = sum(float(s["speed_ms"]) for s in states)
    jerk_sum = sum(abs(float(s.get("long_jerk", 0.0))) for s in states)
    ttc_sum = sum(float(s.get("ttc", 30.0)) for s in states)
    ttc_vals = [float(s.get("ttc", 30.0)) for s in states]

    return {
        "mean_speed": speed_sum / n,
        "crash_rate": 1.0 if crashed else 0.0,
        "mean_overtakes": float(overtakes),
        "total_overtakes": float(overtakes),
        "total_lane_changes": float(lane_changes),
        "n_episodes": 1,
        "mean_long_jerk": jerk_sum / n,
        "mean_ttc": ttc_sum / n,
        "p10_ttc": percentile(ttc_vals, 10),
        "min_ttc": min(ttc_vals) if ttc_vals else 30.0,
        "completion_rate": 0.0 if crashed else 1.0,
    }


def aggregate_episode_stats(
    episode_stats: Sequence[EpisodeStats | Mapping[str, Any]],
    *,
    use_trajectory_samples: bool = False,
    enrich: bool = True,
    include_ttc_vals: bool = True,
) -> FitnessMetrics:
    """Aggregate training ``episode_stats`` dicts into archive-ready metrics."""
    n = max(len(episode_stats), 1)
    crashes = sum(1 for s in episode_stats if s.get("collisions", 0) > 0)
    total_overtakes = sum(s.get("total_overtakes", 0) for s in episode_stats)
    p10_ttc, min_ttc, all_ttc = pool_episode_ttc(episode_stats, use_trajectory_samples=use_trajectory_samples)
    aggregated: FitnessMetrics = {
        "n_episodes": n,
        "mean_speed": sum(s.get("mean_speed", 0) for s in episode_stats) / n,
        "crash_rate": crashes / n,
        "completion_rate": 1.0 - crashes / n,
        "mean_overtakes": total_overtakes / n,
        "mean_steps": sum(s.get("steps", 0) for s in episode_stats) / n,
        "mean_ttc": sum(s.get("mean_ttc", 0) for s in episode_stats) / n,
        "p10_ttc": p10_ttc,
        "min_ttc": min_ttc,
        "mean_rel_vel": sum(s.get("mean_rel_vel", 0) for s in episode_stats) / n,
        "mean_long_jerk": sum(s.get("mean_long_jerk", 0) for s in episode_stats) / n,
        "mean_lat_jerk": sum(s.get("mean_lat_jerk", 0) for s in episode_stats) / n,
        "mean_accel": sum(s.get("mean_accel", 0) for s in episode_stats) / n,
        "total_overtakes": total_overtakes,
        "total_lane_changes": sum(s.get("total_lane_changes", 0) for s in episode_stats),
        "max_steps": HIGHWAY_MAX_STEPS,
    }
    if all_ttc and include_ttc_vals:
        aggregated["ttc_vals"] = all_ttc
    if enrich:
        return enrich_fitness_metrics(aggregated)
    return aggregated


def aggregate_eval_fitness_metrics(
    episode_results: Sequence[EvalEpisodeResult | Mapping[str, Any]],
) -> FitnessMetrics:
    """
    Fitness metric dict for evaluate.py — same keys as the legacy inline builder.

    Omits enriched/extended fields so evaluation fitness stays unchanged.
    """
    n = max(len(episode_results), 1)
    crash_rate = sum(1 for r in episode_results if r.get("crashed")) / n
    p10_ttc, min_ttc = pool_ttc_p10_min(episode_results)
    return {
        "mean_speed": sum(r.get("mean_speed", 0) for r in episode_results) / n,
        "crash_rate": crash_rate,
        "mean_overtakes": sum(r.get("overtakes", 0) for r in episode_results) / n,
        "mean_steps": sum(r.get("steps", 0) for r in episode_results) / n,
        "completion_rate": 1.0 - crash_rate,
        "mean_ttc": sum(r.get("mean_ttc", 30.0) for r in episode_results) / n,
        "p10_ttc": p10_ttc,
        "min_ttc": min_ttc,
        "mean_long_jerk": sum(r.get("mean_long_jerk", 0) for r in episode_results) / n,
        "mean_accel": sum(r.get("mean_accel", 0) for r in episode_results) / n,
        "max_steps": HIGHWAY_MAX_STEPS,
    }
