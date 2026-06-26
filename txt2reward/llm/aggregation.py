"""Aggregate episode stats from training into archive-ready metrics.

Public helpers format metric trends and trajectory samples for LLM critique
prompts during reward evolution.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

from txt2reward.archive.curriculum import infer_curriculum_transition
from txt2reward.core.metrics import aggregate_episode_stats
from txt2reward.core.types import EpisodeStats, FitnessMetrics


def aggregate_episode_metrics(episode_stats: Sequence[EpisodeStats | Mapping[str, Any]]) -> FitnessMetrics:
    """Aggregate wrapper episode stats into enriched fitness metrics.

    Args:
        episode_stats: Completed episodes from ``LLMRewardWrapper`` (uses
            trajectory samples for TTC pooling when present).

    Returns:
        ``FitnessMetrics`` with derived fields (near-miss rate, curriculum
        phase, etc.) suitable for archive storage and critique.
    """
    return aggregate_episode_stats(episode_stats, use_trajectory_samples=True, enrich=True)


def format_metric_trend(
    current_metrics: FitnessMetrics | Mapping[str, Any],
    previous_entry: Mapping[str, Any] | None,
) -> str:
    """Human-readable delta summary between two archive generations.

    Args:
        current_metrics: Metrics from the evolution window just completed.
        previous_entry: Prior archive entry (``generation``, ``metrics``), or
            ``None`` for generation 0.

    Returns:
        Multi-line string for the critique prompt, including curriculum
        transition and a warning when crash rate improved but speed/overtakes
        fell (classic passive-driving hack).
    """
    if previous_entry is None:
        return "(no previous generation to compare against — this is generation 0)"
    prev = cast(Mapping[str, Any], previous_entry["metrics"])
    transition = infer_curriculum_transition(prev, current_metrics)

    def _delta(key: str, fmt: str = "{:+.2f}") -> str:
        cur = cast(float, current_metrics.get(key, 0.0))
        old = cast(float, prev.get(key, 0.0))
        return fmt.format(cur - old)

    warning = ""
    if (
        float(current_metrics.get("mean_speed", 0.0)) < float(prev.get("mean_speed", 0.0))
        or float(current_metrics.get("mean_overtakes", 0.0)) < float(prev.get("mean_overtakes", 0.0))
    ) and float(current_metrics.get("crash_rate", 1.0)) < float(prev.get("crash_rate", 1.0)):
        warning = (
            "\n  !! WARNING: crash_rate improved but speed and/or overtakes "
            "DECREASED vs the previous generation. This is the classic "
            "'slow down to stay safe' reward-hacking pattern.\n"
        )
    return (
        f"  {transition}\n"
        f"  previous generation : {previous_entry['generation']}\n"
        f"  mean_speed     delta: {_delta('mean_speed')} m/s\n"
        f"  mean_overtakes delta: {_delta('mean_overtakes')} per episode\n"
        f"  crash_rate     delta: {_delta('crash_rate', '{:+.1%}')}"
        f"{warning}"
    )


def format_trajectory_samples(episode_stats: Sequence[EpisodeStats | Mapping[str, Any]]) -> str:
    """Format recent per-step snapshots for the LLM critique prompt.

    Args:
        episode_stats: Recent evolution-window episodes (last five used).

    Returns:
        Multi-line text block, or ``(no samples)`` when trajectory data is absent.
    """
    lines = []
    for s in episode_stats[-5:]:
        for sample in s.get("trajectory_samples", [])[:3]:
            if isinstance(sample, dict):
                lines.append(
                    f"  speed={sample.get('speed_ms', 0):>6.2f} m/s  "
                    f"lane={sample.get('lane', '?')}  "
                    f"front={sample.get('front_dist', 0):>6.1f} m  "
                    f"ttc={sample.get('ttc', 0):>5.1f} s  "
                    f"overtook={sample.get('overtook', False)}  "
                    f"crash={sample.get('collided', False)}"
                )
    return "\n".join(lines) if lines else "  (no samples)"
