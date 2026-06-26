"""Metrics-driven curriculum phase inference for LLM prompts."""

from __future__ import annotations

from typing import Any, Mapping

from txt2reward.core.types import CURRICULUM_PHASES, CoreMetrics, CurriculumPhase, FitnessMetrics

__all__ = [
    "CURRICULUM_GUIDANCE",
    "CURRICULUM_PHASES",
    "curriculum_guidance",
    "infer_curriculum_phase",
    "infer_curriculum_transition",
]


def infer_curriculum_phase(metrics: CoreMetrics | FitnessMetrics | Mapping[str, Any]) -> CurriculumPhase:
    """
    Metrics-driven curriculum (not generation count).

    Phase 1 survive  : crash still dominant — prioritise not crashing
    Phase 2 speed    : mostly safe — push speed without losing safety
    Phase 3 overtake : fast enough — reward active overtaking
    Phase 4 refine   : balance comfort, efficiency, and sustained activity
    """
    crash_rate = float(metrics.get("crash_rate", 1.0))
    mean_speed = float(metrics.get("mean_speed", 0.0))
    mean_overtakes = float(metrics.get("mean_overtakes", 0.0))
    if crash_rate > 0.35:
        return "survive"
    if crash_rate > 0.12 or mean_speed < 22.0:
        return "speed"
    if mean_overtakes < 1.0:
        return "overtake"
    return "refine"


CURRICULUM_GUIDANCE: dict[CurriculumPhase, str] = {
    "survive": (
        "Agent crashes too often. Prioritise survival: strong collision penalty "
        "(-70 to -100), stronger TTC/tailgate penalties, moderate speed reward."
    ),
    "speed": (
        "Crashes are improving but speed is low OR still elevated. Balance safety "
        "with speed — do not remove collision penalty; increase speed incentive only "
        "under safe TTC/front_dist conditions."
    ),
    "overtake": (
        "Agent is reasonably safe and fast but under-overtaking. Increase overtake "
        "bonus and cruise_tax on clear roads without overtakes; keep collision penalty."
    ),
    "refine": (
        "All core metrics are reasonable — refine comfort, lane efficiency, and "
        "avoid jerk/accel spam while maintaining speed and overtakes."
    ),
}


def curriculum_guidance(phase: CurriculumPhase | str) -> str:
    """LLM-facing instructions for the current metrics-driven curriculum phase."""
    if phase in CURRICULUM_GUIDANCE:
        return CURRICULUM_GUIDANCE[phase]
    return CURRICULUM_GUIDANCE["survive"]


def infer_curriculum_transition(
    prev_metrics: FitnessMetrics | Mapping[str, Any] | None,
    current_metrics: FitnessMetrics | Mapping[str, Any],
) -> str:
    """Human-readable phase change summary for critique / logs."""
    cur_phase = infer_curriculum_phase(current_metrics)
    if prev_metrics is None:
        return f"curriculum_phase={cur_phase} (first generation)"
    prev_phase = infer_curriculum_phase(prev_metrics)
    if prev_phase == cur_phase:
        return f"curriculum_phase={cur_phase} (unchanged)"
    return f"curriculum_phase {prev_phase} → {cur_phase}"
