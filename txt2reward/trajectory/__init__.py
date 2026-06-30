"""Synthetic trajectory bank for reward-hacking detection."""

from txt2reward.trajectory.bank import (
    BANK_MAX_VIOLATION_RATE,
    BANK_MIN_FITNESS_GAP,
    TRAJECTORY_REF_FITNESS_VERSION,
    build_trajectory_bank,
    build_trajectory_bank_lite,
    evaluate_consistency,
    get_trajectory_bank,
    measure_gate_stats,
)

__all__ = [
    "BANK_MAX_VIOLATION_RATE",
    "BANK_MIN_FITNESS_GAP",
    "TRAJECTORY_REF_FITNESS_VERSION",
    "build_trajectory_bank",
    "build_trajectory_bank_lite",
    "evaluate_consistency",
    "get_trajectory_bank",
    "measure_gate_stats",
]
