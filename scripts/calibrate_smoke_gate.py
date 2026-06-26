"""Calibrate Stage B trajectory-bank gate thresholds (Task 4 analysis)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from txt2reward.llm.prompts import DEFAULT_BOOTSTRAP_REWARD_BODY
from txt2reward.llm.validation import _full_validation_pipeline
from txt2reward.sandbox.sandbox import compile_reward_function
from txt2reward.trajectory.bank import (  # noqa: E402
    BANK_MAX_VIOLATION_RATE,
    BANK_MIN_FITNESS_GAP,
    TRAJECTORY_REF_FITNESS_VERSION,
    build_trajectory_bank,
    evaluate_consistency,
    measure_gate_stats,
)

CANDIDATES = {
    "bootstrap": DEFAULT_BOOTSTRAP_REWARD_BODY,
    "gen3_weak_collision": """
def compute_reward(state):
    if state["collided"]:
        return -30.0
    r = 0.15 * state["speed_ms"]
    if state["overtook"]:
        r += 3.0
    return r
""",
    "passive_safe_gap": """
def compute_reward(state):
    if state["collided"]:
        return -80.0
    r = 0.05 * state["speed_ms"]
    if state["front_dist"] > 40:
        r += 2.0
    return r
""",
}


def main() -> None:
    bank = build_trajectory_bank()
    print(f"ref_fitness_version={TRAJECTORY_REF_FITNESS_VERSION}")
    print(f"configured threshold={BANK_MAX_VIOLATION_RATE:.1%}  min_gap={BANK_MIN_FITNESS_GAP}")
    print()

    for name, code in CANDIDATES.items():
        fn = compile_reward_function(code.strip())
        stats = measure_gate_stats(fn, bank=bank, min_fitness_gap=BANK_MIN_FITNESS_GAP)
        ok, report, console = evaluate_consistency(
            fn,
            bank=bank,
            max_violation_rate=BANK_MAX_VIOLATION_RATE,
            min_fitness_gap=BANK_MIN_FITNESS_GAP,
        )
        pipeline_ok, _, _ = _full_validation_pipeline(code.strip())
        print(
            f"{name:22} soft={stats.soft_violation_rate:5.1%} "
            f"passive={stats.passive_violations} hard={stats.hard_violations} "
            f"stage_b={ok} pipeline={pipeline_ok} | {console}"
        )


if __name__ == "__main__":
    main()
