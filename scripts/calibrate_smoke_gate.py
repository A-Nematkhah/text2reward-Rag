"""Calibrate Stage B trajectory-bank gate thresholds (post-clip baseline)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from txt2reward.config.validation import (
    BANK_MAX_VIOLATION_RATE,
    BANK_MAX_VIOLATION_RATE_BY_PHASE,
    BANK_MIN_FITNESS_GAP,
    BANK_PASSIVE_VIOLATION_TOLERANCE_BY_PHASE,
    LITE_BANK_MAX_SOFT_VIOLATIONS,
    SMOKE_COLLISION_SEVERITY_MAX,
    TRAJECTORY_REF_FITNESS_VERSION,
    bank_max_violation_rate_for_phase,
    bank_passive_violation_tolerance_for_phase,
)
from txt2reward.llm.prompts import DEFAULT_BOOTSTRAP_REWARD_BODY
from txt2reward.llm.validation import (
    _full_validation_pipeline,
    format_smoke_gate_failure_report,
    reset_smoke_gate_failure_counts,
    smoke_gate_failure_counts,
)
from txt2reward.sandbox.sandbox import compile_reward_function
from txt2reward.trajectory.bank import (
    build_trajectory_bank,
    build_trajectory_bank_lite,
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


def _measure(fn, bank):
    return measure_gate_stats(fn, bank=bank, min_fitness_gap=BANK_MIN_FITNESS_GAP)


def main() -> None:
    reset_smoke_gate_failure_counts()
    full = build_trajectory_bank()
    lite = build_trajectory_bank_lite()
    bootstrap_fn = compile_reward_function(DEFAULT_BOOTSTRAP_REWARD_BODY.strip())
    bootstrap_full = _measure(bootstrap_fn, full)
    bootstrap_lite = _measure(bootstrap_fn, lite)

    print(f"ref_fitness_version={TRAJECTORY_REF_FITNESS_VERSION}")
    print(f"collision_severity_max={SMOKE_COLLISION_SEVERITY_MAX}")
    print(f"full soft threshold={BANK_MAX_VIOLATION_RATE:.1%}  min_gap={BANK_MIN_FITNESS_GAP}")
    print(f"phase soft rates={BANK_MAX_VIOLATION_RATE_BY_PHASE}")
    print(f"phase passive tolerance={BANK_PASSIVE_VIOLATION_TOLERANCE_BY_PHASE}")
    print(f"lite soft cap={LITE_BANK_MAX_SOFT_VIOLATIONS}")
    print(
        f"bootstrap full: soft={bootstrap_full.soft_violation_rate:.1%} "
        f"passive={bootstrap_full.passive_violations} hard={bootstrap_full.hard_violations}"
    )
    print(
        f"bootstrap lite: soft={bootstrap_lite.soft_violation_rate:.1%} "
        f"soft_n={bootstrap_lite.soft_violations} passive={bootstrap_lite.passive_violations}"
    )
    print()

    for name, code in CANDIDATES.items():
        fn = compile_reward_function(code.strip())
        stats = _measure(fn, full)
        ok, _, console = evaluate_consistency(
            fn,
            bank=full,
            max_violation_rate=BANK_MAX_VIOLATION_RATE,
            min_fitness_gap=BANK_MIN_FITNESS_GAP,
        )
        pipeline_ok, _, _ = _full_validation_pipeline(code.strip())
        print(
            f"{name:22} soft={stats.soft_violation_rate:5.1%} "
            f"passive={stats.passive_violations} hard={stats.hard_violations} "
            f"stage_b={ok} pipeline={pipeline_ok} | {console}"
        )

    gate_counts = smoke_gate_failure_counts()
    gate_report = format_smoke_gate_failure_report(gate_counts)
    print()
    print(gate_report)

    baseline = {
        "phase": "phase4-post-clip-gates",
        "ref_fitness_version": TRAJECTORY_REF_FITNESS_VERSION,
        "smoke_collision_severity_max": SMOKE_COLLISION_SEVERITY_MAX,
        "bank_max_violation_rate": BANK_MAX_VIOLATION_RATE,
        "bank_max_violation_rate_by_phase": BANK_MAX_VIOLATION_RATE_BY_PHASE,
        "bank_passive_violation_tolerance_by_phase": BANK_PASSIVE_VIOLATION_TOLERANCE_BY_PHASE,
        "lite_bank_max_soft_violations": LITE_BANK_MAX_SOFT_VIOLATIONS,
        "bootstrap_full": {
            "soft_violation_rate": bootstrap_full.soft_violation_rate,
            "soft_violations": bootstrap_full.soft_violations,
            "passive_violations": bootstrap_full.passive_violations,
            "hard_violations": bootstrap_full.hard_violations,
        },
        "bootstrap_lite": {
            "soft_violation_rate": bootstrap_lite.soft_violation_rate,
            "soft_violations": bootstrap_lite.soft_violations,
            "passive_violations": bootstrap_lite.passive_violations,
            "hard_violations": bootstrap_lite.hard_violations,
        },
        "survive_phase_rates": {
            phase: bank_max_violation_rate_for_phase(phase)
            for phase in ("survive", "speed", "overtake", "refine")
        },
        "survive_phase_passive_tolerance": {
            phase: bank_passive_violation_tolerance_for_phase(phase)
            for phase in ("survive", "speed", "overtake", "refine")
        },
        "smoke_gate_failure_counts": gate_counts,
    }
    out = ROOT / "docs" / "baselines" / "phase4-post-clip-gates.json"
    out.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
