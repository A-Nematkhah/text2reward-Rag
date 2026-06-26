"""Task 4 — evolution robustness: smoke gate calibration and Stage B behaviour."""

from reward_archive import FITNESS_VERSION_DEFAULT
from reward_designer import (
    DEFAULT_BOOTSTRAP_REWARD_BODY,
    _full_validation_pipeline,
)
from reward_sandbox import compile_reward_function
from trajectory_bank import (
    BANK_MAX_VIOLATION_RATE,
    BANK_MIN_FITNESS_GAP,
    TRAJECTORY_REF_FITNESS_VERSION,
    build_trajectory_bank,
    evaluate_consistency,
    measure_gate_stats,
)

_GEN3_WEAK_COLLISION = """
def compute_reward(state):
    if state["collided"]:
        return -30.0
    r = 0.15 * state["speed_ms"]
    if state["overtook"]:
        r += 3.0
    return r
"""

_PASSIVE_SAFE_GAP = """
def compute_reward(state):
    if state["collided"]:
        return -80.0
    r = 0.05 * state["speed_ms"]
    if state["front_dist"] > 40:
        r += 2.0
    return r
"""


def test_ref_fitness_pinned_to_v7_independent_of_archive():
    assert TRAJECTORY_REF_FITNESS_VERSION == 7
    assert FITNESS_VERSION_DEFAULT >= 8


def test_bootstrap_passes_full_validation_pipeline():
    ok, err, console = _full_validation_pipeline(DEFAULT_BOOTSTRAP_REWARD_BODY)
    assert ok, err
    assert console == "PASS"


def test_bootstrap_soft_violation_rate_below_configured_threshold():
    bank = build_trajectory_bank()
    fn = compile_reward_function(DEFAULT_BOOTSTRAP_REWARD_BODY)
    stats = measure_gate_stats(fn, bank=bank, min_fitness_gap=BANK_MIN_FITNESS_GAP)
    assert stats.passive_violations == 0
    assert stats.hard_violations == 0
    assert stats.soft_violation_rate < BANK_MAX_VIOLATION_RATE


def test_bootstrap_fails_if_threshold_tightened_below_measured_rate():
    bank = build_trajectory_bank()
    fn = compile_reward_function(DEFAULT_BOOTSTRAP_REWARD_BODY)
    stats = measure_gate_stats(fn, bank=bank, min_fitness_gap=BANK_MIN_FITNESS_GAP)
    tight = max(0.0, stats.soft_violation_rate - 0.001)
    ok, _, _ = evaluate_consistency(
        fn,
        bank=bank,
        max_violation_rate=tight,
        min_fitness_gap=BANK_MIN_FITNESS_GAP,
    )
    assert not ok


def test_gen3_style_reward_fails_stage_a_before_stage_b():
    ok, err, _ = _full_validation_pipeline(_GEN3_WEAK_COLLISION)
    assert not ok
    assert "collision" in err.lower() or "crash" in err.lower() or "farming" in err.lower()


def test_passive_safe_gap_fails_passive_hard_check():
    bank = build_trajectory_bank()
    fn = compile_reward_function(_PASSIVE_SAFE_GAP)
    stats = measure_gate_stats(fn, bank=bank, min_fitness_gap=BANK_MIN_FITNESS_GAP)
    ok, report, _ = evaluate_consistency(
        fn,
        bank=bank,
        max_violation_rate=1.0,
        min_fitness_gap=BANK_MIN_FITNESS_GAP,
    )
    assert stats.passive_violations > 0 or not ok
    assert "passive" in report.lower()


def test_min_fitness_gap_excludes_near_ties_from_decisive_pairs():
    bank = build_trajectory_bank()
    fn = compile_reward_function(DEFAULT_BOOTSTRAP_REWARD_BODY)
    loose = measure_gate_stats(fn, bank=bank, min_fitness_gap=0.0)
    strict = measure_gate_stats(fn, bank=bank, min_fitness_gap=0.20)
    assert strict.decisive_pairs <= loose.decisive_pairs
