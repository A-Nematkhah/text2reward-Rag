"""Smoke-test pipeline, bootstrap incentives, and trajectory-bank gates."""

import pytest
from txt2reward.archive.archive import FITNESS_VERSION_DEFAULT
from txt2reward.llm.designer import (
    DEFAULT_BOOTSTRAP_REWARD_BODY,
    _full_validation_pipeline,
)
from txt2reward.llm.prompts import DEFAULT_BOOTSTRAP_REWARD_BODY as BOOTSTRAP
from txt2reward.llm.validation import _smoke_test_reward_code, validate_reward_for_use
from txt2reward.sandbox.sandbox import compile_reward_function, execute_reward, validate_reward_code
from txt2reward.trajectory.bank import (
    BANK_MAX_VIOLATION_RATE,
    BANK_MIN_FITNESS_GAP,
    TRAJECTORY_REF_FITNESS_VERSION,
    _cumulative_return,
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

_WEAK_COLLISION_SMOKE = (
    "def compute_reward(state):\n"
    "    if state['collided']:\n"
    "        return -30.0\n"
    "    return clip(state['speed_ms'] * 0.15, 0.0, 5.0)\n"
)

_BINARY_CRUISE_TAX = """
def compute_reward(state):
    if state["collided"]:
        return -80.0
    speed = state["speed_ms"]
    r = clip(speed * 0.09, 0.0, 3.0)
    if state["front_dist"] > 35.0 and state["ttc"] > 5.0 and not state["overtook"]:
        if speed > 22.0:
            r += -1.8
        if not state["lane_changed"]:
            r += -0.85
    return r
"""

_ABSOLUTE_ACCEL_TRAP = """
def compute_reward(state):
    if state["collided"]:
        return -80.0
    r = clip(state["speed_ms"] * 0.09, 0.0, 3.0)
    r += -2.0 * abs(state["accel_ms2"])
    r += -1.5 * (abs(state["long_jerk"]) + abs(state["lat_jerk"]))
    return r
"""


def _episodic_fast_crash_total(reward_fn, *, n_pre_crash: int = 39) -> float:
    fast_step = {
        "speed_ms": 29.0,
        "front_dist": 55.0,
        "ttc": 12.0,
        "rel_vel_ms": 0.0,
        "lane": 1,
        "overtook": False,
        "lane_changed": False,
        "collided": False,
        "nearby_vehicles": 2,
        "accel_ms2": 0.5,
        "long_jerk": 0.3,
        "lat_jerk": 0.0,
    }
    crash_step = dict(fast_step, collided=True, front_dist=0.0, ttc=0.0)
    total = sum(float(execute_reward("", fast_step, compiled_fn=reward_fn)) for _ in range(n_pre_crash))
    return total + float(execute_reward("", crash_step, compiled_fn=reward_fn))


def _episodic_cautious_total(reward_fn, *, steps: int = 40) -> float:
    safe_step = {
        "speed_ms": 14.0,
        "front_dist": 33.0,
        "ttc": 12.0,
        "rel_vel_ms": 0.0,
        "lane": 1,
        "overtook": False,
        "lane_changed": False,
        "collided": False,
        "nearby_vehicles": 1,
        "accel_ms2": 0.0,
        "long_jerk": 0.0,
        "lat_jerk": 0.0,
    }
    return sum(float(execute_reward("", safe_step, compiled_fn=reward_fn)) for _ in range(steps))


# ── Full pipeline ─────────────────────────────────────────────────────────────


def test_bootstrap_passes_full_validation_pipeline():
    ok, err, console = _full_validation_pipeline(DEFAULT_BOOTSTRAP_REWARD_BODY)
    assert ok, err
    assert console == "PASS"


def test_ref_fitness_pinned_to_v7_independent_of_archive():
    assert TRAJECTORY_REF_FITNESS_VERSION == 7
    assert FITNESS_VERSION_DEFAULT >= 8


# ── Bootstrap reward incentives ───────────────────────────────────────────────


def test_bootstrap_collision_penalty_is_strong():
    fn = compile_reward_function(BOOTSTRAP)
    collided = {
        "speed_ms": 30.0,
        "front_dist": 0.0,
        "ttc": 0.0,
        "rel_vel_ms": 0.0,
        "lane": 1,
        "overtook": False,
        "lane_changed": False,
        "collided": True,
        "nearby_vehicles": 2,
        "accel_ms2": 0.0,
        "long_jerk": 0.0,
        "lat_jerk": 0.0,
    }
    assert execute_reward("", collided, compiled_fn=fn) <= -40.0


def test_bootstrap_crash_farming_not_profitable():
    fn = compile_reward_function(BOOTSTRAP)
    fast_crash = _episodic_fast_crash_total(fn)
    cautious = _episodic_cautious_total(fn)
    assert fast_crash < cautious
    assert fast_crash < 0.0


def test_bootstrap_jerk_spam_scores_below_safe_fast():
    fn = compile_reward_function(BOOTSTRAP)
    bank = build_trajectory_bank()
    safe = [s for s in bank if s.category == "safe_fast"]
    jerk = [s for s in bank if s.category == "jerk_accel_spam"]
    safe_mean = sum(_cumulative_return(fn, s.states) for s in safe) / len(safe)
    jerk_mean = sum(_cumulative_return(fn, s.states) for s in jerk) / len(jerk)
    assert safe_mean > jerk_mean


# ── Stage A smoke gates ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "code,err_fragment",
    [
        (_WEAK_COLLISION_SMOKE, "collision"),
        (_GEN3_WEAK_COLLISION, "collision"),
    ],
)
def test_weak_collision_rewards_rejected_by_smoke_gate(code, err_fragment):
    ok, err = _smoke_test_reward_code(code)
    assert not ok
    assert err_fragment in err.lower() or "crash" in err.lower() or "farming" in err.lower()


def test_gen3_style_reward_fails_stage_a_before_stage_b():
    ok, err, _ = _full_validation_pipeline(_GEN3_WEAK_COLLISION)
    assert not ok
    assert "collision" in err.lower() or "crash" in err.lower() or "farming" in err.lower()


def test_collision_gate_does_not_require_fixed_gap_for_negative_normal():
    normal_r, collision_r = -12.0, -30.0
    assert collision_r < normal_r
    assert collision_r <= -10.0
    assert collision_r >= (normal_r - 20.0)


def test_validate_rejects_dynamic_pow_exponent():
    code = 'def compute_reward(state):\n    return state["speed_ms"] ** round(state["front_dist"])\n'
    ok, err = validate_reward_code(code)
    assert not ok
    assert "dynamic exponents" in err.lower() or "constant literal" in err.lower()


def test_smoke_test_uses_execute_reward_path():
    code = (
        "def compute_reward(state):\n"
        "    if state['collided']:\n"
        "        return -80.0\n"
        "    return float(state['speed_ms']) * 0.09\n"
    )
    ok, err = _smoke_test_reward_code(code)
    assert ok, err


def test_speed_gradient_gate_rejects_binary_cruise_tax():
    ok, err = _smoke_test_reward_code(_BINARY_CRUISE_TAX)
    assert not ok
    assert "speed gradient" in err.lower()


def test_acceleration_roi_gate_rejects_absolute_accel_penalty():
    ok, err = _smoke_test_reward_code(_ABSOLUTE_ACCEL_TRAP)
    assert not ok
    assert "acceleration roi" in err.lower()


def test_gen27_reward_passes_full_validation():
    from pathlib import Path

    code = Path("reward_program.py").read_text(encoding="utf-8")
    ok, err = validate_reward_for_use(code)
    assert ok, err


# ── Stage B trajectory bank ───────────────────────────────────────────────────


def test_bootstrap_soft_violation_rate_below_configured_threshold():
    bank = build_trajectory_bank()
    fn = compile_reward_function(BOOTSTRAP)
    stats = measure_gate_stats(fn, bank=bank, min_fitness_gap=BANK_MIN_FITNESS_GAP)
    assert stats.passive_violations == 0
    assert stats.hard_violations == 0
    assert stats.soft_violation_rate < BANK_MAX_VIOLATION_RATE


def test_bootstrap_fails_if_threshold_tightened_below_measured_rate():
    bank = build_trajectory_bank()
    fn = compile_reward_function(BOOTSTRAP)
    stats = measure_gate_stats(fn, bank=bank, min_fitness_gap=BANK_MIN_FITNESS_GAP)
    tight = max(0.0, stats.soft_violation_rate - 0.001)
    ok, _, _ = evaluate_consistency(fn, bank=bank, max_violation_rate=tight, min_fitness_gap=BANK_MIN_FITNESS_GAP)
    assert not ok


def test_passive_safe_gap_fails_passive_hard_check():
    bank = build_trajectory_bank()
    fn = compile_reward_function(_PASSIVE_SAFE_GAP)
    stats = measure_gate_stats(fn, bank=bank, min_fitness_gap=BANK_MIN_FITNESS_GAP)
    ok, report, _ = evaluate_consistency(fn, bank=bank, max_violation_rate=1.0, min_fitness_gap=BANK_MIN_FITNESS_GAP)
    assert stats.passive_violations > 0 or not ok
    assert "passive" in report.lower()


def test_min_fitness_gap_excludes_near_ties_from_decisive_pairs():
    bank = build_trajectory_bank()
    fn = compile_reward_function(BOOTSTRAP)
    loose = measure_gate_stats(fn, bank=bank, min_fitness_gap=0.0)
    strict = measure_gate_stats(fn, bank=bank, min_fitness_gap=0.20)
    assert strict.decisive_pairs <= loose.decisive_pairs


def test_crash_rewarding_function_fails_hard_check():
    bank = build_trajectory_bank()

    def bad(state):
        return 100.0 if state["collided"] else 0.0

    ok, report, _ = evaluate_consistency(bad, bank=bank, max_violation_rate=1.0)
    assert not ok
    assert "hard safety violations" in report


def test_bank_max_violation_rate_looser_in_survive_phase():
    from txt2reward.config.validation import (
        BANK_MAX_VIOLATION_RATE,
        bank_max_violation_rate_for_phase,
    )

    assert bank_max_violation_rate_for_phase("survive") > BANK_MAX_VIOLATION_RATE
    assert bank_max_violation_rate_for_phase("refine") == BANK_MAX_VIOLATION_RATE
    assert bank_max_violation_rate_for_phase(None) == BANK_MAX_VIOLATION_RATE


def test_smoke_gate_failure_counts_increment():
    from txt2reward.llm.validation import (
        record_smoke_gate_failure,
        smoke_gate_failure_counts,
    )

    before = dict(smoke_gate_failure_counts())
    record_smoke_gate_failure("stage_a_test_gate")
    after = smoke_gate_failure_counts()
    assert after.get("stage_a_test_gate", 0) >= before.get("stage_a_test_gate", 0) + 1
