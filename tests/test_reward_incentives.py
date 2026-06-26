"""Task 2 — PPO incentive alignment: crash/speed/jerk/lane-thrash guards."""

from reward_designer import (
    DEFAULT_BOOTSTRAP_REWARD_BODY,
    _full_validation_pipeline,
    _smoke_test_reward_code,
)
from reward_sandbox import compile_reward_function, execute_reward
from trajectory_bank import _cumulative_return, build_trajectory_bank


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
    crash_step = dict(fast_step)
    crash_step.update(collided=True, front_dist=0.0, ttc=0.0)
    total = sum(
        float(execute_reward("", fast_step, compiled_fn=reward_fn))
        for _ in range(n_pre_crash)
    )
    total += float(execute_reward("", crash_step, compiled_fn=reward_fn))
    return total


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
    return sum(
        float(execute_reward("", safe_step, compiled_fn=reward_fn)) for _ in range(steps)
    )


def test_bootstrap_collision_penalty_is_strong():
    fn = compile_reward_function(DEFAULT_BOOTSTRAP_REWARD_BODY)
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
    fn = compile_reward_function(DEFAULT_BOOTSTRAP_REWARD_BODY)
    fast_crash = _episodic_fast_crash_total(fn)
    cautious = _episodic_cautious_total(fn)
    assert fast_crash < cautious
    assert fast_crash < 0.0


def test_weak_collision_reward_rejected_by_smoke_test():
    code = (
        "def compute_reward(state):\n"
        "    if state['collided']:\n"
        "        return -30.0\n"
        "    return clip(state['speed_ms'] * 0.15, 0.0, 5.0)\n"
    )
    ok, err = _smoke_test_reward_code(code)
    assert not ok
    assert "Crash-Farming" in err or "collision penalty" in err.lower()


def test_gen3_style_reward_fails_crash_farming_gate():
    """Reproduce the failure mode from the 100% crash training run."""
    code = (
        "def compute_reward(state):\n"
        "    if state['collided']:\n"
        "        return -30.0\n"
        "    speed = state['speed_ms']\n"
        "    return clip(speed * 0.15, 0.0, 5.0) + (4.0 if state['overtook'] else 0.0)\n"
    )
    ok, err = _smoke_test_reward_code(code)
    assert not ok


def test_bootstrap_jerk_spam_scores_below_safe_fast():
    fn = compile_reward_function(DEFAULT_BOOTSTRAP_REWARD_BODY)
    bank = build_trajectory_bank()
    safe = [s for s in bank if s.category == "safe_fast"]
    jerk = [s for s in bank if s.category == "jerk_accel_spam"]
    safe_mean = sum(_cumulative_return(fn, s.states) for s in safe) / len(safe)
    jerk_mean = sum(_cumulative_return(fn, s.states) for s in jerk) / len(jerk)
    assert safe_mean > jerk_mean


def test_bootstrap_passes_full_validation_pipeline():
    ok, err, _ = _full_validation_pipeline(DEFAULT_BOOTSTRAP_REWARD_BODY)
    assert ok, err
