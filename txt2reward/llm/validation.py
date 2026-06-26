"""Smoke-test and full validation pipeline for generated reward code.

Public API:
    ``validate_reward_for_use`` — AST + Stage A/B gates (archive restore, eval).
    ``write_validated_reward_tempfile`` — validate then write a temp ``.py`` file.

Internal helpers (``_smoke_test_*``, ``_full_validation_pipeline``) run the
same gates used during LLM repair in ``RewardDesigner``.
"""

from __future__ import annotations

import os
import tempfile

from txt2reward.config.validation import (
    BANK_MAX_VIOLATION_RATE,
    BANK_MIN_FITNESS_GAP,
    SMOKE_TEST_TIMEOUT_SEC,
)
from txt2reward.core.types import RewardFn
from txt2reward.sandbox.sandbox import (
    compile_reward_function,
    execute_reward,
    validate_reward_code,
)
from txt2reward.trajectory.bank import (
    build_trajectory_bank,
    evaluate_consistency,
)

_TRAJECTORY_BANK = build_trajectory_bank()


# ── Smoke-test helper ─────────────────────────────────────────────────────────

# Two representative sample states:
#   _SAMPLE_STATE_NORMAL   — typical mid-episode state, no crash, no overtake
#   _SAMPLE_STATE_COLLIDED — collision state to exercise the penalty branch
_SAMPLE_STATE_NORMAL: dict = {
    "speed_ms": 20.0,
    "front_dist": 40.0,
    "ttc": 10.0,
    "rel_vel_ms": -2.0,
    "lane": 1,
    "overtook": False,
    "lane_changed": False,
    "collided": False,
    "nearby_vehicles": 2,
    "accel_ms2": 0.5,
    "long_jerk": 0.1,
    "lat_jerk": 0.0,
}

_SAMPLE_STATE_OVERTAKE: dict = {
    "speed_ms": 28.0,
    "front_dist": 60.0,
    "ttc": 20.0,
    "rel_vel_ms": 5.0,
    "lane": 2,
    "overtook": True,
    "lane_changed": True,
    "collided": False,
    "nearby_vehicles": 1,
    "accel_ms2": 1.2,
    "long_jerk": 0.3,
    "lat_jerk": 0.2,
}

# Speed incentive gate: identical safety conditions, only speed differs.
# Gate 2b requires that fast_safe gives strictly more reward than slow_safe.
_SAMPLE_STATE_FAST_SAFE: dict = {
    "speed_ms": 28.0,
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

_SAMPLE_STATE_SLOW_SAFE: dict = {
    "speed_ms": 14.0,  # clearly below any acceptable minimum
    "front_dist": 33.0,  # identical — moderate traffic, not open-road cruising
    "ttc": 12.0,  # identical
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

_SAMPLE_STATE_COLLIDED: dict = {
    "speed_ms": 15.0,
    "front_dist": 0.0,
    "ttc": 0.0,
    "rel_vel_ms": -10.0,
    "lane": 0,
    "overtook": False,
    "lane_changed": False,
    "collided": True,
    "nearby_vehicles": 3,
    "accel_ms2": -8.0,
    "long_jerk": -5.0,
    "lat_jerk": 0.5,
}


def _smoke_sample_key_error(name: str, exc: KeyError) -> tuple[bool, str]:
    key = str(exc)
    valid_keys = ", ".join(sorted(_SAMPLE_STATE_NORMAL.keys()))
    return False, (
        f"Runtime error on sample state '{name}': KeyError {key} — "
        f"this key does not exist in the state dict. "
        f"Valid keys are: {valid_keys}"
    )


def _smoke_timeout_message(context: str, exc: BaseException) -> str:
    if context.startswith("on sample state"):
        return (
            f"Timeout {context}: {exc}. The reward function "
            "is too computationally expensive or contains a runaway "
            "expression -- simplify it (e.g. avoid large/nested exponents)."
        )
    if "speed-incentive" in context:
        return f"Timeout during speed-incentive gate: {exc}"
    return f"Timeout {context}: {exc}. Simplify the reward function -- it is too computationally expensive."


def _smoke_runtime_failure(context: str, exc: BaseException) -> tuple[bool, str]:
    """Map sandbox execution errors to smoke-test ``(ok, err)`` tuples."""
    if isinstance(exc, TimeoutError):
        return False, _smoke_timeout_message(context, exc)
    if isinstance(exc, RuntimeError) and "timed out" in str(exc).lower():
        return False, _smoke_timeout_message(context, exc)
    return False, f"Runtime error {context}: {type(exc).__name__}: {exc}"


def _smoke_test_reward_code(
    code: str,
    *,
    compiled_fn: RewardFn | None = None,
) -> tuple[bool, str]:
    """
    Execute the generated compute_reward(state) against representative sample
    states to catch runtime errors that structural AST validation cannot catch.

    All runtime calls go through reward_sandbox.execute_reward() so the smoke
    test uses the same execution path as training workers.
    """
    try:
        reward_fn = compiled_fn if compiled_fn is not None else compile_reward_function(code)
    except Exception as exc:
        return False, f"Compile error: {type(exc).__name__}: {exc}"

    rewards: dict[str, float] = {}
    for name, sample in [
        ("normal", _SAMPLE_STATE_NORMAL),
        ("overtake", _SAMPLE_STATE_OVERTAKE),
        ("collision", _SAMPLE_STATE_COLLIDED),
    ]:
        try:
            result = execute_reward(
                "",
                sample,
                timeout_sec=SMOKE_TEST_TIMEOUT_SEC,
                compiled_fn=reward_fn,
            )
            rewards[name] = float(result)
        except KeyError as exc:
            return _smoke_sample_key_error(name, exc)
        except (TimeoutError, RuntimeError, Exception) as exc:
            return _smoke_runtime_failure(f"on sample state '{name}'", exc)

    # ── Gate 1b: collision branch must be strictly worse than normal driving ─
    _COLLISION_SEVERITY_MAX = -40.0
    _MIN_GAP_WHEN_NORMAL_POSITIVE = 45.0
    if "collision" in rewards and "normal" in rewards:
        normal_r = rewards["normal"]
        collision_r = rewards["collision"]
        if collision_r >= normal_r:
            return False, (
                "Safety Gate Violation: collided-state reward must be strictly lower "
                f"than normal-state reward (normal={normal_r:.2f}, collision={collision_r:.2f})."
            )
        if collision_r > _COLLISION_SEVERITY_MAX:
            return False, (
                "Safety Gate Violation: collision penalty is too weak — "
                f"collided-state reward {collision_r:.2f} must be <= "
                f"{_COLLISION_SEVERITY_MAX:.1f}."
            )
        if normal_r > 0 and (normal_r - collision_r) < _MIN_GAP_WHEN_NORMAL_POSITIVE:
            return False, (
                "Safety Gate Violation: collision penalty is suppressed by positive "
                f"normal-state bonuses (normal={normal_r:.2f}, collision={collision_r:.2f}, "
                f"required gap>={_MIN_GAP_WHEN_NORMAL_POSITIVE:.1f})."
            )

    # ── Gate 2: Full Episodic Simulation (Synthetic Trajectories) ────────
    # 1) cautious/steady safe trajectory (40 steps, no collision)
    cautious_return = 0.0
    try:
        for t in range(40):
            cautious_state = {
                "speed_ms": 18.0,
                "front_dist": 50.0,
                "ttc": 30.0,
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
            cautious_return += execute_reward(
                "",
                cautious_state,
                timeout_sec=SMOKE_TEST_TIMEOUT_SEC,
                compiled_fn=reward_fn,
            )
    except (TimeoutError, RuntimeError, Exception) as exc:
        return _smoke_runtime_failure("during cautious-trajectory smoke test", exc)

    # 2) reckless trajectory: high speed, many lane changes/overtakes, collision at final step
    reckless_return = 0.0
    try:
        for t in range(40):
            is_last_step = t == 39
            reckless_state = {
                "speed_ms": 28.0,
                "front_dist": 15.0 if not is_last_step else 0.0,
                "ttc": 2.0 if not is_last_step else 0.0,
                "rel_vel_ms": -5.0,
                "lane": t % 3,
                "overtook": True if (t % 10 == 0 and not is_last_step) else False,
                "lane_changed": True if (t % 5 == 0 and not is_last_step) else False,
                "collided": is_last_step,
                "nearby_vehicles": 4,
                "accel_ms2": 2.0 if t % 2 == 0 else -2.0,
                "long_jerk": 1.5,
                "lat_jerk": 1.0,
            }
            reckless_return += execute_reward(
                "",
                reckless_state,
                timeout_sec=SMOKE_TEST_TIMEOUT_SEC,
                compiled_fn=reward_fn,
            )
    except (TimeoutError, RuntimeError, Exception) as exc:
        return _smoke_runtime_failure("during reckless-trajectory smoke test", exc)

    # 3) Trajectory fitness condition: safe driving must achieve higher episodic return
    if reckless_return >= cautious_return:
        return False, (
            f"Trajectory Gate Violation (Reward Hacking Detected): "
            f"The reckless/crashy trajectory achieved a HIGHER episodic return ({reckless_return:.2f}) "
            f"than the safe/cautious trajectory ({cautious_return:.2f}). "
            f"The reward function is inflating speed/overtake bonuses over survival metrics."
        )

    # ── Gate 2b: Speed Incentive Gate ─────────────────────────────────────
    # At IDENTICAL safety conditions (same front_dist=40, same ttc=12, same
    # everything except speed), the reward function MUST give higher per-step
    # reward at 28 m/s than at 14 m/s. This directly tests whether the speed
    # term is positive and large enough to matter.
    fast_return = 0.0
    slow_return = 0.0
    try:
        for _ in range(20):
            fast_return += execute_reward(
                "",
                _SAMPLE_STATE_FAST_SAFE,
                timeout_sec=SMOKE_TEST_TIMEOUT_SEC,
                compiled_fn=reward_fn,
            )
            slow_return += execute_reward(
                "",
                _SAMPLE_STATE_SLOW_SAFE,
                timeout_sec=SMOKE_TEST_TIMEOUT_SEC,
                compiled_fn=reward_fn,
            )
    except RuntimeError as exc:
        if "timed out" in str(exc).lower():
            return False, _smoke_timeout_message("during speed-incentive gate", exc)
        return False, f"Runtime error during speed-incentive gate: {type(exc).__name__}: {exc}"
    except Exception as exc:
        return False, f"Runtime error during speed-incentive gate: {type(exc).__name__}: {exc}"

    if fast_return <= slow_return:
        return False, (
            f"Speed Incentive Gate Violation: a 28 m/s safe trajectory scored "
            f"{fast_return:.2f} total reward over 20 steps, but a 14 m/s safe "
            f"trajectory (identical front_dist=40, ttc=12, no collision, no "
            f"overtake) scored {slow_return:.2f}. The reward function does not "
            f"incentivise driving faster — the agent will learn to crawl. "
            f"Ensure the speed_ms term has a positive coefficient large enough "
            f"that higher speed always produces strictly more per-step reward "
            f"under safe conditions. Avoid letting safe_gap / front_dist bonuses "
            f"dominate so heavily that slowing down (which increases front_dist "
            f"in practice) becomes the optimal strategy."
        )

    # ── Gate 2c: Crash-farming gate ─────────────────────────────────────────
    # Simulate "drive fast then crash" vs a full cautious episode (~40 steps).
    fast_pre_crash = 0.0
    cautious_episode = 0.0
    try:
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
        crash_step["collided"] = True
        crash_step["front_dist"] = 0.0
        crash_step["ttc"] = 0.0
        for _ in range(39):
            fast_pre_crash += execute_reward(
                "",
                fast_step,
                timeout_sec=SMOKE_TEST_TIMEOUT_SEC,
                compiled_fn=reward_fn,
            )
        fast_episode_total = fast_pre_crash + execute_reward(
            "",
            crash_step,
            timeout_sec=SMOKE_TEST_TIMEOUT_SEC,
            compiled_fn=reward_fn,
        )
        for _ in range(40):
            cautious_episode += execute_reward(
                "",
                _SAMPLE_STATE_SLOW_SAFE,
                timeout_sec=SMOKE_TEST_TIMEOUT_SEC,
                compiled_fn=reward_fn,
            )
    except Exception as exc:
        return False, f"Runtime error during crash-farming gate: {type(exc).__name__}: {exc}"

    if fast_episode_total >= cautious_episode:
        return False, (
            f"Crash-Farming Gate Violation: a 39-step fast drive plus collision "
            f"scored {fast_episode_total:.2f}, but a 40-step cautious safe episode "
            f"scored {cautious_episode:.2f}. Crashing must NOT be net-profitable — "
            f"increase |collision penalty| or reduce per-step speed bonuses."
        )

    return True, ""


def _full_validation_pipeline(code: str) -> tuple[bool, str, str]:
    """
    Two-stage smoke test, run in sequence:

      Stage A (fast)  — _smoke_test_reward_code(): 3 sample states + the
                        quantitative collision-suppression gate + the
                        original 2-trajectory (cautious vs reckless) check.
                        Cheap (~3 + 80 compute_reward calls), so it runs
                        first and rejects most broken code immediately
                        without ever touching the larger bank.

      Stage B (thorough) — trajectory_bank.evaluate_consistency(): only
                        runs if Stage A passed. Executes the candidate
                        against the full ~40-trajectory bank (8 behavioural
                        categories) and checks pairwise-ranking consistency
                        against each trajectory's independent reference
                        fitness. This is what catches reward-hacking
                        loopholes that don't show up in the single
                        cautious-vs-reckless comparison, e.g. tailgating
                        without ever crashing, lane-thrashing, or
                        accel/jerk spam with no net speed gain.

    Returns (ok, full_error_message, console_summary). On Stage B failure,
    full_error_message is the per-pair violation report for the LLM repair
    prompt; console_summary is a one-line summary for terminal logs.
    """
    try:
        reward_fn = compile_reward_function(code)
    except Exception as exc:
        msg = f"Compile error: {type(exc).__name__}: {exc}"
        return False, msg, msg

    stage_a_ok, stage_a_err = _smoke_test_reward_code(code, compiled_fn=reward_fn)
    if not stage_a_ok:
        return False, stage_a_err, stage_a_err

    def _timed_fn(state: dict):
        return execute_reward(
            "",
            state,
            timeout_sec=SMOKE_TEST_TIMEOUT_SEC,
            compiled_fn=reward_fn,
        )

    stage_b_ok, stage_b_report, stage_b_console = evaluate_consistency(
        _timed_fn,
        bank=_TRAJECTORY_BANK,
        max_violation_rate=BANK_MAX_VIOLATION_RATE,
        min_fitness_gap=BANK_MIN_FITNESS_GAP,
    )
    if not stage_b_ok:
        return False, stage_b_report, stage_b_console

    return True, "", "PASS"


def validate_reward_for_use(code: str) -> tuple[bool, str]:
    """
    Structural AST validation plus full smoke-test pipeline.

    Returns ``(ok, error_message)``. Used when restoring or evaluating archived
    reward programs so every path shares the same gate as fresh LLM output.
    """
    ok, err = validate_reward_code(code)
    if not ok:
        return False, err
    smoke_ok, smoke_err, _ = _full_validation_pipeline(code)
    if not smoke_ok:
        return False, smoke_err
    return True, ""


def write_validated_reward_tempfile(code: str, generation: int) -> str:
    """
    Validate reward code and write it to a temporary file for evaluation.

    Returns the path to the temp file. Raises ``ValueError`` when validation fails.
    """
    ok, err = validate_reward_for_use(code)
    if not ok:
        raise ValueError(f"Generation {generation} failed validation: {err}")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=f"_reward_gen{generation}.py", prefix="txt2reward_")
    os.close(tmp_fd)
    try:
        os.chmod(tmp_path, 0o600)
    except OSError:
        pass  # Windows may not support Unix permission bits
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(code)
    return tmp_path
