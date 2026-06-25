"""
reward_designer.py
──────────────────
LLM-driven Text-to-Reward pipeline for highway-env PPO training.

Architecture
────────────
Natural Language Goal
        ↓
  [LLM + RAG context from archive]
        ↓
  Generated reward_program.py (Python source)
        ↓
  [Sandbox validation: AST check + type check + SMOKE TEST execution]
        ↓
  PPO Training (uses LLMRewardWrapper → reward_program.py)
        ↓
  Evaluation → metrics
        ↓
  [LLM Critique: detect hacking, propose improvements]
        ↓
  archive.add_entry(code, metrics, critique)
        ↓
  Next generation (loop)

Key changes from weight-based system
─────────────────────────────────────
  OLD: LLM → reward_weights.json → compute_shaped_reward(weights, ...)
  NEW: LLM → reward_program.py  → compute_reward(state)

  The LLM now generates a COMPLETE reward function, not just scalar weights.
  The archive provides RAG-style memory of the best programs and their metrics.
  A critique phase after each evaluation sends diagnostics back to the LLM.

Reward hacking detection
────────────────────────
The critique prompt explicitly asks the LLM to look for:
  * oscillatory lane changes (lane_changed high but overtakes low)
  * acceleration spam (accel/jerk high without speed gain)
  * brake-acceleration exploits (alternating accel cycles)
  * stationary behaviour (mean_speed very low)
  * reward farming loops (shaped_reward high but env_reward low)

Smoke-test validation (TWO STAGES)
────────────────────────────────────
Stage A (fast): the original two-sample + collision-gap + 2-trajectory check.
Cheap, runs on every attempt, catches structural/runtime breakage fast
(KeyError on a wrong state key, suppressed collision penalty, etc.).

Stage B (thorough): only runs once Stage A passes. Executes the candidate
against the full ~40-trajectory bank (trajectory_bank.py) spanning 8
behavioural categories, and checks pairwise-ranking consistency against
each trajectory's independent reference fitness (reward_archive.compute_fitness
applied to the trajectory's own aggregate metrics — computed completely
independently of the candidate reward function). This catches reward-hacking
loopholes that a single cautious-vs-reckless comparison cannot, such as
tailgating-without-crashing or lane-thrashing. Failures are fed back through
the same repair loop as Stage A, with a per-pair violation report.
"""

from __future__ import annotations

import os
import re
import time
import math
from collections import deque
from typing import Deque

from groq import Groq

from reward_sandbox import validate_reward_code, extract_reward_body, execute_reward, compile_reward_function
from reward_archive import RewardArchive
from trajectory_bank import build_trajectory_bank, evaluate_consistency
from key_manager import call_with_rotation  # ← چرخش خودکار کلید

# ── Groq client ───────────────────────────────────────────────────────────────
# _client دیگه استفاده نمیشه — key_manager مدیریت می‌کنه
_client: Groq | None = None
MODEL = "llama-3.3-70b-versatile"

REWARD_PROGRAM_PATH = "reward_program.py"

# Shipped bootstrap body (no header) — used when LLM bootstrap fails and disk is empty.
DEFAULT_BOOTSTRAP_REWARD_BODY = """\
def compute_reward(state):
    if state["collided"]:
        return -30.0
    speed = state["speed_ms"]
    clear_road = state["front_dist"] > 40.0 and state["ttc"] > 5.0
    speed_reward = clip(speed * 0.12, 0.0, 4.0)
    slow_penalty = -0.6 if clear_road and speed < 22.0 else 0.0
    cruise_tax = -2.0 if clear_road and not state["overtook"] and speed > 22.0 else 0.0
    no_overtake_tax = (
        -1.1
        if clear_road and not state["overtook"] and not state["lane_changed"]
        else (-0.45 if clear_road and not state["overtook"] else 0.0)
    )
    ttc_penalty = -3.0 if state["ttc"] < 1.0 else -1.5 if state["ttc"] < 3.0 else 0.0
    tailgate_penalty = -1.2 if state["front_dist"] < 20.0 and state["ttc"] < 4.0 else 0.0
    overtake_bonus = 3.0 if state["overtook"] else 0.0
    jerk_penalty = -0.08 * (abs(state["long_jerk"]) + abs(state["lat_jerk"]))
    accel_penalty = -0.05 * abs(state["accel_ms2"])
    gap_bonus = 0.008 * clip(state["front_dist"] - 25.0, 0.0, 15.0) if speed >= 22.0 else 0.0
    lc_penalty = -0.35 if state["lane_changed"] and not state["overtook"] else 0.0
    return (
        speed_reward
        + slow_penalty
        + cruise_tax
        + no_overtake_tax
        + ttc_penalty
        + tailgate_penalty
        + overtake_bonus
        + jerk_penalty
        + accel_penalty
        + gap_bonus
        + lc_penalty
    )
"""


def _is_placeholder_code(code: str) -> bool:
    """True for missing, empty, or non-actionable archived reward sources."""
    if not code or not code.strip():
        return True
    if "placeholder" in code:
        return True
    return False


def write_default_reward_program(path: str = REWARD_PROGRAM_PATH) -> None:
    """Writes the shipped bootstrap reward program to disk."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(
            '"""\nreward_program.py -- Generation 0 (bootstrap default)\n'
            "Auto-generated by RewardDesigner. DO NOT EDIT MANUALLY.\n\"\"\"\n\n"
        )
        f.write(DEFAULT_BOOTSTRAP_REWARD_BODY)
    os.replace(tmp, path)

# Hard wall-clock timeout for every single compute_reward() call made during
# the smoke test (fixes audit finding #3). The structural AST check already
# blocks the most obvious DoS vector (a huge literal exponent -- see
# reward_sandbox._MAX_POW_EXPONENT), but it cannot catch every pathological
# construct an LLM might still produce within the allowed grammar (e.g. a
# deeply nested chain of Pow/exp/log operations on values near the bounds of
# float range). Without a timeout here, such code would hang the MAIN
# training process (not just a worker) during the repair loop, since
# _smoke_test_reward_code previously called fn(...) directly with no time
# bound at all.
_SMOKE_TEST_TIMEOUT_SEC = 0.5

# Stage B (trajectory bank) gate parameters. Kept here rather than in
# trajectory_bank.py so the designer's repair-loop tolerance is tunable
# independently of the bank/gate module itself.
_BANK_MAX_VIOLATION_RATE = 0.12
_BANK_MIN_FITNESS_GAP = 0.06

# Built once at import time: the bank is deterministic (fixed seed), so
# there is no reason to rebuild it for every single generation/repair
# attempt. ~40 trajectories x up to a few hundred steps each is cheap to
# build but rebuilding it dozens of times per evolution step (every
# generate + every repair retry) is needless overhead.
_TRAJECTORY_BANK = build_trajectory_bank()


_STATE_SCHEMA = """\
State keys available inside compute_reward(state):
  speed_ms        : float   ego speed in m/s (range 0-40)
  front_dist      : float   distance to front vehicle [m] (0-200, 200 = clear)
  ttc             : float   time-to-collision [s] (0-30, 30 = no threat)
  rel_vel_ms      : float   v_front - v_ego [m/s] (negative = approaching)
  lane            : int     lane index, 0 = rightmost
  overtook        : bool    completed an overtake this step   ← NOTE: "overtook", NOT "overtake"
  lane_changed    : bool    lane changed since last step
  collided        : bool    collision detected
  nearby_vehicles : int     vehicles within ~30 m radius
  accel_ms2       : float   longitudinal acceleration [m/s2]
  long_jerk       : float   longitudinal jerk [m/s3]
  lat_jerk        : float   lateral jerk [m/s3]

CRITICAL: The only valid state keys are EXACTLY the ones listed above.
  * Use state["overtook"]  ← correct (past tense, with k)
  * NEVER use state["overtake"]  ← this key does NOT exist and will crash

Safe math available (no imports, just use by name):
  min, max, abs, round, float, int, bool
  sqrt, exp, log, sin, cos, tan, atan, atan2
  floor, ceil, clip(val, lo, hi), pi, e, inf
"""

_GENERATION_SYSTEM = """\
You are a reinforcement learning reward engineer for a PPO highway driving agent.

Your task: write a Python function compute_reward(state) that returns a float.
The agent drives on a 4-lane highway. Goal: high-speed, safe, efficient driving with active overtaking.

HARD RULES (violation = sandbox rejection):
    * Function signature: def compute_reward(state):
    * HARD SAFETY CHECK: If state["collided"] is True, the function MUST immediately
      return the collision penalty (e.g., -30.0) without any other positive terms,
      speed rewards, or bonuses calculated or added in that same step. You MUST use
      exactly this pattern at the very beginning of the function body:
          if state["collided"]:
              return -30.0
    * No import statements
    * No attribute access (no obj.method)
    * No loops (for/while)
    * No builtins except: min, max, abs, round, float, int, bool
    * Only approved math: sqrt, exp, log, sin, cos, tan, atan2, floor, ceil, clip, pi
    * Must return a float value
    * Single local variables allowed; no nested functions

DESIGN PRINCIPLES:
    * Collision penalty should dominate (-10 to -30) to prevent reward hacking
    * Speed reward should be continuous and always incentivise going faster (target 24-30 m/s)
    * TTC penalty should activate only below 3 s -- not for normal driving
    * Overtake bonus: large one-shot reward (+2 to +4) when overtook == True
    * Jerk/accel penalties should be small (0.01-0.05 scale) to not suppress action
    * Avoid rewarding stationary behaviour or unnecessary lane changes
    * ANTI-PASSIVE-DRIVING: zero crashes alone is NOT success. When front_dist > 40
      and ttc > 5 (clear road), penalise speed > 22 m/s without overtakes (cruise_tax)
      and speed < 22 m/s (slow_penalty). Do NOT use large safe_gap / front_dist
      bonuses — they let passive cruising farm reward without overtaking.
    * SPEED INCENTIVE TEST: the validation pipeline will execute your function at
      28 m/s and 14 m/s with identical safe conditions (front_dist=40, ttc=12,
      no collision). Your function MUST return strictly more reward at 28 m/s.
      If the gap between fast and slow reward is too small, the agent will not
      learn to drive faster. Ensure the speed term coefficient is large enough.
    * The archive fitness function applies a multiplicative passive-driving gate:
      once crash_rate is below 20%, fitness = base × speed_factor × overtake_factor,
      where speed_factor = (mean_speed/24)² and overtake_factor = (mean_overtakes/1.5)^0.5.
      An agent at 20 m/s with 0 overtakes gets gate = 0.10 — 90% fitness penalty.
      Programs that are "safe but slow" score very poorly — do not copy them.

{state_schema}

Reply ONLY with the Python source of compute_reward(state). No explanation, no markdown fences.
"""

_CRITIQUE_SYSTEM = """\
You are a reinforcement learning reward auditor. Analyse the reward function and metrics below.

Identify:
1. Reward hacking patterns:
   - Passive driving / "slow to survive": crash_rate near 0 but mean_speed < 22 m/s
     and mean_overtakes near 0 — agent crawls to avoid risk instead of driving well
   - Oscillatory lane changes: lane_changes >> overtakes (agent thrashing lanes for reward)
   - Acceleration spam: high mean_accel with low speed gain (braking-acceleration exploit)
   - Stationary farming: very low mean_speed but high shaped_reward
   - TTC exploitation: very low ttc or min_ttc but no crashes (agent riding tailgate for some bonus)
2. Missing incentives (what good behaviour is not rewarded)
3. Misaligned incentives (what bad behaviour is inadvertently rewarded)
4. Proposed improvements with SPECIFIC code changes

Be concise (max 300 words). End with 3 concrete bullet-point improvements.

IMPORTANT: At the very end of your response, append a machine-readable metadata block
on a single line in this EXACT format (no whitespace before the colon):
CRITIQUE_META:{"failure_modes":["tag1","tag2"],"strengths":["s1"],"summary":"one sentence"}

Valid failure_mode tags: tailgating, passive_driving, oscillatory_lane_changes,
acceleration_spam, stationary_farming, reward_hacking
Valid strength tags: high_speed, good_overtaking, safe_driving, smooth_driving
"""

_GENERATION_USER_TEMPLATE = """\
=== DRIVING GOAL ===
{goal}

=== ARCHIVE MEMORY ===
{archive_context}

The archive is organised into sections:
  A) Top performers — adopt their strengths
  B) Most recent — continue or fix the current trajectory
  C) Failed rewards — do NOT repeat their mistakes
  D) Similar failure modes — study why the same issues appeared before

=== TASK ===
Generate an improved compute_reward(state) function that achieves the goal above.
- Learn from top performers: adopt what scored well.
- Avoid the failure patterns shown in sections C and D.
- If a failure mode is listed (e.g. tailgating, passive_driving), explicitly add
  a term that penalises it.
- Do NOT replicate "safe but slow" rewards (0% crash, speed ~20 m/s, no overtakes).
  The fitness function now penalises this via a passive-driving gate.
- Prioritise: (1) no collisions, (2) speed >= 24 m/s when road is clear, (3) active overtaking.
Return ONLY the Python function source. No explanation, no markdown.
"""

_CRITIQUE_USER_TEMPLATE = """\
=== REWARD PROGRAM (Generation {generation}) ===
```python
{reward_code}
```

=== EVALUATION METRICS ===
  mean_speed       : {mean_speed:.2f} m/s
  crash_rate       : {crash_rate:.1%}
  mean_overtakes   : {mean_overtakes:.2f} per episode
  completion_rate  : {completion_rate:.1%}
  mean_steps       : {mean_steps:.0f}
  mean_ttc         : {mean_ttc:.2f} s
  p10_ttc          : {p10_ttc:.2f} s   (10th-percentile TTC — near-miss indicator)
  min_ttc          : {min_ttc:.2f} s   (worst single-step TTC)
  mean_long_jerk   : {mean_long_jerk:.3f} m/s3
  mean_accel       : {mean_accel:.3f} m/s2
  total_lc         : {total_lane_changes} lane changes
  fitness          : {fitness:.4f}

=== TREND VS PREVIOUS GENERATION ===
{trend_summary}

=== EPISODE TRAJECTORY SAMPLES ===
{trajectory_summary}

Identify reward hacking, failure modes, and propose 3 specific improvements.
If mean_speed or mean_overtakes is DECREASING while crash_rate also decreases,
treat this as a strong reward-hacking signal (the agent is likely slowing down
or refusing to overtake just to avoid crashing, instead of driving well) and
say so explicitly.
"""

_REPAIR_USER_TEMPLATE = """\
The reward function you generated failed validation with this error:

  {error}

Common causes:
  * Using a state key that does not exist, e.g. state["overtake"] — the correct
    key is state["overtook"] (past tense, with k). Other valid keys are:
    speed_ms, front_dist, ttc, rel_vel_ms, lane, overtook, lane_changed,
    collided, nearby_vehicles, accel_ms2, long_jerk, lat_jerk.
    DO NOT invent new key names.
  * Using a disallowed builtin or math function
  * Syntax errors, loops, or import statements

Here is the rejected code:
```python
{rejected_code}
```

Fix ALL issues and return ONLY the corrected compute_reward(state) function.
No explanation, no markdown fences.
"""

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
    "front_dist": 40.0,
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
    "speed_ms": 14.0,       # clearly below any acceptable minimum
    "front_dist": 40.0,     # identical — tests speed term in isolation
    "ttc": 12.0,            # identical
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


def _smoke_test_reward_code(code: str) -> tuple[bool, str]:
    """
    Execute the generated compute_reward(state) against representative sample
    states to catch runtime errors that structural AST validation cannot catch.

    All runtime calls go through reward_sandbox.execute_reward() so the smoke
    test uses the same execution path as training workers.
    """
    try:
        reward_fn = compile_reward_function(code)
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
                timeout_sec=_SMOKE_TEST_TIMEOUT_SEC,
                compiled_fn=reward_fn,
            )
            rewards[name] = float(result)
        except KeyError as exc:
            key = str(exc)
            valid_keys = ", ".join(sorted(_SAMPLE_STATE_NORMAL.keys()))
            return False, (
                f"Runtime error on sample state '{name}': KeyError {key} — "
                f"this key does not exist in the state dict. "
                f"Valid keys are: {valid_keys}"
            )
        except TimeoutError as exc:
            return False, (
                f"Timeout on sample state '{name}': {exc}. The reward function "
                "is too computationally expensive or contains a runaway "
                "expression -- simplify it (e.g. avoid large/nested exponents)."
            )
        except RuntimeError as exc:
            if "timed out" in str(exc).lower():
                return False, (
                    f"Timeout on sample state '{name}': {exc}. The reward function "
                    "is too computationally expensive or contains a runaway "
                    "expression -- simplify it (e.g. avoid large/nested exponents)."
                )
            return False, f"Runtime error on sample state '{name}': {type(exc).__name__}: {exc}"
        except Exception as exc:
            return False, (f"Runtime error on sample state '{name}': " f"{type(exc).__name__}: {exc}")

    # ── Gate 1b: collision branch must be strictly worse than normal driving ─
    _COLLISION_SEVERITY_MAX = -10.0
    _MIN_GAP_WHEN_NORMAL_POSITIVE = 20.0
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
                timeout_sec=_SMOKE_TEST_TIMEOUT_SEC,
                compiled_fn=reward_fn,
            )
    except TimeoutError as exc:
        return False, (
            f"Timeout during cautious-trajectory smoke test: {exc}. "
            "Simplify the reward function -- it is too computationally expensive."
        )
    except RuntimeError as exc:
        if "timed out" in str(exc).lower():
            return False, (
                f"Timeout during cautious-trajectory smoke test: {exc}. "
                "Simplify the reward function -- it is too computationally expensive."
            )
        return False, f"Runtime error during cautious-trajectory smoke test: {type(exc).__name__}: {exc}"
    except Exception as exc:
        return False, f"Runtime error during cautious-trajectory smoke test: {type(exc).__name__}: {exc}"

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
                timeout_sec=_SMOKE_TEST_TIMEOUT_SEC,
                compiled_fn=reward_fn,
            )
    except TimeoutError as exc:
        return False, (
            f"Timeout during reckless-trajectory smoke test: {exc}. "
            "Simplify the reward function -- it is too computationally expensive."
        )
    except RuntimeError as exc:
        if "timed out" in str(exc).lower():
            return False, (
                f"Timeout during reckless-trajectory smoke test: {exc}. "
                "Simplify the reward function -- it is too computationally expensive."
            )
        return False, f"Runtime error during reckless-trajectory smoke test: {type(exc).__name__}: {exc}"
    except Exception as exc:
        return False, f"Runtime error during reckless-trajectory smoke test: {type(exc).__name__}: {exc}"

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
                timeout_sec=_SMOKE_TEST_TIMEOUT_SEC,
                compiled_fn=reward_fn,
            )
            slow_return += execute_reward(
                "",
                _SAMPLE_STATE_SLOW_SAFE,
                timeout_sec=_SMOKE_TEST_TIMEOUT_SEC,
                compiled_fn=reward_fn,
            )
    except RuntimeError as exc:
        if "timed out" in str(exc).lower():
            return False, f"Timeout during speed-incentive gate: {exc}"
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
    stage_a_ok, stage_a_err = _smoke_test_reward_code(code)
    if not stage_a_ok:
        return False, stage_a_err, stage_a_err

    try:
        reward_fn = compile_reward_function(code)
    except Exception as exc:
        msg = f"Compile error during Stage B setup: {type(exc).__name__}: {exc}"
        return False, msg, msg

    def _timed_fn(state: dict):
        return execute_reward(
            "",
            state,
            timeout_sec=_SMOKE_TEST_TIMEOUT_SEC,
            compiled_fn=reward_fn,
        )

    stage_b_ok, stage_b_report, stage_b_console = evaluate_consistency(
        _timed_fn,
        bank=_TRAJECTORY_BANK,
        max_violation_rate=_BANK_MAX_VIOLATION_RATE,
        min_fitness_gap=_BANK_MIN_FITNESS_GAP,
    )
    if not stage_b_ok:
        return False, stage_b_report, stage_b_console

    return True, "", "PASS"


# ── Groq client ───────────────────────────────────────────────────────────────


def _get_client() -> Groq:
    """Deprecated: از call_with_rotation استفاده کنید."""
    from key_manager import get_groq_client

    return get_groq_client()


class RewardDesigner:
    """
    Text-to-Reward evolutionary loop.

    Responsibilities
    ----------------
    1. generate_reward(goal)   -> LLM -> validated Python -> reward_program.py
    2. record_episode(stats)   -> accumulate behaviour metrics
    3. maybe_evolve()          -> every N episodes: critique + generate new reward
    4. get_policy_snapshot()   -> averaged PPO health metrics for LLM context
    5. push_policy_metrics()   -> called by SB3 callback after each PPO update
    """

    def __init__(
        self,
        goal: str = "Drive fast, overtake slow vehicles, avoid collisions.",
        evolve_every: int = 20,
        warmup_episodes: int = 40,
        reward_path: str = REWARD_PROGRAM_PATH,
        archive_path: str = "reward_archive.json",
        initial_episode_count: int = 0,
        initial_last_evolution_index: int = -1,
        verbose: bool = True,
    ):
        self.goal = goal
        self.evolve_every = evolve_every
        self.warmup_episodes = warmup_episodes
        self.reward_path = reward_path
        self.verbose = verbose

        self.archive = RewardArchive(archive_path)

        self._episode_stats: list[dict] = []
        self._episode_count = max(0, int(initial_episode_count))
        self._last_evolution_index = int(initial_last_evolution_index)
        # NOTE: generation is NEVER tracked as an independent counter.
        # It is always derived from len(self.archive.entries) — this is the
        # single source of truth, fixing a bug where an independent counter
        # could drift out of sync with the archive.

        _WIN = 10
        self._policy_buf: Deque[dict] = deque(maxlen=_WIN)

        self._current_code: str = ""
        self._active_generation = 0
        self._current_code = self._load_current_code()
        self._reconcile_disk_with_archive()
        self._sync_active_generation()

        if self.verbose:
            print(
                f"[designer] Text-to-Reward | goal='{goal[:60]}' | "
                f"evolve_every={evolve_every} | warmup={warmup_episodes} | "
                f"episodes={self._episode_count} | "
                f"archive={len(self.archive.entries)} entries | "
                f"active_generation={self._active_generation}"
            )

    def _sync_active_generation(self) -> None:
        """
        Align the logged/active generation label with disk vs archive.

        After a failed LLM update the archive grows but disk still runs the
        program that was just archived — active_generation must stay on that
        index instead of jumping to len(archive).
        """
        n = len(self.archive.entries)
        if n == 0:
            self._active_generation = 0
            return
        latest = self.archive.entries[-1]
        disk_body = extract_reward_body(self._current_code).strip()
        arch_body = extract_reward_body(latest["reward_code"]).strip()
        if disk_body and arch_body and disk_body == arch_body:
            self._active_generation = latest["generation"]
        else:
            self._active_generation = n

    def _reconcile_disk_with_archive(self) -> None:
        """
        Sync disk reward with archive only when disk has no usable program.

        A freshly bootstrapped or hand-edited reward_program.py must NOT be
        overwritten by an older archive entry on startup.
        """
        latest_entry = self.archive.get_latest()
        if latest_entry is None:
            return

        disk_code = self._current_code
        archive_code = latest_entry["reward_code"]
        if disk_code == archive_code:
            return

        disk_usable = bool(disk_code) and not _is_placeholder_code(disk_code)
        archive_usable = not _is_placeholder_code(archive_code)

        if disk_usable:
            if self.verbose:
                print(
                    "[designer] Keeping reward on disk — it differs from archive "
                    f"generation {latest_entry['generation']} and takes precedence."
                )
            return

        if archive_usable:
            if self.verbose:
                print(
                    "[designer] reward_program.py missing or placeholder — "
                    f"restoring from archive generation {latest_entry['generation']}."
                )
            self._restore_from_archive_entry(latest_entry)

    def _restore_from_archive_entry(self, entry: dict) -> None:
        """
        Restores `reward_program.py` from an archive entry -- but ONLY after
        re-running the exact same validate + smoke-test pipeline used for
        freshly-generated code (fixes audit finding #5).

        Why this matters: previously, a disk/archive entry was written
        straight to `reward_program.py` on restore with no re-validation at
        all. Combined with the other two issues this audit flagged --
        `_load_reward_fn` not stripping real builtins (#6) and the per-step
        execution path bypassing the timeout-protected sandbox (#2) -- a
        single corrupted or maliciously-modified archive entry on disk would
        have been: (a) trusted unconditionally, (b) executed with real
        Python builtins available, (c) with no timeout. That three-step
        chain is a complete RCE path. Re-validating here closes the first
        link: even a corrupted archive entry can no longer reach disk/exec
        without passing the same AST + smoke-test gate as a brand-new LLM
        generation.

        On validation/smoke-test failure, this does NOT write the untrusted
        code to disk. It loudly warns and writes a minimal, known-safe
        placeholder instead, so training can still proceed and the failure
        is visible rather than silent.
        """
        restored_code = entry["reward_code"]

        ok, err = validate_reward_code(restored_code)
        smoke_ok, smoke_err, smoke_console = (False, "(skipped: structural validation failed)", "")
        if ok:
            smoke_ok, smoke_err, _smoke_console = _full_validation_pipeline(restored_code)

        if ok and smoke_ok:
            self._save_reward_program(restored_code)
            return

        reason = err if not ok else smoke_err
        gen = entry.get("generation", "?")
        print(
            f"[designer] WARNING: archive entry for generation {gen} FAILED "
            f"re-validation on restore ({reason}) — refusing to write it to disk. "
            "Removing corrupt entry from archive and falling back to a safe "
            "placeholder reward program instead."
        )
        if isinstance(gen, int):
            self.archive.remove_generation(gen)
            self._sync_active_generation()
        placeholder = (
            "def compute_reward(state):\n"
            '    if state["collided"]:\n'
            "        return -30.0\n"
            "    return 0.0  # placeholder: archived code failed re-validation on restore\n"
        )
        self._save_reward_program(placeholder)

    @property
    def generation(self) -> int:
        """Number of archived generations (len of archive)."""
        return len(self.archive.entries)

    # ── Backward-compat shim so train.py get_weights() still works ------------

    def get_weights(self) -> dict:
        """Compatibility stub. Returns active generation info for logging."""
        return {"generation": self._active_generation, "reward_path": self.reward_path}

    # ── Code management -------------------------------------------------------

    def _load_current_code(self) -> str:
        if os.path.exists(self.reward_path):
            try:
                with open(self.reward_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
        return ""

    def _save_reward_program(self, code: str, generation_label: int | None = None) -> None:
        """
        Writes `code` to disk as the reward program currently in effect.
        """
        gen_label = self._active_generation if generation_label is None else generation_label
        tmp = self.reward_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(
                f'"""\nreward_program.py -- Generation {gen_label}\n'
                f'Auto-generated by RewardDesigner. DO NOT EDIT MANUALLY.\n"""\n\n'
            )
            f.write(code)
        os.replace(tmp, self.reward_path)
        self._current_code = code
        self._active_generation = gen_label
        print(f"[designer] reward_program.py updated (generation {gen_label})")

    # ── PPO policy metrics ----------------------------------------------------

    def push_policy_metrics(
        self,
        entropy: float,
        value_loss: float,
        policy_loss: float,
        explained_variance: float,
    ) -> None:
        self._policy_buf.append(
            {
                "entropy": entropy,
                "value_loss": value_loss,
                "policy_loss": policy_loss,
                "explained_variance": explained_variance,
            }
        )

    def get_policy_snapshot(self) -> dict | None:
        if not self._policy_buf:
            return None
        n = len(self._policy_buf)
        return {
            "n_updates": n,
            "entropy": sum(d["entropy"] for d in self._policy_buf) / n,
            "value_loss": sum(d["value_loss"] for d in self._policy_buf) / n,
            "policy_loss": sum(d["policy_loss"] for d in self._policy_buf) / n,
            "explained_variance": sum(d["explained_variance"] for d in self._policy_buf) / n,
        }

    # ── Episode recording -----------------------------------------------------

    def accumulate_episode(self, stats: dict) -> None:
        """Record one completed episode's stats (no evolution trigger)."""
        self._episode_stats.append(stats)
        self._episode_count += 1

    def maybe_evolve(self) -> bool:
        """
        Run one evolution step when a pending evolve boundary has been crossed.

        Boundaries are indexed from 1 after warmup. When parallel envs finish
        multiple episodes in one SB3 step, callers should accumulate each
        episode then call this after every accumulate so exact boundaries are
        not skipped. The highest-index check also catches a single end-of-batch
        call that jumps past a boundary.
        """
        if self._episode_count < self.warmup_episodes:
            return False

        past_warmup = self._episode_count - self.warmup_episodes
        if past_warmup <= 0:
            return False

        highest_index = past_warmup // self.evolve_every
        completed_count = max(self._last_evolution_index, 0)
        if highest_index <= completed_count:
            return False

        self._last_evolution_index = completed_count + 1
        return self._evolve()

    def record_episode(self, stats: dict) -> bool:
        """Accumulate stats and maybe evolve (single-env convenience API)."""
        self.accumulate_episode(stats)
        return self.maybe_evolve()

    def _evolve(self) -> bool:
        """
        One evolutionary step. Order of operations matters:

          1. Aggregate metrics for the reward program that JUST RAN.
          2. Archive it UNCONDITIONALLY (add_entry).
          3. Critique the entry just archived.
          4. Store the critique back onto that same entry.
          5. Generate + validate + smoke-test an improved reward program.
          6. Only on successful validation AND smoke-test, save to disk.
             On any failure, the previous program stays in effect.
        """
        if not self._episode_stats:
            return False

        window_stats = self._episode_stats[: self.evolve_every]
        overflow_stats = self._episode_stats[self.evolve_every :]

        metrics = self._aggregate_metrics(window_stats)
        current_gen = self._active_generation

        if self.verbose:
            print(
                f"\n[designer] Generation {current_gen} | "
                f"episodes={len(window_stats)} | "
                f"speed={metrics.get('mean_speed', 0):.2f} m/s | "
                f"crash={metrics.get('crash_rate', 0):.1%} | "
                f"overtakes={metrics.get('mean_overtakes', 0):.2f}/ep"
            )

        current_code = self._current_code or self._load_current_code()
        if not current_code or _is_placeholder_code(current_code):
            print(
                "[designer] WARNING: no usable reward code on disk — "
                "skipping archive (will not pollute RAG with placeholder)."
            )
            archive_context = self.archive.format_for_llm(k=3)
            new_code = self._call_generate_with_repair(archive_context)
            if new_code is None:
                print("[designer] LLM generation failed -- keeping current reward.")
            else:
                self._save_reward_program(new_code, generation_label=current_gen)
            self._episode_stats = overflow_stats
            return new_code is not None

        # ── 1+2. Archive the program that just ran ───────────────────────────
        previous_entry = self.archive.get_latest()
        trend_summary = self._format_trend(metrics, previous_entry)

        entry = self.archive.add_entry(
            reward_code=current_code,
            metrics=metrics,
            critique="",
        )

        # ── 3+4. Critique the entry we just archived ─────────────────────────
        traj_summary = self._format_trajectory_samples(window_stats[-5:])
        critique = self._call_critique(
            reward_code=entry["reward_code"],
            metrics=metrics,
            trajectory_summary=traj_summary,
            generation=entry["generation"],
            fitness=entry["fitness"],
            trend_summary=trend_summary,
        )
        if critique:
            self.archive.update_critique(entry["generation"], critique)
            if self.verbose:
                print(f"[designer] Critique stored for generation {entry['generation']}")

        # ── 5+6. Generate, validate, smoke-test, and save ────────────────────
        # Improvement #5: pass current failure modes for targeted retrieval
        current_failure_modes = entry.get("critique_meta", {}).get("failure_modes", [])
        archive_context = self.archive.format_for_llm(
            k=3, current_failure_modes=current_failure_modes
        )
        new_code = self._call_generate_with_repair(archive_context)

        if new_code is None:
            self._active_generation = entry["generation"]
            self._episode_stats = overflow_stats
            return False

        self._save_reward_program(new_code, generation_label=len(self.archive.entries))
        self._episode_stats = overflow_stats
        return True

    def _call_generate_with_repair(
        self,
        archive_context: str,
        max_retries: int = 3,
    ) -> str | None:
        """
        Generate a reward function, then validate (AST) + smoke-test (execution,
        two stages -- see _full_validation_pipeline). If either AST validation
        or either smoke-test stage fails, send the error back to the LLM for
        repair. Returns the first code that passes both stages, or None on
        total failure.
        """
        system = _GENERATION_SYSTEM.format(state_schema=_STATE_SCHEMA)
        user = _GENERATION_USER_TEMPLATE.format(
            goal=self.goal,
            archive_context=archive_context,
        )

        raw: str | None = None
        repair_error: str = ""

        if self.verbose:
            print(f"[designer] Generating reward ({max_retries} attempts max)...")

        for attempt in range(1, max_retries + 1):
            # On attempt 1, use the standard generation prompt.
            # On later attempts, use the repair prompt with the previous error.
            if attempt == 1 or raw is None:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            else:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": _REPAIR_USER_TEMPLATE.format(
                            error=repair_error,
                            rejected_code=raw,
                        ),
                    },
                ]

            try:
                resp = call_with_rotation(
                    model=MODEL,
                    messages=messages,
                    temperature=0.5,
                    max_tokens=800,
                )
                raw = resp.choices[0].message.content.strip()
                raw = re.sub(r"```python\n?|```\n?", "", raw).strip()
                if "def compute_reward" in raw:
                    idx = raw.index("def compute_reward")
                    raw = raw[idx:]
            except Exception as exc:
                if self.verbose:
                    print(
                        f"[designer] attempt {attempt}/{max_retries}: "
                        f"API error — {type(exc).__name__}: {exc}"
                    )
                if attempt < max_retries:
                    time.sleep(2**attempt)
                raw = None
                repair_error = f"API error: {exc}"
                continue

            # ── Structural validation (AST) ───────────────────────────────
            ok, err = validate_reward_code(raw)
            if not ok:
                if self.verbose:
                    print(
                        f"[designer] attempt {attempt}/{max_retries}: "
                        f"AST fail — {err}"
                    )
                repair_error = f"Structural validation error: {err}"
                continue

            # ── Smoke-test: Stage A (fast) then Stage B (full bank) ────────
            smoke_ok, smoke_err, smoke_console = _full_validation_pipeline(raw)
            if not smoke_ok:
                if self.verbose:
                    print(
                        f"[designer] attempt {attempt}/{max_retries}: "
                        f"smoke fail — {smoke_console}"
                    )
                repair_error = smoke_err
                continue

            if self.verbose:
                print(
                    f"[designer] attempt {attempt}/{max_retries}: "
                    f"accepted ({len(raw)} chars)"
                )
            # Both checks passed — return the valid code.
            return raw

        if self.verbose:
            print(
                f"[designer] evolution skipped — all {max_retries} attempts "
                f"failed smoke-test; keeping current reward"
            )
        return None

    # ── LLM: legacy generate (kept for internal use; routes to repair loop) ──

    def _call_generate(self, archive_context: str, max_retries: int = 3) -> str | None:
        """Thin wrapper around _call_generate_with_repair for backward compat."""
        return self._call_generate_with_repair(archive_context, max_retries)

    # ── LLM: critique ---------------------------------------------------------

    def _call_critique(
        self,
        reward_code: str,
        metrics: dict,
        trajectory_summary: str,
        generation: int,
        fitness: float,
        trend_summary: str = "(no previous generation to compare against)",
        max_retries: int = 2,
    ) -> str:
        user = _CRITIQUE_USER_TEMPLATE.format(
            generation=generation,
            reward_code=reward_code,
            mean_speed=metrics.get("mean_speed", 0.0),
            crash_rate=metrics.get("crash_rate", 0.0),
            mean_overtakes=metrics.get("mean_overtakes", 0.0),
            completion_rate=metrics.get("completion_rate", 0.0),
            mean_steps=metrics.get("mean_steps", 0.0),
            mean_ttc=metrics.get("mean_ttc", 0.0),
            p10_ttc=metrics.get("p10_ttc", -1.0),
            min_ttc=metrics.get("min_ttc", -1.0),
            mean_long_jerk=metrics.get("mean_long_jerk", 0.0),
            mean_accel=metrics.get("mean_accel", 0.0),
            total_lane_changes=metrics.get("total_lane_changes", 0),
            fitness=fitness,
            trend_summary=trend_summary,
            trajectory_summary=trajectory_summary,
        )

        for attempt in range(1, max_retries + 1):
            try:
                resp = call_with_rotation(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": _CRITIQUE_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.3,
                    max_tokens=500,
                )
                return resp.choices[0].message.content.strip()

            except Exception as exc:
                print(f"[designer] Critique attempt {attempt}/{max_retries} failed: " f"{type(exc).__name__}: {exc}")
                if attempt < max_retries:
                    time.sleep(2**attempt)

        return "(critique unavailable)"

    # ── Manual generation (CLI / bootstrap) -----------------------------------

    def generate_reward(self, goal: str | None = None) -> bool:
        """
        Bootstraps an initial reward program before any training/evaluation
        has happened. Used by train.py's --bootstrap step.

        Deliberately does NOT call archive.add_entry() here: there are no
        real metrics yet (no episodes have run), so archiving now would
        create a generation 0 entry with fake/empty metrics.

        Uses the same validate + smoke-test pipeline as _evolve() so a bad
        bootstrap reward program never gets written to disk silently.
        """
        if goal:
            self.goal = goal

        archive_context = self.archive.format_for_llm(k=3)
        new_code = self._call_generate_with_repair(archive_context)

        if new_code is None:
            print("[designer] Bootstrap generation failed — no reward program written.")
            return False

        # Both structural validation and smoke-test passed inside
        # _call_generate_with_repair, so it's safe to write to disk.
        self._save_reward_program(new_code)
        return True

    # ── Helpers ---------------------------------------------------------------

    @staticmethod
    def _percentile(values: list[float], pct: int) -> float:
        if not values:
            return 30.0
        sorted_vals = sorted(values)
        k = (len(sorted_vals) - 1) * pct / 100.0
        lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
        frac = k - lo
        return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac

    @staticmethod
    def _aggregate_metrics(episode_stats: list[dict]) -> dict:
        n = max(len(episode_stats), 1)
        crashes = sum(1 for s in episode_stats if s.get("collisions", 0) > 0)
        total_overtakes = sum(s.get("total_overtakes", 0) for s in episode_stats)

        all_ttc: list[float] = []
        for s in episode_stats:
            if s.get("ttc_vals"):
                all_ttc.extend(float(v) for v in s["ttc_vals"])
            else:
                for sample in s.get("trajectory_samples", []):
                    all_ttc.append(float(sample.get("ttc", 30.0)))

        if all_ttc:
            p10_ttc = RewardDesigner._percentile(all_ttc, 10)
            min_ttc = min(all_ttc)
        else:
            p10_ttc = sum(s.get("p10_ttc", 30.0) for s in episode_stats) / n
            min_ttc = min((s.get("min_ttc", 30.0) for s in episode_stats), default=30.0)

        return {
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
            "max_steps": 300,
        }

    @staticmethod
    def _format_trend(current_metrics: dict, previous_entry: dict | None) -> str:
        """
        Compares current metrics to the previous archived generation, making
        speed/overtake/crash trends explicit so the LLM can catch the
        "slow down to avoid crashing" reward-hacking pattern even when each
        individual generation's metrics look reasonable in isolation.
        """
        if previous_entry is None:
            return "(no previous generation to compare against — this is generation 0)"

        prev = previous_entry["metrics"]

        def _delta(key: str, fmt: str = "{:+.2f}") -> str:
            cur_v = current_metrics.get(key, 0.0)
            prev_v = prev.get(key, 0.0)
            return fmt.format(cur_v - prev_v)

        speed_delta = _delta("mean_speed")
        overtake_delta = _delta("mean_overtakes")
        crash_delta = _delta("crash_rate", "{:+.1%}")

        warning = ""
        speed_dropped = current_metrics.get("mean_speed", 0.0) < prev.get("mean_speed", 0.0)
        overtakes_dropped = current_metrics.get("mean_overtakes", 0.0) < prev.get("mean_overtakes", 0.0)
        crash_improved = current_metrics.get("crash_rate", 1.0) < prev.get("crash_rate", 1.0)
        if (speed_dropped or overtakes_dropped) and crash_improved:
            warning = (
                "\n  !! WARNING: crash_rate improved but speed and/or overtakes "
                "DECREASED vs the previous generation. This is the classic "
                "'slow down to stay safe' reward-hacking pattern — the agent "
                "may be avoiding risk by driving passively rather than driving well.\n"
            )

        return (
            f"  previous generation : {previous_entry['generation']}\n"
            f"  mean_speed     delta: {speed_delta} m/s\n"
            f"  mean_overtakes delta: {overtake_delta} per episode\n"
            f"  crash_rate     delta: {crash_delta}"
            f"{warning}"
        )

    @staticmethod
    def _format_trajectory_samples(episode_stats: list[dict]) -> str:
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
