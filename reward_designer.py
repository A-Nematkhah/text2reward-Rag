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

Smoke-test validation (NEW)
────────────────────────────
After structural AST validation, every generated reward function is
executed against two representative sample states before being written
to disk. This catches runtime errors (e.g. KeyError: 'overtake' instead
of the correct key 'overtook') that structural checks cannot detect.
Failures are fed back through the same repair-loop as structural errors.
"""

from __future__ import annotations

import os
import re
import time
import math
import threading
from collections import deque
from typing import Deque

from groq import Groq

from reward_sandbox import validate_reward_code
from reward_archive import RewardArchive
from key_manager import call_with_rotation  # ← چرخش خودکار کلید

# ── Groq client ───────────────────────────────────────────────────────────────
# _client دیگه استفاده نمیشه — key_manager مدیریت می‌کنه
_client: Groq | None = None
MODEL = "llama-3.3-70b-versatile"

REWARD_PROGRAM_PATH = "reward_program.py"

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


def _call_with_timeout(fn, state: dict, timeout_sec: float = _SMOKE_TEST_TIMEOUT_SEC):
    """
    Executes fn(state) under a hard wall-clock timeout, mirroring the same
    thread-based timeout used by reward_sandbox.execute_reward(). Re-raises
    whatever exception fn(state) raised (including KeyError, so the existing
    "wrong state key" repair-loop messaging downstream still works
    unchanged), or raises TimeoutError if fn does not return in time.
    """
    result: list = []
    exc: list[BaseException] = []

    def _run():
        try:
            result.append(fn(state))
        except Exception as e:  # noqa: BLE001 - intentionally broad, re-raised below
            exc.append(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)

    if t.is_alive():
        raise TimeoutError(
            f"compute_reward timed out after {timeout_sec}s during smoke test "
            "(possible runaway computation, e.g. a deeply nested/expensive expression)"
        )
    if exc:
        raise exc[0]
    if not result:
        raise RuntimeError("compute_reward returned no value during smoke test")
    return result[0]


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
    * Speed reward should be continuous and always incentivise going faster
    * TTC penalty should activate only below 3 s -- not for normal driving
    * Overtake bonus: large one-shot reward (+1 to +3) when overtook == True
    * Jerk/accel penalties should be small (0.01-0.05 scale) to not suppress action
    * Avoid rewarding stationary behaviour or unnecessary lane changes

{state_schema}

Reply ONLY with the Python source of compute_reward(state). No explanation, no markdown fences.
"""

_CRITIQUE_SYSTEM = """\
You are a reinforcement learning reward auditor. Analyse the reward function and metrics below.

Identify:
1. Reward hacking patterns:
   - Oscillatory lane changes: lane_changes >> overtakes (agent thrashing lanes for reward)
   - Acceleration spam: high mean_accel with low speed gain (braking-acceleration exploit)
   - Stationary farming: very low mean_speed but high shaped_reward
   - TTC exploitation: very low ttc but no crashes (agent riding tailgate for some bonus)
2. Missing incentives (what good behaviour is not rewarded)
3. Misaligned incentives (what bad behaviour is inadvertently rewarded)
4. Proposed improvements with SPECIFIC code changes

Be concise (max 300 words). End with 3 concrete bullet-point improvements.
"""

_GENERATION_USER_TEMPLATE = """\
=== DRIVING GOAL ===
{goal}

=== ARCHIVE MEMORY (top performing reward programs) ===
{archive_context}

=== TASK ===
Generate an improved compute_reward(state) function that achieves the goal above.
Learn from the archive: adopt what worked, fix what was critiqued.
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
    states to catch runtime errors that structural AST validation cannot catch,
    such as KeyError from a wrong state key (e.g. state['overtake'] instead of
    the correct state['overtook']).

    Returns (ok, error_message).  ok=True means all samples ran cleanly.
    """

    def _clip(val, lo, hi):
        return max(lo, min(hi, val))

    safe_globals = {
        "__builtins__": {},
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "float": float,
        "int": int,
        "bool": bool,
        "sqrt": math.sqrt,
        "exp": math.exp,
        "log": math.log,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "atan": math.atan,
        "atan2": math.atan2,
        "floor": math.floor,
        "ceil": math.ceil,
        "clip": _clip,
        "pi": math.pi,
        "e": math.e,
        "inf": math.inf,
    }

    local_ns: dict = {}
    try:
        exec(compile(code, "<generated>", "exec"), safe_globals, local_ns)  # noqa: S102
    except Exception as exc:
        return False, f"Compile error: {type(exc).__name__}: {exc}"

    fn = local_ns.get("compute_reward")
    if fn is None:
        return False, "compute_reward function not found after exec"

    rewards: dict[str, float] = {}
    for name, sample in [
        ("normal", _SAMPLE_STATE_NORMAL),
        ("overtake", _SAMPLE_STATE_OVERTAKE),
        ("collision", _SAMPLE_STATE_COLLIDED),
    ]:
        try:
            result = _call_with_timeout(fn, sample)
            if not isinstance(result, (int, float)):
                return False, (
                    f"Runtime error on sample state '{name}': "
                    f"compute_reward returned {type(result).__name__} instead of float"
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
        except Exception as exc:
            return False, (f"Runtime error on sample state '{name}': " f"{type(exc).__name__}: {exc}")

    # ── Gate 1b: Quantitative Collision Penalty Check (X = 20) ────────────
    X = 20.0
    if "collision" in rewards and "normal" in rewards:
        # The collided state's reward must be at least X points lower than a
        # typical normal state's reward. If not, the collision penalty is being
        # suppressed by other bonuses in the same step.
        if rewards["collision"] >= (rewards["normal"] - X):
            msg = (
                "Safety Gate Violation: Collision penalty is suppressed by positive bonuses! "
                f"Normal state reward: {rewards['normal']:.2f}, "
                f"Collided state reward: {rewards['collision']:.2f}. "
                f"The collision reward must be at least X={X} points lower than the normal reward to bypass suppression."
            )
            return False, msg

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
            cautious_return += _call_with_timeout(fn, cautious_state)
    except TimeoutError as exc:
        return False, (
            f"Timeout during cautious-trajectory smoke test: {exc}. "
            "Simplify the reward function -- it is too computationally expensive."
        )
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
            reckless_return += _call_with_timeout(fn, reckless_state)
    except TimeoutError as exc:
        return False, (
            f"Timeout during reckless-trajectory smoke test: {exc}. "
            "Simplify the reward function -- it is too computationally expensive."
        )
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

    return True, ""


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
        verbose: bool = True,
    ):
        self.goal = goal
        self.evolve_every = evolve_every
        self.warmup_episodes = warmup_episodes
        self.reward_path = reward_path
        self.verbose = verbose

        self.archive = RewardArchive(archive_path)

        self._episode_stats: list[dict] = []
        self._episode_count = 0
        # NOTE: generation is NEVER tracked as an independent counter.
        # It is always derived from len(self.archive.entries) — this is the
        # single source of truth, fixing a bug where an independent counter
        # could drift out of sync with the archive.

        _WIN = 10
        self._policy_buf: Deque[dict] = deque(maxlen=_WIN)

        self._current_code: str = self._load_current_code()

        # ── Reconcile disk vs archive ───────────────────────────────────────
        latest_entry = self.archive.get_latest()
        if latest_entry is not None and self._current_code != latest_entry["reward_code"]:
            if self.verbose:
                print(
                    "[designer] reward_program.py is missing or out of sync "
                    f"with archive generation {latest_entry['generation']} — "
                    "restoring it from the archive."
                )
            self._restore_from_archive_entry(latest_entry)

        if self.verbose:
            print(
                f"[designer] Text-to-Reward | goal='{goal[:60]}' | "
                f"evolve_every={evolve_every} | warmup={warmup_episodes} | "
                f"archive={len(self.archive.entries)} entries | "
                f"generation={self.generation}"
            )

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
        smoke_ok, smoke_err = (False, "(skipped: structural validation failed)")
        if ok:
            smoke_ok, smoke_err = _smoke_test_reward_code(restored_code)

        if ok and smoke_ok:
            self._save_reward_program(restored_code)
            return

        reason = err if not ok else smoke_err
        print(
            f"[designer] WARNING: archive entry for generation "
            f"{entry.get('generation', '?')} FAILED re-validation on restore "
            f"({reason}) — refusing to write it to disk. Falling back to a "
            "safe placeholder reward program instead."
        )
        placeholder = (
            "def compute_reward(state):\n"
            '    if state["collided"]:\n'
            "        return -30.0\n"
            "    return 0.0  # placeholder: archived code failed re-validation on restore\n"
        )
        self._save_reward_program(placeholder)

    @property
    def generation(self) -> int:
        """Current generation number — always derived from the archive length."""
        return len(self.archive.entries)

    # ── Backward-compat shim so train.py get_weights() still works ------------

    def get_weights(self) -> dict:
        """Compatibility stub. Returns generation info instead of weights."""
        return {"generation": self.generation, "reward_path": self.reward_path}

    # ── Code management -------------------------------------------------------

    def _load_current_code(self) -> str:
        if os.path.exists(self.reward_path):
            try:
                with open(self.reward_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
        return ""

    def _save_reward_program(self, code: str) -> None:
        """
        Writes `code` to disk as the reward program currently in effect.

        The header always reflects the NEXT generation number that will be
        produced once this program is evaluated and archived (i.e. the
        program about to run is generation `self.generation`, since the
        archive has not yet grown to include it).
        """
        gen_label = self.generation
        tmp = self.reward_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(
                f'"""\nreward_program.py -- Generation {gen_label}\n'
                f'Auto-generated by RewardDesigner. DO NOT EDIT MANUALLY.\n"""\n\n'
            )
            f.write(code)
        os.replace(tmp, self.reward_path)
        self._current_code = code
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

    def record_episode(self, stats: dict) -> bool:
        self._episode_stats.append(stats)
        self._episode_count += 1

        if self._episode_count < self.warmup_episodes:
            return False

        past_warmup = self._episode_count - self.warmup_episodes
        if past_warmup % self.evolve_every == 0:
            return self._evolve()

        return False

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

        metrics = self._aggregate_metrics(self._episode_stats)
        current_gen = self.generation

        if self.verbose:
            print(
                f"\n[designer] Generation {current_gen} | "
                f"episodes={len(self._episode_stats)} | "
                f"speed={metrics.get('mean_speed', 0):.2f} m/s | "
                f"crash={metrics.get('crash_rate', 0):.1%} | "
                f"overtakes={metrics.get('mean_overtakes', 0):.2f}/ep"
            )

        # ── 1+2. Archive the program that just ran ───────────────────────────
        current_code = self._current_code or self._load_current_code()
        if not current_code:
            print(
                "[designer] WARNING: no current reward code found to archive "
                "(reward_program.py missing or unreadable) — using fallback "
                "placeholder so the archive stays in sync."
            )
            current_code = (
                "def compute_reward(state):\n" "    return 0.0  # placeholder: original code was unavailable\n"
            )

        previous_entry = self.archive.get_latest()
        trend_summary = self._format_trend(metrics, previous_entry)

        entry = self.archive.add_entry(
            reward_code=current_code,
            metrics=metrics,
            critique="",
        )

        # ── 3+4. Critique the entry we just archived ─────────────────────────
        traj_summary = self._format_trajectory_samples(self._episode_stats[-5:])
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
        archive_context = self.archive.format_for_llm(k=3)
        new_code = self._call_generate_with_repair(archive_context)

        if new_code is None:
            print("[designer] LLM generation failed -- keeping current reward.")
            self._episode_stats.clear()
            return False

        self._save_reward_program(new_code)
        self._episode_stats.clear()
        return True

    def _call_generate_with_repair(
        self,
        archive_context: str,
        max_retries: int = 3,
    ) -> str | None:
        """
        Generate a reward function, then validate (AST) + smoke-test (execution).
        If either check fails, send the error back to the LLM for repair.
        Returns the first code that passes both checks, or None on total failure.
        """
        system = _GENERATION_SYSTEM.format(state_schema=_STATE_SCHEMA)
        user = _GENERATION_USER_TEMPLATE.format(
            goal=self.goal,
            archive_context=archive_context,
        )

        raw: str | None = None
        repair_error: str = ""

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
                print(
                    f"[designer] Generate attempt {attempt}/{max_retries} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                if attempt < max_retries:
                    time.sleep(2**attempt)
                raw = None
                repair_error = f"API error: {exc}"
                continue

            if self.verbose:
                print(f"[designer] Generated reward ({len(raw)} chars)")

            # ── Structural validation (AST) ───────────────────────────────
            ok, err = validate_reward_code(raw)
            if not ok:
                print(f"[designer] Validation failed (attempt {attempt}): {err}")
                repair_error = f"Structural validation error: {err}"
                continue

            # ── Smoke-test: actually execute against sample states ────────
            smoke_ok, smoke_err = _smoke_test_reward_code(raw)
            if not smoke_ok:
                print(f"[designer] Smoke-test failed (attempt {attempt}): {smoke_err}")
                repair_error = smoke_err
                continue

            # Both checks passed — return the valid code.
            return raw

        print(f"[designer] All {max_retries} attempts failed — keeping current reward.")
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
    def _aggregate_metrics(episode_stats: list[dict]) -> dict:
        n = max(len(episode_stats), 1)
        crashes = sum(1 for s in episode_stats if s.get("collisions", 0) > 0)
        total_overtakes = sum(s.get("total_overtakes", 0) for s in episode_stats)
        return {
            "n_episodes": n,
            "mean_speed": sum(s.get("mean_speed", 0) for s in episode_stats) / n,
            "crash_rate": crashes / n,
            "completion_rate": 1.0 - crashes / n,
            "mean_overtakes": total_overtakes / n,
            "mean_steps": sum(s.get("steps", 0) for s in episode_stats) / n,
            "mean_ttc": sum(s.get("mean_ttc", 0) for s in episode_stats) / n,
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
