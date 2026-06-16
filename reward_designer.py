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
  [Sandbox validation: AST check + type check]
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
"""

from __future__ import annotations

import os
import re
import json
import time
from collections import deque
from typing import Deque

from groq import Groq

from reward_sandbox import validate_reward_code
from reward_archive import RewardArchive

# ── Groq client ───────────────────────────────────────────────────────────────
_client: Groq | None = None
MODEL = "llama-3.3-70b-versatile"

REWARD_PROGRAM_PATH = "reward_program.py"

_STATE_SCHEMA = """\
State keys available inside compute_reward(state):
  speed_ms        : float   ego speed in m/s (range 0-40)
  front_dist      : float   distance to front vehicle [m] (0-200, 200 = clear)
  ttc             : float   time-to-collision [s] (0-30, 30 = no threat)
  rel_vel_ms      : float   v_front - v_ego [m/s] (negative = approaching)
  lane            : int     lane index, 0 = rightmost
  overtook        : bool    completed an overtake this step
  lane_changed    : bool    lane changed since last step
  collided        : bool    collision detected
  nearby_vehicles : int     vehicles within ~30 m radius
  accel_ms2       : float   longitudinal acceleration [m/s2]
  long_jerk       : float   longitudinal jerk [m/s3]
  lat_jerk        : float   lateral jerk [m/s3]

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

=== EPISODE TRAJECTORY SAMPLES ===
{trajectory_summary}

Identify reward hacking, failure modes, and propose 3 specific improvements.
"""


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "\n[ERROR] GROQ_API_KEY is not set.\n"
                "Set it before starting training:\n"
                "  Linux   : export GROQ_API_KEY=gsk_xxxxxxxx\n"
                "  Windows : set GROQ_API_KEY=gsk_xxxxxxxx\n"
                "  Colab   : import os; os.environ['GROQ_API_KEY'] = 'gsk_xxxxxxxx'"
            )
        _client = Groq(api_key=api_key)
    return _client


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
        self.goal            = goal
        self.evolve_every    = evolve_every
        self.warmup_episodes = warmup_episodes
        self.reward_path     = reward_path
        self.verbose         = verbose

        self.archive = RewardArchive(archive_path)

        self._episode_stats: list[dict] = []
        self._episode_count  = 0
        self._generation     = len(self.archive.entries)

        _WIN = 10
        self._policy_buf: Deque[dict] = deque(maxlen=_WIN)

        self._current_code: str = self._load_current_code()

        if self.verbose:
            print(
                f"[designer] Text-to-Reward | goal='{goal[:60]}' | "
                f"evolve_every={evolve_every} | warmup={warmup_episodes} | "
                f"archive={len(self.archive.entries)} entries"
            )

    # ── Backward-compat shim so train.py get_weights() still works ------------

    def get_weights(self) -> dict:
        """Compatibility stub. Returns generation info instead of weights."""
        return {"generation": self._generation, "reward_path": self.reward_path}

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
        tmp = self.reward_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(
                f'"""\nreward_program.py -- Generation {self._generation}\n'
                f'Auto-generated by RewardDesigner. DO NOT EDIT MANUALLY.\n"""\n\n'
            )
            f.write(code)
        os.replace(tmp, self.reward_path)
        self._current_code = code
        print(f"[designer] reward_program.py updated (generation {self._generation})")

    # ── PPO policy metrics ----------------------------------------------------

    def push_policy_metrics(
        self,
        entropy: float,
        value_loss: float,
        policy_loss: float,
        explained_variance: float,
    ) -> None:
        self._policy_buf.append({
            "entropy":            entropy,
            "value_loss":         value_loss,
            "policy_loss":        policy_loss,
            "explained_variance": explained_variance,
        })

    def get_policy_snapshot(self) -> dict | None:
        if not self._policy_buf:
            return None
        n = len(self._policy_buf)
        return {
            "n_updates":          n,
            "entropy":            sum(d["entropy"]            for d in self._policy_buf) / n,
            "value_loss":         sum(d["value_loss"]         for d in self._policy_buf) / n,
            "policy_loss":        sum(d["policy_loss"]        for d in self._policy_buf) / n,
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
        if not self._episode_stats:
            return False

        metrics = self._aggregate_metrics(self._episode_stats)

        if self.verbose:
            print(
                f"\n[designer] Generation {self._generation} | "
                f"episodes={len(self._episode_stats)} | "
                f"speed={metrics.get('mean_speed', 0):.2f} m/s | "
                f"crash={metrics.get('crash_rate', 0):.1%} | "
                f"overtakes={metrics.get('mean_overtakes', 0):.2f}/ep"
            )

        # Critique current reward
        critique = ""
        latest = self.archive.get_latest()
        if latest is not None:
            traj_summary = self._format_trajectory_samples(self._episode_stats[-5:])
            critique = self._call_critique(
                reward_code        = latest["reward_code"],
                metrics            = metrics,
                trajectory_summary = traj_summary,
                generation         = latest["generation"],
                fitness            = latest["fitness"],
            )
            if critique:
                self.archive.update_critique(latest["generation"], critique)
                if self.verbose:
                    print(f"[designer] Critique stored for generation {latest['generation']}")

        # Store current reward if not already archived
        if self._generation == len(self.archive.entries):
            current_code = self._load_current_code()
            if current_code:
                self.archive.add_entry(
                    reward_code = current_code,
                    metrics     = metrics,
                    critique    = critique,
                )

        # Generate new reward function
        archive_context = self.archive.format_for_llm(k=3)
        new_code = self._call_generate(archive_context)

        if new_code is None:
            print("[designer] LLM generation failed -- keeping current reward.")
            self._episode_stats.clear()
            return False

        ok, err = validate_reward_code(new_code)
        if not ok:
            print(f"[designer] Validation failed: {err}")
            print("[designer] Keeping current reward program.")
            self._episode_stats.clear()
            return False

        self._generation += 1
        self._save_reward_program(new_code)
        self._episode_stats.clear()
        return True

    # ── LLM: generate ---------------------------------------------------------

    def _call_generate(
        self,
        archive_context: str,
        max_retries: int = 3,
    ) -> str | None:
        system = _GENERATION_SYSTEM.format(state_schema=_STATE_SCHEMA)
        user   = _GENERATION_USER_TEMPLATE.format(
            goal            = self.goal,
            archive_context = archive_context,
        )

        for attempt in range(1, max_retries + 1):
            try:
                client = _get_client()
                resp = client.chat.completions.create(
                    model    = MODEL,
                    messages = [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    temperature = 0.5,
                    max_tokens  = 800,
                )
                raw = resp.choices[0].message.content.strip()
                raw = re.sub(r"```python\n?|```\n?", "", raw).strip()
                if "def compute_reward" in raw:
                    idx = raw.index("def compute_reward")
                    raw = raw[idx:]
                if self.verbose:
                    print(f"[designer] Generated reward ({len(raw)} chars)")
                return raw

            except Exception as exc:
                print(
                    f"[designer] Generate attempt {attempt}/{max_retries} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

        return None

    # ── LLM: critique ---------------------------------------------------------

    def _call_critique(
        self,
        reward_code: str,
        metrics: dict,
        trajectory_summary: str,
        generation: int,
        fitness: float,
        max_retries: int = 2,
    ) -> str:
        user = _CRITIQUE_USER_TEMPLATE.format(
            generation         = generation,
            reward_code        = reward_code,
            mean_speed         = metrics.get("mean_speed",         0.0),
            crash_rate         = metrics.get("crash_rate",         0.0),
            mean_overtakes     = metrics.get("mean_overtakes",     0.0),
            completion_rate    = metrics.get("completion_rate",    0.0),
            mean_steps         = metrics.get("mean_steps",         0.0),
            mean_ttc           = metrics.get("mean_ttc",           0.0),
            mean_long_jerk     = metrics.get("mean_long_jerk",     0.0),
            mean_accel         = metrics.get("mean_accel",         0.0),
            total_lane_changes = metrics.get("total_lane_changes", 0),
            fitness            = fitness,
            trajectory_summary = trajectory_summary,
        )

        for attempt in range(1, max_retries + 1):
            try:
                client = _get_client()
                resp = client.chat.completions.create(
                    model    = MODEL,
                    messages = [
                        {"role": "system", "content": _CRITIQUE_SYSTEM},
                        {"role": "user",   "content": user},
                    ],
                    temperature = 0.3,
                    max_tokens  = 500,
                )
                return resp.choices[0].message.content.strip()

            except Exception as exc:
                print(
                    f"[designer] Critique attempt {attempt}/{max_retries} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

        return "(critique unavailable)"

    # ── Manual generation (CLI / bootstrap) -----------------------------------

    def generate_reward(self, goal: str | None = None) -> bool:
        if goal:
            self.goal = goal
        archive_context = self.archive.format_for_llm(k=3)
        new_code = self._call_generate(archive_context)
        if new_code is None:
            return False
        ok, err = validate_reward_code(new_code)
        if not ok:
            print(f"[designer] Validation failed: {err}")
            return False
        self._save_reward_program(new_code)
        return True

    # ── Helpers ---------------------------------------------------------------

    @staticmethod
    def _aggregate_metrics(episode_stats: list[dict]) -> dict:
        n = max(len(episode_stats), 1)
        crashes = sum(1 for s in episode_stats if s.get("collisions", 0) > 0)
        total_overtakes = sum(s.get("total_overtakes", 0) for s in episode_stats)
        return {
            "n_episodes":         n,
            "mean_speed":         sum(s.get("mean_speed",       0) for s in episode_stats) / n,
            "crash_rate":         crashes / n,
            "completion_rate":    1.0 - crashes / n,
            "mean_overtakes":     total_overtakes / n,
            "mean_steps":         sum(s.get("steps",            0) for s in episode_stats) / n,
            "mean_ttc":           sum(s.get("mean_ttc",         0) for s in episode_stats) / n,
            "mean_rel_vel":       sum(s.get("mean_rel_vel",     0) for s in episode_stats) / n,
            "mean_long_jerk":     sum(s.get("mean_long_jerk",   0) for s in episode_stats) / n,
            "mean_lat_jerk":      sum(s.get("mean_lat_jerk",    0) for s in episode_stats) / n,
            "mean_accel":         sum(s.get("mean_accel",       0) for s in episode_stats) / n,
            "total_overtakes":    total_overtakes,
            "total_lane_changes": sum(s.get("total_lane_changes", 0) for s in episode_stats),
            "max_steps":          300,
        }

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
