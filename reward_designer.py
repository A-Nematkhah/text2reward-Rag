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
        # NOTE: generation is NEVER tracked as an independent counter.
        # It is always derived from len(self.archive.entries) — this is the
        # single source of truth, fixing a bug where an independent counter
        # could drift out of sync with the archive (e.g. if add_entry was
        # skipped once, the counter would keep incrementing forever while
        # the archive silently stopped growing).

        _WIN = 10
        self._policy_buf: Deque[dict] = deque(maxlen=_WIN)

        self._current_code: str = self._load_current_code()

        # ── Reconcile disk vs archive ───────────────────────────────────────
        # Covers the case where Drive restore brought back an archive with
        # entries but the reward_program.py file is missing or doesn't match
        # the latest archived code (e.g. only one of the two files was
        # restored, or the file was wiped between runs). Without this check,
        # the wrapper would keep running stale/placeholder code while the
        # archive thinks a different program is "current", silently
        # reintroducing the same kind of drift this refactor fixes.
        latest_entry = self.archive.get_latest()
        if latest_entry is not None and self._current_code != latest_entry["reward_code"]:
            if self.verbose:
                print(
                    "[designer] reward_program.py is missing or out of sync "
                    f"with archive generation {latest_entry['generation']} — "
                    "restoring it from the archive."
                )
            self._save_reward_program(latest_entry["reward_code"])

        if self.verbose:
            print(
                f"[designer] Text-to-Reward | goal='{goal[:60]}' | "
                f"evolve_every={evolve_every} | warmup={warmup_episodes} | "
                f"archive={len(self.archive.entries)} entries | "
                f"generation={self.generation}"
            )

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
        """
        One evolutionary step. Order of operations matters:

          1. Aggregate metrics for the reward program that JUST RAN
             (self._current_code, currently on disk as reward_path).
          2. Archive it UNCONDITIONALLY (add_entry). This is the only
             place generations are created — there is no separate counter
             that can drift out of sync with the archive.
          3. Critique the entry just archived (always the latest one —
             never a stale reference captured before the archive write).
          4. Store the critique back onto that same entry.
          5. Generate + validate an improved reward program using the
             now-updated archive as RAG context.
          6. Only on successful validation, save the new program to disk.
             On any failure, the previous program stays in effect and the
             archive already has a clean, critiqued record of it.
        """
        if not self._episode_stats:
            return False

        metrics = self._aggregate_metrics(self._episode_stats)
        current_gen = self.generation   # generation index BEFORE this archive write

        if self.verbose:
            print(
                f"\n[designer] Generation {current_gen} | "
                f"episodes={len(self._episode_stats)} | "
                f"speed={metrics.get('mean_speed', 0):.2f} m/s | "
                f"crash={metrics.get('crash_rate', 0):.1%} | "
                f"overtakes={metrics.get('mean_overtakes', 0):.2f}/ep"
            )

        # ── 1+2. Archive the program that just ran — unconditional ────────────
        current_code = self._current_code or self._load_current_code()
        if not current_code:
            # This should never happen in practice (reward_program.py is
            # always written before training starts), but fail loudly
            # instead of silently skipping the archive write, which is
            # exactly the bug that previously caused the archive to stop
            # growing while the on-disk generation label kept incrementing.
            print(
                "[designer] WARNING: no current reward code found to archive "
                "(reward_program.py missing or unreadable) — using fallback "
                "placeholder so the archive stays in sync."
            )
            current_code = (
                "def compute_reward(state):\n"
                "    return 0.0  # placeholder: original code was unavailable\n"
            )

        # Capture the previous entry BEFORE add_entry mutates the archive,
        # so the trend comparison is against the generation that actually
        # preceded this one.
        previous_entry = self.archive.get_latest()
        trend_summary   = self._format_trend(metrics, previous_entry)

        entry = self.archive.add_entry(
            reward_code = current_code,
            metrics     = metrics,
            critique    = "",
        )

        # ── 3+4. Critique the entry we just archived ───────────────────────────
        traj_summary = self._format_trajectory_samples(self._episode_stats[-5:])
        critique = self._call_critique(
            reward_code        = entry["reward_code"],
            metrics            = metrics,
            trajectory_summary = traj_summary,
            generation          = entry["generation"],
            fitness             = entry["fitness"],
            trend_summary        = trend_summary,
        )
        if critique:
            self.archive.update_critique(entry["generation"], critique)
            if self.verbose:
                print(f"[designer] Critique stored for generation {entry['generation']}")

        # ── 5. Generate an improved reward function ─────────────────────────────
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

        # ── 6. Save the new program. self.generation is now len(entries),
        #      which already reflects the entry we just archived above —
        #      no separate counter to increment or risk drifting out of sync.
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
        trend_summary: str = "(no previous generation to compare against)",
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
            trend_summary       = trend_summary,
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

    # ── Manual generation (CLI / bootstrap) -------------------------------------

    def generate_reward(self, goal: str | None = None) -> bool:
        """
        Bootstraps an initial reward program before any training/evaluation
        has happened. Used by train.py's --bootstrap step.

        Deliberately does NOT call archive.add_entry() here: there are no
        real metrics yet (no episodes have run), so archiving now would
        create a generation 0 entry with fake/empty metrics. Instead this
        only updates reward_program.py on disk and self._current_code; the
        FIRST call to _evolve() (after warmup_episodes) will archive this
        exact code together with its real, measured metrics. This keeps
        len(archive.entries) as the single source of truth for `generation`
        with no separate counter that could drift out of sync.
        """
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
            cur_v  = current_metrics.get(key, 0.0)
            prev_v = prev.get(key, 0.0)
            return fmt.format(cur_v - prev_v)

        speed_delta     = _delta("mean_speed")
        overtake_delta  = _delta("mean_overtakes")
        crash_delta     = _delta("crash_rate", "{:+.1%}")

        warning = ""
        speed_dropped    = current_metrics.get("mean_speed", 0.0)     < prev.get("mean_speed", 0.0)
        overtakes_dropped = current_metrics.get("mean_overtakes", 0.0) < prev.get("mean_overtakes", 0.0)
        crash_improved   = current_metrics.get("crash_rate", 1.0)     < prev.get("crash_rate", 1.0)
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
