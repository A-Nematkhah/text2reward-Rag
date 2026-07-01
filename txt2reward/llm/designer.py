"""
LLM-driven Text-to-Reward evolutionary loop.

Orchestrates generation, critique, and evolution. Validation lives in
``txt2reward.llm.validation``; prompts in ``txt2reward.llm.prompts``.
"""

from __future__ import annotations

import os
import re
import time
from collections import deque
from typing import Any, Deque, Mapping

from txt2reward.archive.archive import (
    CURRICULUM_GUIDANCE,
    RewardArchive,
    curriculum_guidance,
    infer_curriculum_phase,
)
from txt2reward.config.llm import (
    CRITIQUE_MAX_RETRIES,
    CRITIQUE_MAX_TOKENS,
    CRITIQUE_TEMPERATURE,
    GENERATION_MAX_RETRIES,
    GENERATION_MAX_TOKENS,
    GENERATION_TEMPERATURE,
)
from txt2reward.config.paths import ARCHIVE_FILE, REWARD_PROGRAM_PATH
from txt2reward.config.training import (
    DEFAULT_EVOLVE_EVERY,
    DEFAULT_FREEZE_RESET_GRACE_WINDOWS,
    DEFAULT_MAX_FREEZE_WINDOWS,
    DEFAULT_WARMUP_EPISODES,
    EVOLVE_MAX_CRASH_RATE,
)
from txt2reward.config.validation import SMOKE_COLLISION_SEVERITY_MAX
from txt2reward.core.log import get_logger
from txt2reward.core.types import CurriculumPhase, EpisodeStats, FitnessMetrics
from txt2reward.llm.aggregation import (
    aggregate_episode_metrics,
    format_metric_trend,
    format_trajectory_samples,
)
from txt2reward.llm.key_manager import call_with_rotation
from txt2reward.llm.prompts import (
    _CRITIQUE_SYSTEM,
    _CRITIQUE_USER_TEMPLATE,
    _GENERATION_SYSTEM,
    _GENERATION_USER_TEMPLATE,
    _REPAIR_USER_TEMPLATE,
    _STATE_SCHEMA,
    DEFAULT_BOOTSTRAP_REWARD_BODY,
    MODEL,
)
from txt2reward.llm.validation import (
    _full_validation_pipeline,
    validate_reward_for_use,
)
from txt2reward.archive.retrieval import reward_code_hash
from txt2reward.reward.wrapper import clear_reward_fn_cache
from txt2reward.sandbox.sandbox import extract_reward_body, validate_reward_code

log = get_logger("designer")

_BOOTSTRAP_BODY_HASH = reward_code_hash(DEFAULT_BOOTSTRAP_REWARD_BODY.strip())


def _is_bootstrap_code(code: str) -> bool:
    """True when ``code`` matches the shipped bootstrap ``compute_reward`` body."""
    if not code or not code.strip():
        return False
    body = extract_reward_body(code) if "def compute_reward" in code else code
    return reward_code_hash(body.strip()) == _BOOTSTRAP_BODY_HASH


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
        f.write('"""\nreward_program.py — bootstrap default (phase-3 hybrid, safe tailgate)\n"""\n\n')
        f.write(DEFAULT_BOOTSTRAP_REWARD_BODY)
    os.replace(tmp, path)
    clear_reward_fn_cache(path)


class RewardDesigner:
    """LLM-driven reward evolution: generate, validate, archive, critique."""

    def __init__(
        self,
        goal: str = "Drive fast, overtake slow vehicles, avoid collisions.",
        evolve_every: int = DEFAULT_EVOLVE_EVERY,
        warmup_episodes: int = DEFAULT_WARMUP_EPISODES,
        evolve_max_crash_rate: float = EVOLVE_MAX_CRASH_RATE,
        max_freeze_windows: int = DEFAULT_MAX_FREEZE_WINDOWS,
        freeze_reset_grace_windows: int = DEFAULT_FREEZE_RESET_GRACE_WINDOWS,
        reward_path: str = REWARD_PROGRAM_PATH,
        archive_path: str = ARCHIVE_FILE,
        initial_episode_count: int = 0,
        initial_last_evolution_index: int = -1,
        verbose: bool = True,
    ):
        """Wire evolution schedule, archive path, and resume counters.

        Args:
            goal: Natural-language driving objective for the LLM.
            evolve_every: Episodes between reward generations (post-warmup).
            warmup_episodes: Episodes before the first LLM generation.
            evolve_max_crash_rate: Freeze LLM evolution while window crash_rate
                is at or above this value (archive + generate skipped).
            max_freeze_windows: After this many consecutive frozen windows,
                force one archive/LLM evolution attempt anyway.
            freeze_reset_grace_windows: While crash_rate is high, keep a newly
                deployed non-bootstrap reward for this many frozen windows
                before reverting to bootstrap (0 = revert on first freeze).
            reward_path: Hot-reloaded ``reward_program.py`` path.
            archive_path: JSON archive for generations and metrics.
            initial_episode_count: Resume episode counter from a prior log.
            initial_last_evolution_index: Resume evolution boundary index.
            verbose: Log designer progress when True.

        Side effects:
            Loads archive and reward program from disk; may reconcile disk
            with archive on startup.
        """
        self.goal = goal
        self.evolve_every = evolve_every
        self.warmup_episodes = warmup_episodes
        self.evolve_max_crash_rate = float(evolve_max_crash_rate)
        self.max_freeze_windows = max(1, int(max_freeze_windows))
        self.freeze_reset_grace_windows = max(0, int(freeze_reset_grace_windows))
        self.reward_path = reward_path
        self.verbose = verbose

        self.archive = RewardArchive(archive_path)

        self._episode_stats: list[EpisodeStats | Mapping[str, Any]] = []
        self._episode_count = max(0, int(initial_episode_count))
        self._last_evolution_index = int(initial_last_evolution_index)
        self._consecutive_frozen_windows = 0
        self._deploy_grace_remaining = 0

        _WIN = 10
        self._policy_buf: Deque[dict] = deque(maxlen=_WIN)

        self._current_code: str = ""
        self._active_generation = 0
        self._last_evolution_metrics: dict[str, Any] | None = None
        self._current_code = self._load_current_code()
        self._reconcile_disk_with_archive()
        self._sync_active_generation()

        if self.verbose:
            log.info(
                f"[designer] Text-to-Reward | goal='{goal[:60]}' | "
                f"evolve_every={evolve_every} | warmup={warmup_episodes} | "
                f"evolve_max_crash={self.evolve_max_crash_rate:.0%} | "
                f"max_freeze_windows={self.max_freeze_windows} | "
                f"freeze_reset_grace={self.freeze_reset_grace_windows} | "
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
                log.info(
                    "[designer] Keeping reward on disk — it differs from archive "
                    f"generation {latest_entry['generation']} and takes precedence."
                )
            return

        if archive_usable:
            if self.verbose:
                log.info(
                    "[designer] reward_program.py missing or placeholder — "
                    f"restoring from archive generation {latest_entry['generation']}."
                )
            self._restore_from_archive_entry(latest_entry)

    def _restore_from_archive_entry(self, entry: Mapping[str, Any]) -> None:
        """Restore reward_program.py from archive after full validation pipeline."""
        restored_code = entry["reward_code"]

        ok, err = validate_reward_for_use(restored_code)

        if ok:
            self._save_reward_program(restored_code)
            return

        reason = err
        gen = entry.get("generation", "?")
        log.warning(
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

    def get_last_evolution_metrics(self) -> dict[str, Any] | None:
        """Metrics from the most recent evolution window (after warmup)."""
        return self._last_evolution_metrics

    # ── Code management -------------------------------------------------------

    def _load_current_code(self) -> str:
        if os.path.exists(self.reward_path):
            try:
                with open(self.reward_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as ex:
                if self.verbose:
                    log.info(f"[designer] Could not read reward program '{self.reward_path}': {ex} — treating as empty")
        return ""

    def _save_reward_program(self, code: str, generation_label: int | None = None) -> None:
        """
        Writes `code` to disk as the reward program currently in effect.
        """
        gen_label = self._active_generation if generation_label is None else generation_label
        tmp = self.reward_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(f'"""\nreward_program.py — generation {gen_label}\n"""\n\n')
            f.write(code)
        os.replace(tmp, self.reward_path)
        clear_reward_fn_cache(self.reward_path)
        self._current_code = code
        self._active_generation = gen_label
        if not _is_bootstrap_code(code):
            self._deploy_grace_remaining = self.freeze_reset_grace_windows
        log.info(f"[designer] reward_program.py updated (generation {gen_label})")

    # ── PPO policy metrics ----------------------------------------------------

    def push_policy_metrics(
        self,
        entropy: float,
        value_loss: float,
        policy_loss: float,
        explained_variance: float,
    ) -> None:
        """Buffer one PPO rollout's health metrics for critique context."""
        self._policy_buf.append(
            {
                "entropy": entropy,
                "value_loss": value_loss,
                "policy_loss": policy_loss,
                "explained_variance": explained_variance,
            }
        )

    def get_policy_snapshot(self) -> dict | None:
        """Mean PPO metrics over the recent rollout window, or ``None`` if empty."""
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

    def accumulate_episode(self, stats: EpisodeStats | Mapping[str, Any]) -> None:
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

        metrics = aggregate_episode_metrics(window_stats)
        self._last_evolution_metrics = dict(metrics)
        current_gen = self._active_generation
        phase = metrics.get("curriculum_phase", infer_curriculum_phase(metrics))

        if self.verbose:
            log.info(
                f"\n[designer] Generation {current_gen} | "
                f"total_episodes={self._episode_count} | window={len(window_stats)} | "
                f"speed={metrics.get('mean_speed', 0):.2f} m/s | "
                f"crash={metrics.get('crash_rate', 0):.1%} | "
                f"overtakes={metrics.get('mean_overtakes', 0):.2f}/ep | "
                f"curriculum={phase}"
            )

        current_code = self._current_code or self._load_current_code()
        crash_rate = float(metrics.get("crash_rate", 1.0))
        freeze_due_to_crash = (
            bool(current_code)
            and not _is_placeholder_code(current_code)
            and crash_rate >= self.evolve_max_crash_rate
        )

        if freeze_due_to_crash:
            self._consecutive_frozen_windows += 1
            if not _is_bootstrap_code(current_code):
                if self._deploy_grace_remaining > 0:
                    self._deploy_grace_remaining -= 1
                    if self.verbose:
                        log.info(
                            "[designer] Freeze grace — keeping deployed reward "
                            "(%s window(s) remaining before bootstrap revert)",
                            self._deploy_grace_remaining,
                        )
                else:
                    write_default_reward_program(self.reward_path)
                    self._current_code = self._load_current_code()
                    current_code = self._current_code
                    self._deploy_grace_remaining = 0
                    if self.verbose:
                        log.info(
                            "[designer] Reset reward_program.py to bootstrap default "
                            "(crash_rate above freeze threshold; non-bootstrap reward discarded)"
                        )

            force_evolve = self._consecutive_frozen_windows >= self.max_freeze_windows
            if not force_evolve:
                if self.verbose:
                    log.info(
                        "[designer] Evolution frozen — crash_rate=%.1f%% >= %.0f%% threshold "
                        "(%s/%s windows); training current reward without archive/LLM update",
                        crash_rate * 100.0,
                        self.evolve_max_crash_rate * 100.0,
                        self._consecutive_frozen_windows,
                        self.max_freeze_windows,
                    )
                self._last_evolution_metrics = dict(metrics)
                self._episode_stats = overflow_stats
                return False

            self._consecutive_frozen_windows = 0
            if self.verbose:
                log.info(
                    "[designer] Forcing evolution after %s consecutive frozen windows "
                    "(crash_rate=%.1f%%)",
                    self.max_freeze_windows,
                    crash_rate * 100.0,
                )
        else:
            self._consecutive_frozen_windows = 0

        if not current_code or _is_placeholder_code(current_code):
            log.warning(
                "[designer] WARNING: no usable reward code on disk — "
                "skipping archive (will not pollute RAG with placeholder)."
            )
            archive_context = self.archive.format_for_llm(
                k=3,
                curriculum_phase=phase,
            )
            new_code = self._call_generate_with_repair(
                archive_context,
                curriculum_phase=phase,
            )
            if new_code is None:
                log.warning("[designer] LLM generation failed -- keeping current reward.")
            else:
                self._save_reward_program(new_code, generation_label=current_gen)
            self._episode_stats = overflow_stats
            return new_code is not None

        # ── 1+2. Archive the program that just ran ───────────────────────────
        previous_entry = self.archive.get_latest()
        trend_summary = format_metric_trend(metrics, previous_entry)

        entry = self.archive.add_entry(
            reward_code=current_code,
            metrics=metrics,
            critique="",
        )
        self._last_evolution_metrics = dict(entry["metrics"])
        self._last_evolution_metrics["fitness"] = entry["fitness"]

        # ── 3+4. Critique the entry we just archived ─────────────────────────
        traj_summary = format_trajectory_samples(window_stats[-5:])
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
                log.info(f"[designer] Critique stored for generation {entry['generation']}")

        # ── 5+6. Generate, validate, smoke-test, and save ────────────────────
        # Improvement #5: pass current failure modes for targeted retrieval
        current_failure_modes = entry.get("critique_meta", {}).get("failure_modes", [])
        archive_context = self.archive.format_for_llm(
            k=3,
            current_failure_modes=current_failure_modes,
            curriculum_phase=metrics.get("curriculum_phase", phase),
        )
        new_code = self._call_generate_with_repair(
            archive_context,
            curriculum_phase=metrics.get("curriculum_phase", phase),
        )

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
        max_retries: int = GENERATION_MAX_RETRIES,
        curriculum_phase: CurriculumPhase | str = "survive",
    ) -> str | None:
        """
        Generate a reward function, then validate (AST) + smoke-test (execution,
        two stages -- see _full_validation_pipeline). If either AST validation
        or either smoke-test stage fails, send the error back to the LLM for
        repair. Returns the first code that passes both stages, or None on
        total failure.
        """
        system = _GENERATION_SYSTEM.format(
            state_schema=_STATE_SCHEMA,
            collision_max=SMOKE_COLLISION_SEVERITY_MAX,
        )
        phase = curriculum_phase if curriculum_phase in CURRICULUM_GUIDANCE else "survive"
        user = _GENERATION_USER_TEMPLATE.format(
            goal=self.goal,
            archive_context=archive_context,
            curriculum_phase=phase,
            curriculum_guidance=curriculum_guidance(phase),
        )

        raw: str | None = None
        repair_error: str = ""

        if self.verbose:
            log.info(f"[designer] Generating reward ({max_retries} attempts max)...")

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
                    temperature=GENERATION_TEMPERATURE,
                    max_tokens=GENERATION_MAX_TOKENS,
                )
                raw = resp.choices[0].message.content.strip()
                raw = re.sub(r"```python\n?|```\n?", "", raw).strip()
                if "def compute_reward" in raw:
                    idx = raw.index("def compute_reward")
                    raw = raw[idx:]
            except Exception as exc:
                if self.verbose:
                    log.info(f"[designer] attempt {attempt}/{max_retries}: API error — {type(exc).__name__}: {exc}")
                if attempt < max_retries:
                    time.sleep(2**attempt)
                raw = None
                repair_error = f"API error: {exc}"
                continue

            # ── Structural validation (AST) ───────────────────────────────
            ok, err = validate_reward_code(raw)
            if not ok:
                if self.verbose:
                    log.info(f"[designer] attempt {attempt}/{max_retries}: AST fail — {err}")
                repair_error = f"Structural validation error: {err}"
                continue

            # ── Smoke-test: Stage A (fast) then Stage B (full bank) ────────
            smoke_ok, smoke_err, smoke_console = _full_validation_pipeline(
                raw,
                curriculum_phase=phase,
            )
            if not smoke_ok:
                if self.verbose:
                    log.info(f"[designer] attempt {attempt}/{max_retries}: smoke fail — {smoke_console}")
                repair_error = smoke_err
                continue

            if self.verbose:
                log.info(f"[designer] attempt {attempt}/{max_retries}: accepted ({len(raw)} chars)")
            # Both checks passed — return the valid code.
            return raw

        if self.verbose:
            from txt2reward.llm.validation import smoke_gate_failure_counts

            counts = smoke_gate_failure_counts()
            extra = f" | gate failures: {counts}" if counts else ""
            log.info(
                f"[designer] evolution skipped — all {max_retries} attempts failed smoke-test; "
                f"keeping current reward{extra}"
            )
        return None

    # ── LLM: critique ---------------------------------------------------------

    def _call_critique(
        self,
        reward_code: str,
        metrics: FitnessMetrics | Mapping[str, Any],
        trajectory_summary: str,
        generation: int,
        fitness: float,
        trend_summary: str = "(no previous generation to compare against)",
        max_retries: int = CRITIQUE_MAX_RETRIES,
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
            near_miss_rate=metrics.get("near_miss_rate", 0.0),
            safe_overtake_ratio=metrics.get("safe_overtake_ratio", 0.0),
            lane_change_rate=metrics.get("lane_change_rate", 0.0),
            curriculum_phase=metrics.get("curriculum_phase", "survive"),
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
                    temperature=CRITIQUE_TEMPERATURE,
                    max_tokens=CRITIQUE_MAX_TOKENS,
                )
                return (resp.choices[0].message.content or "").strip()

            except Exception as exc:
                log.warning(f"[designer] Critique attempt {attempt}/{max_retries} failed: {type(exc).__name__}: {exc}")
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

        latest = self.archive.get_latest()
        bootstrap_phase = "survive"
        if latest:
            bootstrap_phase = latest.get("metrics", {}).get(
                "curriculum_phase",
                infer_curriculum_phase(latest.get("metrics", {})),
            )

        archive_context = self.archive.format_for_llm(
            k=3,
            curriculum_phase=bootstrap_phase,
        )
        new_code = self._call_generate_with_repair(
            archive_context,
            curriculum_phase=bootstrap_phase,
        )

        if new_code is None:
            log.warning("[designer] Bootstrap generation failed — no reward program written.")
            return False

        # Both structural validation and smoke-test passed inside
        # _call_generate_with_repair, so it's safe to write to disk.
        self._save_reward_program(new_code)
        return True

    # ── Helpers (backward-compatible static aliases) ───────────────────────────

    @staticmethod
    def _aggregate_metrics(episode_stats: list[EpisodeStats]) -> FitnessMetrics:
        return aggregate_episode_metrics(episode_stats)

    @staticmethod
    def _format_trend(
        current_metrics: FitnessMetrics | Mapping[str, object],
        previous_entry: Mapping[str, object] | None,
    ) -> str:
        return format_metric_trend(current_metrics, previous_entry)

    @staticmethod
    def _format_trajectory_samples(episode_stats: list[EpisodeStats]) -> str:
        return format_trajectory_samples(episode_stats)
