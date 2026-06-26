"""Persistent reward archive storage and CRUD."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Mapping, cast

from txt2reward.archive.critique import parse_structured_critique
from txt2reward.archive.curriculum import curriculum_guidance
from txt2reward.archive.fitness import compute_fitness, is_passive_driving
from txt2reward.archive.retrieval import (
    _format_entry,
    cached_effective_fitness,
    dedupe_entries_by_code,
    effective_fitness,
    is_crash_farming,
    is_pathological_for_retrieval,
    prefetch_effective_fitness,
    reward_code_hash,
)
from txt2reward.config.fitness import (
    ARCHIVE_FAILED_MAX_FITNESS,
    ARCHIVE_MIN_TOP_FITNESS,
    FITNESS_VERSION_DEFAULT,
)
from txt2reward.config.paths import ARCHIVE_FILE
from txt2reward.core.log import get_logger
from txt2reward.core.metrics import enrich_fitness_metrics
from txt2reward.core.types import ArchiveEntry, FitnessMetrics

log = get_logger("archive")

# ── Archive class ─────────────────────────────────────────────────────────────


class RewardArchive:
    """
    Persistent store for reward programs, metrics, fitness, and critiques.
    All writes are atomic (write-to-tmp then rename).

    Improvements #3 & #5: enriched retrieval API for archive-guided hill climbing.
    """

    def __init__(self, path: str = ARCHIVE_FILE):
        """Load an existing archive from disk or start empty.

        Args:
            path: JSON file path (default from ``config.paths.ARCHIVE_FILE``).

        Side effects:
            Reads ``path`` when it exists; logs load warnings on parse failure.
        """
        self.path = path
        self.entries: list[ArchiveEntry] = []
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.entries = data.get("entries", [])
            # Back-fill critique_meta for legacy entries that lack it
            for e in self.entries:
                if "critique_meta" not in e:
                    e["critique_meta"] = parse_structured_critique(e.get("critique", ""), e.get("metrics", {}))
                if "fitness_version" not in e:
                    e["fitness_version"] = 6
            log.info("[archive] Loaded %s entries from '%s'", len(self.entries), self.path)
        except Exception as ex:
            log.warning("[archive] Failed to load '%s': %s — starting fresh", self.path, ex)
            self.entries = []

    def save(self) -> None:
        """Atomically persist ``entries`` to ``self.path`` (write-tmp + rename)."""
        tmp = self.path + ".tmp"
        data = {
            "meta": {
                "total_generations": len(self.entries),
                "fitness_version": FITNESS_VERSION_DEFAULT,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            "entries": self.entries,
        }
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def add_entry(
        self,
        reward_code: str,
        metrics: FitnessMetrics | Mapping[str, object],
        critique: str = "",
    ) -> ArchiveEntry:
        """Append a generation, compute fitness, and persist to disk.

        Args:
            reward_code: Validated ``compute_reward`` source.
            metrics: Raw episode-window metrics (enriched before scoring).
            critique: Free-text LLM critique for this generation.

        Returns:
            The new ``ArchiveEntry`` (also appended to ``self.entries``).

        Side effects:
            Writes archive JSON via ``save()``; logs a one-line summary.
        """
        prev_metrics = self.entries[-1]["metrics"] if self.entries else None
        generation = len(self.entries)
        enriched = enrich_fitness_metrics(dict(metrics))
        fitness = compute_fitness(
            enriched,
            generation=generation,
            prev_metrics=prev_metrics,
        )
        critique_meta = parse_structured_critique(critique, enriched)
        entry: ArchiveEntry = {
            "generation": generation,
            "reward_code": reward_code,
            "metrics": enriched,
            "fitness": fitness,
            "fitness_version": FITNESS_VERSION_DEFAULT,
            "critique": critique,
            "critique_meta": critique_meta,  # improvement #4
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.entries.append(entry)
        self.save()
        log.info(
            "[archive] Generation %s saved | fitness=%.4f | crash_rate=%s | speed=%.1f m/s | overtakes=%.1f/ep",
            entry["generation"],
            fitness,
            f"{metrics.get('crash_rate', 0):.1%}" if isinstance(metrics.get("crash_rate"), (int, float)) else "?",
            float(enriched.get("mean_speed", 0)),
            float(enriched.get("mean_overtakes", 0)),
        )
        return entry

    def update_critique(self, generation: int, critique: str) -> None:
        """Attach or replace critique text and re-parse ``critique_meta``.

        Side effects:
            Persists when the generation exists; logs a warning otherwise.
        """
        for entry in self.entries:
            if entry["generation"] == generation:
                entry["critique"] = critique
                # Improvement #4: re-parse structured metadata when critique updates
                entry["critique_meta"] = parse_structured_critique(critique, entry.get("metrics", {}))
                self.save()
                return
        log.warning("[archive] Warning: generation %s not found for critique update", generation)

    def remove_generation(self, generation: int) -> bool:
        """Remove a corrupt/invalid archive entry (e.g. failed restore re-validation)."""
        before = len(self.entries)
        self.entries = [e for e in self.entries if e["generation"] != generation]
        if len(self.entries) < before:
            for i, entry in enumerate(self.entries):
                entry["generation"] = i
            self.save()
            log.info("[archive] Removed generation %s from archive", generation)
            return True
        return False

    # ── Core retrieval ────────────────────────────────────────────────────────

    def get_top_k(
        self,
        k: int = 3,
        *,
        min_fitness: float = ARCHIVE_MIN_TOP_FITNESS,
        fitness_cache: dict[int, float] | None = None,
    ) -> list[ArchiveEntry]:
        """
        Returns up to k entries with highest effective fitness, excluding
        near-duplicate code and pathological crash-farming clones when better
        alternatives exist in the archive.
        """
        return self._select_diverse_top(
            self.entries,
            k=k,
            min_fitness=min_fitness,
            fitness_cache=fitness_cache,
        )

    def _select_diverse_top(
        self,
        candidates: list[ArchiveEntry],
        *,
        k: int,
        min_fitness: float,
        fitness_cache: dict[int, float] | None = None,
    ) -> list[ArchiveEntry]:
        if not candidates:
            return []

        def _fit(entry: Mapping[str, Any]) -> float:
            if fitness_cache is not None:
                return cached_effective_fitness(entry, fitness_cache)
            return effective_fitness(entry)

        ranked = sorted(
            candidates,
            key=lambda e: (_fit(e), -float(e.get("metrics", {}).get("crash_rate", 1.0))),
            reverse=True,
        )

        has_non_pathological = any(not is_pathological_for_retrieval(e) for e in ranked)
        pool = ranked
        if has_non_pathological:
            non_path = [e for e in ranked if not is_pathological_for_retrieval(e)]
            above_min = [e for e in non_path if _fit(e) > min_fitness]
            if len(above_min) >= k:
                pool = above_min
            elif non_path:
                pool = non_path

        selected: list[ArchiveEntry] = []
        seen_hashes: set[str] = set()
        seen_crash_bands: set[int] = set()

        def _crash_band(cr: float) -> int:
            if cr >= 0.5:
                return 3
            if cr >= 0.15:
                return 2
            return 1

        for entry in pool:
            code = entry.get("reward_code", "")
            h = reward_code_hash(code)
            if h in seen_hashes:
                continue
            m = entry.get("metrics", {})
            cr = float(m.get("crash_rate", 1.0))
            band = _crash_band(cr)
            # Prefer behavioural spread when filling slots 2..k
            if (
                len(selected) >= 1
                and len(selected) < k
                and band in seen_crash_bands
                and any(
                    _crash_band(float(x.get("metrics", {}).get("crash_rate", 1.0))) not in seen_crash_bands
                    for x in pool
                )
            ):
                continue
            if (
                selected
                and is_crash_farming(m)
                and not is_crash_farming(selected[0].get("metrics", {}))
                and _fit(entry) <= _fit(selected[0]) * 1.05
            ):
                continue
            selected.append(entry)
            seen_hashes.add(h)
            seen_crash_bands.add(band)
            if len(selected) >= k:
                break

        if len(selected) < k:
            for entry in ranked:
                if has_non_pathological and is_pathological_for_retrieval(entry):
                    continue
                h = reward_code_hash(entry.get("reward_code", ""))
                if h in seen_hashes or entry in selected:
                    continue
                selected.append(entry)
                seen_hashes.add(h)
                if len(selected) >= k:
                    break
        return selected[:k]

    def get_latest(self) -> ArchiveEntry | None:
        """Most recent archive entry, or ``None`` when empty."""
        return self.entries[-1] if self.entries else None

    def get_by_generation(self, gen: int) -> ArchiveEntry | None:
        """Lookup entry by generation index, or ``None`` if missing."""
        for entry in self.entries:
            if entry["generation"] == gen:
                return entry
        return None

    # ── Improvement #3: Richer retrieval API ─────────────────────────────────

    def get_recent_rewards(self, k: int = 3) -> list[ArchiveEntry]:
        """Most recently archived k entries (newest first)."""
        return list(reversed(self.entries))[:k]

    def get_failed_rewards(
        self,
        k: int = 3,
        max_fitness: float = ARCHIVE_FAILED_MAX_FITNESS,
        fitness_cache: dict[int, float] | None = None,
    ) -> list[ArchiveEntry]:
        """
        Negative examples for LLM context: low effective fitness, passive-but-safe
        traps, and crash-farming rewards (even when fitness is misleadingly high
        under legacy scoring).
        """

        def _fit(entry: Mapping[str, Any]) -> float:
            if fitness_cache is not None:
                return cached_effective_fitness(entry, fitness_cache)
            return effective_fitness(entry)

        failed = [e for e in self.entries if _fit(e) <= max_fitness]
        passive = [e for e in self.entries if e not in failed and is_passive_driving(e.get("metrics", {}))]
        crash_farm = [
            e
            for e in self.entries
            if e not in failed
            and is_crash_farming(e.get("metrics", {}))
            and not is_passive_driving(e.get("metrics", {}))
        ]
        combined = sorted(failed, key=_fit) + sorted(passive, key=_fit) + sorted(crash_farm, key=lambda e: -_fit(e))
        return cast(list[ArchiveEntry], dedupe_entries_by_code(combined)[:k])

    def get_entries_by_failure_modes(
        self,
        modes: list[str],
        k: int = 3,
        fitness_cache: dict[int, float] | None = None,
    ) -> list[ArchiveEntry]:
        """Entries that share ANY of the listed failure mode tags."""

        def _fit(entry: Mapping[str, Any]) -> float:
            if fitness_cache is not None:
                return cached_effective_fitness(entry, fitness_cache)
            return effective_fitness(entry)

        matched = [
            e for e in self.entries if any(m in e.get("critique_meta", {}).get("failure_modes", []) for m in modes)
        ]
        # Sort: highest fitness first so the LLM sees "least bad" examples
        return sorted(matched, key=_fit, reverse=True)[:k]

    # ── Improvement #5: Archive-guided hill-climbing context ─────────────────

    def format_for_llm(
        self,
        k: int = 3,
        current_failure_modes: list[str] | None = None,
        curriculum_phase: str | None = None,
    ) -> str:
        """
        Improvement #5: richly formatted context for archive-guided hill climbing.

        Sections:
          0) Current curriculum phase (metrics-driven, when provided)
          A) Top-k by fitness
          B) Most recent reward (trend context)
          C) Up to 2 known-failed rewards (negative examples)
          D) Rewards sharing current failure modes (targeted repair context)

        Parameters
        ──────────
        k                     : top-k entries to include in section A
        current_failure_modes : failure modes detected in the LATEST generation,
                                used to surface targeted repair examples (section D)
        curriculum_phase      : metrics-inferred phase for the next LLM generation
        """
        if not self.entries:
            return "No previous reward programs in archive."

        fitness_cache = prefetch_effective_fitness(self.entries)
        lines: list[str] = []

        if curriculum_phase:
            lines.append(
                f"=== CURRENT CURRICULUM PHASE: {curriculum_phase} ===\n{curriculum_guidance(curriculum_phase)}\n"
            )

        # ── A) Top performers ────────────────────────────────────────────────
        top = self.get_top_k(k, fitness_cache=fitness_cache)
        lines.append("=== A) TOP REWARD PROGRAMS (by fitness) ===\n")
        for entry in top:
            lines.append(
                _format_entry(
                    entry,
                    show_code=True,
                    fitness=cached_effective_fitness(entry, fitness_cache),
                )
            )

        # ── B) Most recent ───────────────────────────────────────────────────
        recent = self.get_recent_rewards(1)
        if recent and recent[0]["generation"] not in {e["generation"] for e in top}:
            lines.append("=== B) MOST RECENT REWARD (trend context) ===\n")
            lines.append(
                _format_entry(
                    recent[0],
                    show_code=True,
                    fitness=cached_effective_fitness(recent[0], fitness_cache),
                )
            )

        # ── C) Failed rewards (negative examples) ────────────────────────────
        failed = self.get_failed_rewards(k=2, fitness_cache=fitness_cache)
        if failed:
            lines.append(
                "=== C) FAILED / PASSIVE REWARDS (do NOT repeat these patterns) ===\n"
                "(Includes low-fitness entries AND safe-but-slow passive-driving traps)\n"
            )
            for entry in failed:
                lines.append(
                    _format_entry(
                        entry,
                        show_code=False,
                        fitness=cached_effective_fitness(entry, fitness_cache),
                    )
                )

        # ── D) Similar failure mode examples ─────────────────────────────────
        if current_failure_modes:
            similar = self.get_entries_by_failure_modes(
                current_failure_modes,
                k=2,
                fitness_cache=fitness_cache,
            )
            # Exclude entries already shown in A/B/C
            shown_gens = {e["generation"] for e in top + failed + recent}
            similar = [e for e in similar if e["generation"] not in shown_gens]
            if similar:
                lines.append(f"=== D) REWARDS WITH SIMILAR FAILURE MODES ({', '.join(current_failure_modes)}) ===\n")
                lines.append("These previously showed the same issues — study why they failed:\n")
                for entry in similar:
                    lines.append(
                        _format_entry(
                            entry,
                            show_code=True,
                            fitness=cached_effective_fitness(entry, fitness_cache),
                        )
                    )

        return "\n".join(lines)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """One-line archive statistics for end-of-training logs."""
        if not self.entries:
            return "Archive is empty."
        fitness_cache = prefetch_effective_fitness(self.entries)
        fitnesses = list(fitness_cache.values())
        best = max(self.entries, key=lambda e: cached_effective_fitness(e, fitness_cache))
        best_fit = cached_effective_fitness(best, fitness_cache)
        return (
            f"Archive: {len(self.entries)} generations | "
            f"best fitness={best_fit:.4f} (gen {best['generation']}) | "
            f"avg fitness={sum(fitnesses) / len(fitnesses):.4f} | "
            f"speed range: "
            f"{min(e['metrics'].get('mean_speed', 0) for e in self.entries):.1f}"
            f"–{max(e['metrics'].get('mean_speed', 0) for e in self.entries):.1f} m/s"
        )
