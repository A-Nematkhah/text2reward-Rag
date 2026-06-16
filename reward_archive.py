"""
reward_archive.py
─────────────────
Persistent archive of every generated reward program.

Each entry:
  {
    "generation"   : int          — generation index (0-based)
    "reward_code"  : str          — full Python source of compute_reward()
    "metrics"      : dict         — evaluation metrics after PPO training
    "fitness"      : float        — scalar fitness score
    "critique"     : str          — LLM critique of this reward
    "timestamp"    : str          — ISO-8601 creation time
  }

Fitness function (scalar ∈ [0, ∞)):
  fitness = (
      w_speed    * mean_speed_norm       +
      w_overtake * overtake_rate_norm    +
      w_lane     * lane_efficiency       +
      w_safety   * (1 - collision_rate)  +
      w_complete * completion_rate
  ) * collision_penalty

  where collision_penalty = exp(-5 * collision_rate)
  (sharp drop toward zero for high crash rates)

RAG-style retrieval
───────────────────
  top_k = archive.get_top_k(k=3)
  returns the k entries with highest fitness, formatted for LLM context.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any

ARCHIVE_FILE = "reward_archive.json"

# ── Fitness weights ────────────────────────────────────────────────────────────
_FITNESS_WEIGHTS = {
    "w_speed":    0.30,   # mean speed contribution
    "w_overtake": 0.25,   # overtaking rate
    "w_lane":     0.15,   # lane efficiency (steps / max_steps)
    "w_safety":   0.20,   # crash-free rate
    "w_complete": 0.10,   # episode completion
}

# Normalisation references
_SPEED_REF    = 30.0   # m/s — speed at which speed score = 1.0
_OVERTAKE_REF = 10.0   # overtakes per episode at which overtake score = 1.0


# ── Fitness computation ────────────────────────────────────────────────────────

def compute_fitness(metrics: dict[str, Any]) -> float:
    """
    Computes a scalar fitness score from evaluation metrics.

    Parameters
    ──────────
    metrics : dict with keys from evaluate_agent() output:
        mean_speed      float   m/s
        crash_rate      float   [0,1]
        mean_overtakes  float   overtakes/episode
        mean_steps      float   steps/episode
        max_steps       int     episode step limit (default 300)
        completion_rate float   fraction of episodes not ending in crash

    Returns fitness ∈ [0, 1+] (uncapped, but typically 0–1).
    """
    crash_rate      = float(metrics.get("crash_rate",      0.5))
    mean_speed      = float(metrics.get("mean_speed",       0.0))
    mean_overtakes  = float(metrics.get("mean_overtakes",   0.0))
    mean_steps      = float(metrics.get("mean_steps",       0.0))
    max_steps       = float(metrics.get("max_steps",      300.0))
    completion_rate = float(metrics.get("completion_rate",  0.5))

    # Component scores ∈ [0, 1]
    speed_score    = min(1.0, mean_speed / max(_SPEED_REF, 1.0))
    overtake_score = min(1.0, mean_overtakes / _OVERTAKE_REF)
    lane_score     = min(1.0, mean_steps / max(max_steps, 1.0))
    safety_score   = 1.0 - crash_rate
    complete_score = completion_rate

    weighted = (
        _FITNESS_WEIGHTS["w_speed"]    * speed_score    +
        _FITNESS_WEIGHTS["w_overtake"] * overtake_score +
        _FITNESS_WEIGHTS["w_lane"]     * lane_score     +
        _FITNESS_WEIGHTS["w_safety"]   * safety_score   +
        _FITNESS_WEIGHTS["w_complete"] * complete_score
    )

    # Sharp exponential penalty for high crash rates
    collision_penalty = math.exp(-5.0 * crash_rate)

    return round(weighted * collision_penalty, 4)


# ── Archive class ─────────────────────────────────────────────────────────────

class RewardArchive:
    """
    Persistent store for reward programs, metrics, fitness, and critiques.

    All writes are atomic (write-to-tmp then rename).
    """

    def __init__(self, path: str = ARCHIVE_FILE):
        self.path    = path
        self.entries: list[dict[str, Any]] = []
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.entries = data.get("entries", [])
            print(
                f"[archive] Loaded {len(self.entries)} entries from '{self.path}'"
            )
        except Exception as e:
            print(f"[archive] Failed to load '{self.path}': {e} — starting fresh")
            self.entries = []

    def save(self) -> None:
        """Atomic JSON write."""
        tmp = self.path + ".tmp"
        data = {
            "meta": {
                "total_generations": len(self.entries),
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            "entries": self.entries,
        }
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def add_entry(
        self,
        reward_code: str,
        metrics: dict[str, Any],
        critique: str = "",
    ) -> dict[str, Any]:
        """
        Adds a new reward entry. Computes fitness automatically.

        Returns the new entry dict.
        """
        fitness = compute_fitness(metrics)
        entry: dict[str, Any] = {
            "generation":  len(self.entries),
            "reward_code": reward_code,
            "metrics":     dict(metrics),
            "fitness":     fitness,
            "critique":    critique,
            "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.entries.append(entry)
        self.save()
        print(
            f"[archive] Generation {entry['generation']} saved | "
            f"fitness={fitness:.4f} | "
            f"crash_rate={metrics.get('crash_rate', '?'):.1%} | "
            f"speed={metrics.get('mean_speed', 0):.1f} m/s"
        )
        return entry

    def update_critique(self, generation: int, critique: str) -> None:
        """Updates the critique text for an existing generation."""
        for entry in self.entries:
            if entry["generation"] == generation:
                entry["critique"] = critique
                self.save()
                return
        print(f"[archive] Warning: generation {generation} not found for critique update")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_top_k(self, k: int = 3) -> list[dict[str, Any]]:
        """Returns the k entries with highest fitness score."""
        return sorted(self.entries, key=lambda e: e["fitness"], reverse=True)[:k]

    def get_latest(self) -> dict[str, Any] | None:
        """Returns the most recently added entry."""
        return self.entries[-1] if self.entries else None

    def get_by_generation(self, gen: int) -> dict[str, Any] | None:
        """Returns a specific generation entry."""
        for entry in self.entries:
            if entry["generation"] == gen:
                return entry
        return None

    def format_for_llm(self, k: int = 3) -> str:
        """
        Formats the top-k entries as a human-readable string for LLM context.

        Used by RewardDesigner to provide RAG-style memory.
        """
        top = self.get_top_k(k)
        if not top:
            return "No previous reward programs in archive."

        lines = ["=== TOP REWARD PROGRAMS FROM ARCHIVE ===\n"]
        for i, entry in enumerate(top):
            m = entry["metrics"]
            lines.append(
                f"--- Generation {entry['generation']} "
                f"(fitness={entry['fitness']:.4f}) ---\n"
                f"Metrics:\n"
                f"  mean_speed     : {m.get('mean_speed', 0):.2f} m/s\n"
                f"  crash_rate     : {m.get('crash_rate', 0):.1%}\n"
                f"  mean_overtakes : {m.get('mean_overtakes', 0):.2f}/ep\n"
                f"  completion_rate: {m.get('completion_rate', 0):.1%}\n"
                f"  mean_steps     : {m.get('mean_steps', 0):.0f}\n"
            )
            if entry.get("critique"):
                lines.append(f"Critique:\n{entry['critique']}\n")

            lines.append(f"Reward Code:\n```python\n{entry['reward_code']}\n```\n")

        return "\n".join(lines)

    def format_latest_for_critique(self) -> str | None:
        """
        Formats the latest entry's code and metrics for critique prompt.
        Returns None if archive is empty.
        """
        entry = self.get_latest()
        if entry is None:
            return None
        m = entry["metrics"]
        return (
            f"Generation {entry['generation']}\n"
            f"Reward Code:\n```python\n{entry['reward_code']}\n```\n\n"
            f"Evaluation Metrics:\n"
            f"  mean_speed      : {m.get('mean_speed',      0):.2f} m/s\n"
            f"  crash_rate      : {m.get('crash_rate',      0):.1%}\n"
            f"  mean_overtakes  : {m.get('mean_overtakes',  0):.2f} per episode\n"
            f"  completion_rate : {m.get('completion_rate', 0):.1%}\n"
            f"  mean_steps      : {m.get('mean_steps',      0):.0f}\n"
            f"  mean_ttc        : {m.get('mean_ttc',        0):.2f} s\n"
            f"  mean_long_jerk  : {m.get('mean_long_jerk',  0):.3f} m/s³\n"
            f"  mean_lat_jerk   : {m.get('mean_lat_jerk',   0):.3f} m/s³\n"
            f"  mean_accel      : {m.get('mean_accel',      0):.3f} m/s²\n"
            f"  fitness         : {entry['fitness']:.4f}\n"
        )

    # ── Stats ─────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        if not self.entries:
            return "Archive is empty."
        fitnesses = [e["fitness"] for e in self.entries]
        best = max(self.entries, key=lambda e: e["fitness"])
        return (
            f"Archive: {len(self.entries)} generations | "
            f"best fitness={best['fitness']:.4f} (gen {best['generation']}) | "
            f"avg fitness={sum(fitnesses)/len(fitnesses):.4f}"
        )
