"""
reward_archive.py
─────────────────
Persistent archive of every generated reward program.

Each entry:
  {
    "generation"   : int          — generation index (0-based)
    "reward_code"  : str          — full Python source of compute_reward()
    "metrics"      : dict         — evaluation metrics after PPO training
    "fitness"      : float        — scalar fitness score  ∈ [0, 1]
    "critique"     : str          — LLM critique of this reward
    "timestamp"    : str          — ISO-8601 creation time
  }

═══════════════════════════════════════════════════════════════════════════════
FITNESS FUNCTION  (v3 — TTC reweighted)
═══════════════════════════════════════════════════════════════════════════════

DESIGN GOALS
────────────
  1. Crash-free driving is a hard prerequisite, not just one weighted term.
     A 100 % crash rate must produce near-zero fitness regardless of speed.
  2. Speed must be rewarded only above a meaningful threshold (20 m/s).
     Low-speed "safe" driving must score WORSE than fast driving with some
     crashes — the old exp(-5·crash) formula gave virtually the same ~0.007
     score to every generation when crash_rate ≈ 1.0, blinding the LLM.
  3. Each behavioural dimension (speed, overtaking, comfort, completion) is
     normalised to [0, 1] independently before weighting so the weights have
     an intuitive, comparable interpretation.
  4. The final score is always in [0, 1] so fitness values across generations
     are directly comparable and the LLM gets a clear gradient to follow.
  5. (v3) High speed must not be able to outweigh a dangerously low TTC.
     Under the v2 weights (w_speed=0.35, w_ttc=0.10), a trajectory that
     drives at ~29 m/s while tailgating at TTC≈1.4s (never crashing) scored
     a HIGHER fitness than a fully TTC-safe trajectory at 18 m/s — i.e. the
     reference fitness itself ranked a near-miss above safe driving, purely
     because the speed term's weight dwarfed the TTC term's weight. This
     was discovered via trajectory_bank.py's pairwise consistency gate: no
     LLM-generated compute_reward() could simultaneously satisfy the fixed
     -30.0 collision-penalty rule AND match that ordering, because closing
     the gap would require a continuous TTC penalty large enough to rival
     the collision penalty itself -- a structural conflict in the reference
     fitness, not a quality problem with the generated reward code. v3
     rebalances w_speed down and w_ttc up so proximity/near-miss behaviour
     can no longer be masked by raw speed.

FORMULA
───────
  Per-component scores (all ∈ [0, 1]):

    speed_score    = clip((mean_speed - SPEED_MIN) / (SPEED_REF - SPEED_MIN), 0, 1)
                     ^^ zero below 20 m/s, 1.0 at 30 m/s, capped at 1

    overtake_score = min(1, mean_overtakes / OVERTAKE_REF)
                     ^^ 1.0 at ≥10 overtakes/episode

    comfort_score  = exp(-COMFORT_K · mean_long_jerk)
                     ^^ exponential decay: 1.0 at jerk=0, ≈0.37 at jerk=1, ≈0.02 at jerk=4

    ttc_score      = clip(mean_ttc / TTC_SAFE, 0, 1)
                     ^^ 1.0 when TTC ≥ 5 s (safe headway), 0 when tailgating

    completion_score = completion_rate   (fraction of episodes without crash)

  Weighted combination:

    base = w_speed    · speed_score
         + w_overtake · overtake_score
         + w_comfort  · comfort_score
         + w_ttc      · ttc_score
         + w_complete · completion_score

  Safety gate — two-stage penalty:

    Stage 1 (soft gate): if crash_rate > CRASH_THRESHOLD (0.30):
        safety_factor = exp(-CRASH_K_SOFT · (crash_rate - CRASH_THRESHOLD))
        ^^ gentle slope for crash_rate 0–30 %, then sharp drop above 30 %

    Stage 2 (hard gate): if crash_rate > CRASH_HARD_LIMIT (0.80):
        safety_factor *= HARD_PENALTY_SCALE   (= 0.10)
        ^^ an extra 10× suppression for catastrophically unsafe agents,
           ensuring crash_rate ≈ 1.0 never scores above ~0.01

    fitness = clip(base · safety_factor, 0, 1)

  This gives a smooth, navigable fitness landscape:
    • crash_rate = 0.00  → safety_factor = 1.00  (no penalty)
    • crash_rate = 0.30  → safety_factor = 1.00  (threshold not exceeded)
    • crash_rate = 0.50  → safety_factor ≈ 0.37
    • crash_rate = 0.80  → safety_factor ≈ 0.05  (soft gate only)
    • crash_rate = 1.00  → safety_factor ≈ 0.005 (soft + hard gate)

  NOTE: the safety_gate only ever sees crash_rate. It cannot distinguish a
  trajectory that never crashed but tailgated the entire episode (TTC≈1.4s)
  from one that drove with a fully safe headway -- that distinction is
  carried entirely by ttc_score / w_ttc. This is exactly why w_ttc needed
  to be large enough to matter (see v3 note above): the safety gate is not
  a substitute for a meaningful TTC weight, the two checks catch different
  failure modes (crash_rate catches actual collisions; ttc_score catches
  near-misses that technically never cross the line).

WEIGHTS
───────
  w_speed    = 0.25   high speed matters, but no longer dominates TTC
  w_overtake = 0.25   active overtaking rewards skill
  w_comfort  = 0.10   smooth driving (jerk)
  w_ttc      = 0.25   safe headway maintenance -- raised from 0.10 (v2) so a
                       sustained near-miss (low TTC, no crash) cannot be
                       masked by raw speed; see v3 note above
  w_complete = 0.15   episode completion (surrogate for crash-free rate)

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
# w_speed / w_ttc rebalanced (was 0.35 / 0.10) -- fixes a real inconsistency
# surfaced by trajectory_bank.py's pairwise consistency gate: under the old
# weights, a high-speed tailgating trajectory (e.g. 29 m/s, TTC ~1.4s, never
# crashes) scored a HIGHER reference fitness than slower trajectories with a
# fully safe TTC (e.g. oscillating_lanes at 18 m/s, TTC ~10s), purely because
# speed_score's weight (0.35) dwarfed ttc_score's weight (0.10). That made
# the bank's OWN reference ranking favour proximity over safety in some
# pairs -- a structural conflict that no LLM-generated compute_reward()
# could ever satisfy alongside the fixed -30.0 collision-penalty rule,
# since matching that ordering would require a continuous TTC penalty large
# enough to rival the collision term itself. See the v3 docstring section
# above for the full derivation, and trajectory_bank.py for the gate that
# surfaced this. Verified after the change: zero hard safety-category
# violations (legitimate_overtaking still strictly dominates every unsafe
# category), and deliberately-hackable rewards (explicit proximity bonus,
# or no TTC penalty at all) are still correctly rejected by the gate.
_W = {
    "w_speed": 0.25,
    "w_overtake": 0.25,
    "w_comfort": 0.10,
    "w_ttc": 0.25,
    "w_complete": 0.15,
}
assert abs(sum(_W.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# ── Component normalisation references ────────────────────────────────────────
_SPEED_MIN = 20.0  # m/s — speed below this earns zero speed score
_SPEED_REF = 30.0  # m/s — speed at which speed_score reaches 1.0
_OVERTAKE_REF = 10.0  # overtakes/episode → overtake_score = 1.0
_COMFORT_K = 0.5  # exponential decay rate for jerk penalty
_TTC_SAFE = 5.0  # seconds — TTC above which ttc_score = 1.0

# ── Safety gate parameters ─────────────────────────────────────────────────────
_CRASH_THRESHOLD = 0.30  # crash_rate below this → no soft-gate penalty
_CRASH_K_SOFT = 5.0  # soft-gate decay rate (above threshold)
_CRASH_HARD_LIMIT = 0.80  # crash_rate above this triggers the hard gate
_HARD_PENALTY_SCALE = 0.10  # additional ×0.10 multiplier for catastrophic agents


# ── Component scorers (each returns float ∈ [0, 1]) ──────────────────────────


def _speed_score(mean_speed: float) -> float:
    """
    Zero below SPEED_MIN, linear ramp to 1.0 at SPEED_REF.
    Capped at 1.0 above SPEED_REF.

    Rationale: an agent that drives at 14 m/s must score 0 on this
    component — not 0.47 as in the old formula starting from 0 m/s.
    """
    span = max(_SPEED_REF - _SPEED_MIN, 1e-6)
    return float(max(0.0, min(1.0, (mean_speed - _SPEED_MIN) / span)))


def _overtake_score(mean_overtakes: float) -> float:
    """Linear, capped at OVERTAKE_REF overtakes/episode."""
    return float(min(1.0, mean_overtakes / max(_OVERTAKE_REF, 1e-6)))


def _comfort_score(mean_long_jerk: float) -> float:
    """
    Exponential decay on mean longitudinal jerk magnitude.
    score = exp(-COMFORT_K * jerk)
      jerk = 0   → 1.00  (perfectly smooth)
      jerk = 1   → 0.61
      jerk = 2   → 0.37
      jerk = 4   → 0.14
    """
    jerk = max(0.0, float(mean_long_jerk))
    return float(math.exp(-_COMFORT_K * jerk))


def _ttc_score(mean_ttc: float) -> float:
    """
    Linear ramp from 0 (TTC=0) to 1.0 (TTC ≥ TTC_SAFE).
    Measures average headway safety across the episode.
    """
    return float(min(1.0, max(0.0, mean_ttc / _TTC_SAFE)))


def _safety_gate(crash_rate: float) -> float:
    """
    Two-stage multiplicative safety gate.

    Stage 1 — soft gate (activates above CRASH_THRESHOLD):
      Exponential decay so a 50 % crash rate ≈ halves the score.

    Stage 2 — hard gate (activates above CRASH_HARD_LIMIT):
      Additional ×HARD_PENALTY_SCALE suppression for catastrophic agents.
      Ensures crash_rate ≈ 1.0 always scores near zero.

    Returns a multiplier ∈ (0, 1].
    """
    cr = float(crash_rate)

    # Stage 1: soft gate
    if cr <= _CRASH_THRESHOLD:
        factor = 1.0
    else:
        excess = cr - _CRASH_THRESHOLD
        factor = math.exp(-_CRASH_K_SOFT * excess)

    # Stage 2: hard gate
    if cr > _CRASH_HARD_LIMIT:
        factor *= _HARD_PENALTY_SCALE

    return float(factor)


# ── Public fitness function ───────────────────────────────────────────────────


def compute_fitness(metrics: dict[str, Any]) -> float:
    """
    Computes a scalar fitness score in [0, 1] from evaluation metrics.

    Parameters
    ──────────
    metrics : dict with keys from evaluate_agent() output:
        mean_speed       float   m/s
        crash_rate       float   [0, 1]
        mean_overtakes   float   overtakes/episode
        completion_rate  float   fraction of episodes not ending in crash
        mean_long_jerk   float   mean |longitudinal jerk| m/s³
        mean_ttc         float   mean time-to-collision [s]

    Returns
    ───────
    fitness : float ∈ [0, 1]
        Higher is better. Returns 0.0 if all metrics are missing/zero.
    """
    crash_rate = float(metrics.get("crash_rate", 0.5))
    mean_speed = float(metrics.get("mean_speed", 0.0))
    mean_overtakes = float(metrics.get("mean_overtakes", 0.0))
    mean_long_jerk = float(metrics.get("mean_long_jerk", 0.0))
    mean_ttc = float(metrics.get("mean_ttc", 30.0))
    completion_rate = float(metrics.get("completion_rate", 0.5))

    # ── Component scores ─────────────────────────────────────────────────────
    s_speed = _speed_score(mean_speed)
    s_overtake = _overtake_score(mean_overtakes)
    s_comfort = _comfort_score(mean_long_jerk)
    s_ttc = _ttc_score(mean_ttc)
    s_complete = float(max(0.0, min(1.0, completion_rate)))

    # ── Weighted base score ───────────────────────────────────────────────────
    base = (
        _W["w_speed"] * s_speed
        + _W["w_overtake"] * s_overtake
        + _W["w_comfort"] * s_comfort
        + _W["w_ttc"] * s_ttc
        + _W["w_complete"] * s_complete
    )

    # ── Safety gate ───────────────────────────────────────────────────────────
    gate = _safety_gate(crash_rate)

    fitness = float(max(0.0, min(1.0, base * gate)))

    return round(fitness, 4)


# ── Archive class ─────────────────────────────────────────────────────────────


class RewardArchive:
    """
    Persistent store for reward programs, metrics, fitness, and critiques.
    All writes are atomic (write-to-tmp then rename).
    """

    def __init__(self, path: str = ARCHIVE_FILE):
        self.path = path
        self.entries: list[dict[str, Any]] = []
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.entries = data.get("entries", [])
            print(f"[archive] Loaded {len(self.entries)} entries from '{self.path}'")
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

    # ── CRUD ─────────────────────────────────────────────────────────────────

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
            "generation": len(self.entries),
            "reward_code": reward_code,
            "metrics": dict(metrics),
            "fitness": fitness,
            "critique": critique,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.entries.append(entry)
        self.save()
        print(
            f"[archive] Generation {entry['generation']} saved | "
            f"fitness={fitness:.4f} | "
            f"crash_rate={metrics.get('crash_rate', '?'):.1%} | "
            f"speed={metrics.get('mean_speed', 0):.1f} m/s | "
            f"overtakes={metrics.get('mean_overtakes', 0):.1f}/ep"
        )
        return entry

    def update_critique(self, generation: int, critique: str) -> None:
        """Updates the critique text for an existing generation."""
        for entry in self.entries:
            if entry["generation"] == generation:
                entry["critique"] = critique
                self.save()
                return
        print(f"[archive] Warning: generation {generation} not found " "for critique update")

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
        for entry in top:
            m = entry["metrics"]
            lines.append(
                f"--- Generation {entry['generation']} "
                f"(fitness={entry['fitness']:.4f}) ---\n"
                f"Metrics:\n"
                f"  mean_speed     : {m.get('mean_speed',      0):.2f} m/s\n"
                f"  crash_rate     : {m.get('crash_rate',      0):.1%}\n"
                f"  mean_overtakes : {m.get('mean_overtakes',  0):.2f}/ep\n"
                f"  mean_long_jerk : {m.get('mean_long_jerk',  0):.3f} m/s³\n"
                f"  mean_ttc       : {m.get('mean_ttc',        0):.2f} s\n"
                f"  completion_rate: {m.get('completion_rate', 0):.1%}\n"
                f"  mean_steps     : {m.get('mean_steps',      0):.0f}\n"
            )
            # Component breakdown — helps LLM see WHY this program scored well
            cr = m.get("crash_rate", 0.5)
            lines.append(
                f"Fitness breakdown:\n"
                f"  speed_score    : {_speed_score(m.get('mean_speed', 0)):.3f}\n"
                f"  overtake_score : {_overtake_score(m.get('mean_overtakes', 0)):.3f}\n"
                f"  comfort_score  : {_comfort_score(m.get('mean_long_jerk', 0)):.3f}\n"
                f"  ttc_score      : {_ttc_score(m.get('mean_ttc', 30)):.3f}\n"
                f"  safety_gate    : {_safety_gate(cr):.3f}\n"
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
        cr = m.get("crash_rate", 0.5)
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
            f"\nFitness breakdown:\n"
            f"  speed_score     : {_speed_score(m.get('mean_speed', 0)):.3f}  "
            f"(zero below {_SPEED_MIN} m/s)\n"
            f"  overtake_score  : {_overtake_score(m.get('mean_overtakes', 0)):.3f}\n"
            f"  comfort_score   : {_comfort_score(m.get('mean_long_jerk', 0)):.3f}\n"
            f"  ttc_score       : {_ttc_score(m.get('mean_ttc', 30)):.3f}\n"
            f"  safety_gate     : {_safety_gate(cr):.3f}  "
            f"({'HARD gate active' if cr > _CRASH_HARD_LIMIT else 'soft gate' if cr > _CRASH_THRESHOLD else 'no penalty'})\n"
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
            f"avg fitness={sum(fitnesses)/len(fitnesses):.4f} | "
            f"speed range: "
            f"{min(e['metrics'].get('mean_speed',0) for e in self.entries):.1f}"
            f"–{max(e['metrics'].get('mean_speed',0) for e in self.entries):.1f} m/s"
        )


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Fitness Function Self-Test ===\n")

    scenarios = [
        (
            "Perfect agent",
            {
                "mean_speed": 30.0,
                "crash_rate": 0.00,
                "mean_overtakes": 10.0,
                "mean_long_jerk": 0.5,
                "mean_ttc": 8.0,
                "completion_rate": 1.00,
            },
        ),
        (
            "Good agent",
            {
                "mean_speed": 25.0,
                "crash_rate": 0.05,
                "mean_overtakes": 5.0,
                "mean_long_jerk": 1.0,
                "mean_ttc": 6.0,
                "completion_rate": 0.95,
            },
        ),
        (
            "Mediocre agent",
            {
                "mean_speed": 22.0,
                "crash_rate": 0.30,
                "mean_overtakes": 2.0,
                "mean_long_jerk": 2.0,
                "mean_ttc": 4.0,
                "completion_rate": 0.70,
            },
        ),
        (
            "Current (bad)",
            {
                "mean_speed": 14.0,
                "crash_rate": 1.00,
                "mean_overtakes": 30.0,
                "mean_long_jerk": 1.5,
                "mean_ttc": 2.0,
                "completion_rate": 0.00,
            },
        ),
        (
            "Stationary/safe",
            {
                "mean_speed": 5.0,
                "crash_rate": 0.00,
                "mean_overtakes": 0.0,
                "mean_long_jerk": 0.1,
                "mean_ttc": 30.0,
                "completion_rate": 1.00,
            },
        ),
        (
            "Fast but crashy",
            {
                "mean_speed": 28.0,
                "crash_rate": 0.50,
                "mean_overtakes": 8.0,
                "mean_long_jerk": 1.5,
                "mean_ttc": 3.5,
                "completion_rate": 0.50,
            },
        ),
        (
            "Tailgating (fast, low TTC, no crash)",
            {
                "mean_speed": 29.0,
                "crash_rate": 0.00,
                "mean_overtakes": 0.0,
                "mean_long_jerk": 0.15,
                "mean_ttc": 1.4,
                "completion_rate": 1.00,
            },
        ),
        (
            "Old gen 8 (best)",
            {
                "mean_speed": 11.6,
                "crash_rate": 0.90,
                "mean_overtakes": 35.0,
                "mean_long_jerk": 1.2,
                "mean_ttc": 1.5,
                "completion_rate": 0.10,
            },
        ),
    ]

    print(
        f"{'Scenario':<38} {'fitness':>8}  {'speed_s':>7}  {'overt_s':>7}  " f"{'comf_s':>7}  {'ttc_s':>7}  {'gate':>7}"
    )
    print("─" * 100)

    for name, m in scenarios:
        f = compute_fitness(m)
        ss = _speed_score(m["mean_speed"])
        os = _overtake_score(m["mean_overtakes"])
        cs = _comfort_score(m["mean_long_jerk"])
        ts = _ttc_score(m["mean_ttc"])
        g = _safety_gate(m["crash_rate"])
        print(f"{name:<38} {f:>8.4f}  {ss:>7.3f}  {os:>7.3f}  " f"{cs:>7.3f}  {ts:>7.3f}  {g:>7.3f}")

    print("\n✓ All scenarios computed successfully.")
