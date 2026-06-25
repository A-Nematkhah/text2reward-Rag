"""
reward_archive.py
─────────────────
Persistent archive of every generated reward program.

Each entry:
  {
    "generation"      : int          — generation index (0-based)
    "reward_code"     : str          — full Python source of compute_reward()
    "metrics"         : dict         — evaluation metrics after PPO training
    "fitness"         : float        — scalar fitness score  ∈ [0, 1]
    "critique"        : str          — LLM critique of this reward (free text)
    "critique_meta"   : dict         — structured critique metadata (NEW v4)
    "timestamp"       : str          — ISO-8601 creation time
  }

critique_meta schema (improvement #4 — Structured Failure Modes):
  {
    "failure_modes": list[str]   — e.g. ["tailgating", "passive_driving"]
    "strengths":     list[str]   — e.g. ["high_speed", "good_overtaking"]
    "summary":       str         — one-sentence summary
  }

  Known failure_mode tags:
    tailgating            TTC sustained < 2 s, no crash
    passive_driving       speed < 18 m/s or overtakes dropping
    oscillatory_lane_changes  lane_changes >> overtakes
    acceleration_spam     high jerk/accel, no net speed gain
    stationary_farming    mean_speed < 5 m/s
    reward_hacking        catch-all for shaped_reward >> env_reward

═══════════════════════════════════════════════════════════════════════════════
FITNESS FUNCTION  (v4 — sigmoid speed + robust TTC)
═══════════════════════════════════════════════════════════════════════════════

Improvement #1 — Sigmoid Speed Score
──────────────────────────────────────
The v3 linear speed score compressed all highway-env behaviours into the
narrow 20–30 m/s band and gave identical gradient signals to agents at 22 and
28 m/s. A logistic (sigmoid) function centred at 25 m/s provides:

  speed_score = 1 / (1 + exp(-k * (speed - mid)))

  with k = 0.5, mid = 25.0 (tuned so score≈0.12 at 20 m/s, ≈0.88 at 30 m/s)

Values at key speeds:
  20 m/s → 0.076   (meaningfully penalised, not zero like v3)
  22 m/s → 0.182
  24 m/s → 0.378
  25 m/s → 0.500   (midpoint)
  26 m/s → 0.622
  28 m/s → 0.818
  30 m/s → 0.924

Monotonic, differentiable, [0,1]-bounded, provides real gradient everywhere.

Improvement #2 — Robust TTC Score
───────────────────────────────────
mean_ttc is dangerous: 299 safe steps + 1 tailgating step → mean still ~30s.
v4 adds min_ttc and p10_ttc (10th-percentile TTC) collected per-episode and
computes a composite TTC score:

  ttc_score = 0.4 * norm(mean_ttc) + 0.35 * norm(p10_ttc) + 0.25 * norm(min_ttc)

Each component normalised to [0, 1] with ceiling at TTC_SAFE (5 s).
The p10 term catches episodes with sustained low-TTC patches; min_ttc catches
single catastrophic near-misses.

New metrics required in episode_stats (collected in reward_wrapper.py):
  min_ttc   : float   minimum TTC seen during the episode
  p10_ttc   : float   10th-percentile TTC across all steps

WEIGHTS (v6 — overtake up, ttc down)
────────
  w_speed    = 0.25
  w_overtake = 0.30
  w_comfort  = 0.10
  w_ttc      = 0.20
  w_complete = 0.15

Improvement #6 — Multiplicative passive-driving gate (v6)
──────────────────────────────────────────────────────────
Once crash_rate <= 20%, fitness is multiplied by a two-factor gate:
  gate = max(0.10, speed_factor × overtake_factor)
  speed_factor    = min(1, (mean_speed / 24.0)²)      — sharp below 24 m/s
  overtake_factor = min(1, (mean_overtakes / 1.5)^0.5) — rewarded for any overtaking
Both factors must be near 1.0 for the gate to be near 1.0. A fast-but-passive
agent (26 m/s, 0 overtakes) gets gate=0.10. A slow-but-active agent (18 m/s,
3 overtakes) gets gate=0.10. Only fast + active driving escapes the penalty.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any

ARCHIVE_FILE = "reward_archive.json"

# ── Fitness weights ────────────────────────────────────────────────────────────
_W = {
    "w_speed": 0.25,
    "w_overtake": 0.30,   # raised: overtaking is the clearest active-driving signal
    "w_comfort": 0.10,
    "w_ttc": 0.20,        # lowered to compensate; sum stays 1.0
    "w_complete": 0.15,
}
assert abs(sum(_W.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# ── Component normalisation references ────────────────────────────────────────
_SPEED_MIN = 20.0        # kept for backward-compat; sigmoid replaces linear use
_SPEED_REF = 30.0        # kept for backward-compat
_SPEED_SIGMOID_K = 0.5   # logistic steepness (#1)
_SPEED_SIGMOID_MID = 25.0  # midpoint of logistic (#1)

_OVERTAKE_REF = 5.0   # 3 overtakes/ep now gives 0.60 score, not 0.30
_COMFORT_K = 0.5
_TTC_SAFE = 5.0          # s — normalisation ceiling for all TTC components

# Robust TTC sub-weights (must sum to 1.0) — improvement #2
_TTC_W_MEAN = 0.40
_TTC_W_P10  = 0.35
_TTC_W_MIN  = 0.25
assert abs(_TTC_W_MEAN + _TTC_W_P10 + _TTC_W_MIN - 1.0) < 1e-9

# ── Safety gate parameters ─────────────────────────────────────────────────────
_CRASH_THRESHOLD = 0.30
_CRASH_K_SOFT = 5.0
_CRASH_HARD_LIMIT = 0.80
_HARD_PENALTY_SCALE = 0.10

# ── Passive-driving gate (v6) ─────────────────────────────────────────────────
# When the agent is already safe (low crash_rate), suppress fitness if it trades
# speed/overtaking for survival — the "slow down to stay safe" reward hack.
_PASSIVE_CRASH_CEILING = 0.20   # wider: gate applies even with ~20% crash rate
_PASSIVE_SPEED_MIN = 24.0       # m/s — raised: 20-22 m/s is passive, need 24+
_PASSIVE_OVERTAKE_MIN = 1.5     # overtakes/ep — raised: 0.5 was trivially easy
_PASSIVE_GATE_FLOOR = 0.10      # lowered: passive agent gets ≤10% of base fitness
# Note: _PASSIVE_SPEED_WEIGHT and _PASSIVE_OVERTAKE_WEIGHT are removed.
# The new _passive_driving_gate() uses a multiplicative formula instead.


# ── Component scorers (each returns float ∈ [0, 1]) ──────────────────────────


def _speed_score(mean_speed: float) -> float:
    """
    Improvement #1: logistic (sigmoid) speed score.

    score = 1 / (1 + exp(-k * (speed - mid)))

    Monotonic, differentiable, [0,1]-bounded.
    Provides meaningful gradient across the full 15–35 m/s range instead
    of compressing everything between 20 and 30.
    """
    x = _SPEED_SIGMOID_K * (float(mean_speed) - _SPEED_SIGMOID_MID)
    return float(1.0 / (1.0 + math.exp(-x)))


def _overtake_score(mean_overtakes: float) -> float:
    return float(min(1.0, mean_overtakes / max(_OVERTAKE_REF, 1e-6)))


def _comfort_score(mean_long_jerk: float) -> float:
    jerk = max(0.0, float(mean_long_jerk))
    return float(math.exp(-_COMFORT_K * jerk))


def _ttc_component_norm(ttc_val: float) -> float:
    """Normalise a single TTC value to [0, 1] with ceiling at TTC_SAFE."""
    return float(min(1.0, max(0.0, ttc_val / _TTC_SAFE)))


def _ttc_score(mean_ttc: float, p10_ttc: float = -1.0, min_ttc: float = -1.0) -> float:
    """
    Improvement #2: robust TTC score combining mean, 10th-percentile, and min.

    When p10_ttc / min_ttc are absent (legacy metrics without these fields,
    indicated by value < 0), falls back to mean_ttc only so old archive
    entries remain valid.
    """
    if p10_ttc < 0 or min_ttc < 0:
        # Legacy path: only mean_ttc available
        return _ttc_component_norm(mean_ttc)

    s_mean = _ttc_component_norm(mean_ttc)
    s_p10  = _ttc_component_norm(p10_ttc)
    s_min  = _ttc_component_norm(min_ttc)
    return float(_TTC_W_MEAN * s_mean + _TTC_W_P10 * s_p10 + _TTC_W_MIN * s_min)


def _safety_gate(crash_rate: float) -> float:
    cr = float(crash_rate)
    if cr <= _CRASH_THRESHOLD:
        factor = 1.0
    else:
        factor = math.exp(-_CRASH_K_SOFT * (cr - _CRASH_THRESHOLD))
    if cr > _CRASH_HARD_LIMIT:
        factor *= _HARD_PENALTY_SCALE
    return float(factor)


def is_passive_driving(metrics: dict[str, Any]) -> bool:
    """
    True when the agent is crash-free enough that we should expect active
    driving, but mean speed and/or overtakes are too low.

    Uses v7 thresholds when default fitness version is 7.
    """
    crash_rate = float(metrics.get("crash_rate", 1.0))
    ceiling = _V7_PASSIVE_CRASH_CEIL if FITNESS_VERSION_DEFAULT >= 7 else _PASSIVE_CRASH_CEILING
    speed_min = _V7_PASSIVE_SPEED_MIN if FITNESS_VERSION_DEFAULT >= 7 else _PASSIVE_SPEED_MIN
    ot_min = _V7_PASSIVE_OT_MIN if FITNESS_VERSION_DEFAULT >= 7 else _PASSIVE_OVERTAKE_MIN
    if crash_rate > ceiling:
        return False
    mean_speed = float(metrics.get("mean_speed", 0.0))
    mean_overtakes = float(metrics.get("mean_overtakes", 0.0))
    return mean_speed < speed_min or mean_overtakes < ot_min


def _passive_driving_gate(
    mean_speed: float,
    mean_overtakes: float,
    crash_rate: float,
) -> float:
    """
    Multiplicative gate ∈ [_PASSIVE_GATE_FLOOR, 1.0].

    Inactive (returns 1.0) while crash_rate > _PASSIVE_CRASH_CEILING — don't
    punish an agent still learning basic safety.

    Once crash_rate is low enough, applies TWO independent multiplicative factors:

      speed_factor    = min(1, (mean_speed / _PASSIVE_SPEED_MIN)²)
      overtake_factor = min(1, (mean_overtakes / _PASSIVE_OVERTAKE_MIN)^0.5)
      gate            = max(_PASSIVE_GATE_FLOOR, speed_factor × overtake_factor)

    The product means BOTH speed AND overtaking are required to score well — a
    fast agent that never overtakes and an agent that overtakes once while crawling
    both get heavily penalised. Only genuinely active, fast driving escapes the gate.

    The squared speed term creates a sharp gradient below _PASSIVE_SPEED_MIN
    (e.g. 20 m/s → factor 0.694, 18 m/s → factor 0.563), while the sqrt overtake
    term is softer (any overtaking is better than none; the first overtake matters
    most).

    Example values after this change:
      passive (20 m/s, 0 overtakes, 0% crash) → gate = 0.10  (was 0.50)
      semi-active (22 m/s, 1 overtake, 5% crash) → gate = 0.73
      good (26 m/s, 3 overtakes, 5% crash) → gate = 1.00
    """
    if float(crash_rate) > _PASSIVE_CRASH_CEILING:
        return 1.0

    # Speed factor: squared for sharp gradient below the minimum
    speed_ratio = float(mean_speed) / _PASSIVE_SPEED_MIN
    speed_factor = min(1.0, max(0.0, speed_ratio) ** 2)

    # Overtake factor: sqrt for a softer curve (first overtake is the hardest)
    overtake_ratio = float(mean_overtakes) / max(_PASSIVE_OVERTAKE_MIN, 1e-6)
    overtake_factor = min(1.0, max(0.0, overtake_ratio) ** 0.5)

    combined = speed_factor * overtake_factor
    return float(max(_PASSIVE_GATE_FLOOR, combined))


# ── Fitness v7 parameters ─────────────────────────────────────────────────────
FITNESS_VERSION_DEFAULT = 7

_V7_W = {
    "activity": 0.35,
    "speed": 0.15,
    "overtake": 0.15,
    "lane_eff": 0.05,
    "ttc": 0.15,
    "comfort": 0.15,
}
assert abs(sum(_V7_W.values()) - 1.0) < 1e-9

_V7_SPEED_K = 0.55
_V7_SPEED_MID = 24.0
_V7_OVERTAKE_REF = 2.0
_V7_OVERTAKE_ALPHA = 0.7
_V7_ACTIVITY_SPEED_EXP = 0.6
_V7_ACTIVITY_OT_EXP = 0.4
_V7_LANE_EFF_REF = 0.35

_V7_SAFETY_C0 = 0.10
_V7_SAFETY_LAMBDA = 0.55
_V7_SAFETY_GAMMA = 1.2
_V7_NEAR_MISS_LAMBDA = 0.10
_V7_NEAR_MISS_TTC = 2.0

_V7_PASSIVE_CRASH_CEIL = 0.40
_V7_PASSIVE_SPEED_MIN = 24.0
_V7_PASSIVE_OT_MIN = 1.0
_V7_PASSIVE_LAMBDA = 0.35
_V7_PASSIVE_TARGET = 0.5

_V7_TREND_LAMBDA = 0.15
_V7_CURRICULUM_ETA = 0.8
_V7_CURRICULUM_BANDS = (
    (0.50, 0.35),  # phase A — safety
    (0.35, 0.10),  # phase B — efficiency
    (0.15, 0.05),  # phase C — active overtaking
)

_V7_HARD_CRASH_LIMIT = 0.50
_V7_PASSIVE_CAP = 0.12
_V7_STATIONARY_SPEED_MAX = 5.0
_V7_STATIONARY_PENALTY = 0.30


def _speed_score_v7(mean_speed: float) -> float:
    x = _V7_SPEED_K * (float(mean_speed) - _V7_SPEED_MID)
    return float(1.0 / (1.0 + math.exp(-x)))


def _overtake_score_v7(mean_overtakes: float) -> float:
    ratio = float(mean_overtakes) / max(_V7_OVERTAKE_REF, 1e-6)
    return float(min(1.0, max(0.0, ratio) ** _V7_OVERTAKE_ALPHA))


def _activity_score_v7(mean_speed: float, mean_overtakes: float) -> float:
    s_v = _speed_score_v7(mean_speed)
    s_o = _overtake_score_v7(mean_overtakes)
    return float((s_v ** _V7_ACTIVITY_SPEED_EXP) * (s_o ** _V7_ACTIVITY_OT_EXP))


def _activity_product_v7(mean_speed: float, mean_overtakes: float) -> float:
    speed_factor = min(1.0, (float(mean_speed) / _V7_PASSIVE_SPEED_MIN) ** 2)
    overtake_factor = min(1.0, (float(mean_overtakes) / _V7_PASSIVE_OT_MIN) ** 0.5)
    return float(speed_factor * overtake_factor)


def _lane_efficiency_score(metrics: dict[str, Any]) -> float:
    lane_changes = int(metrics.get("total_lane_changes", 0))
    if lane_changes <= 0:
        return 1.0
    total_overtakes = metrics.get("total_overtakes")
    if total_overtakes is None:
        n_eps = max(int(metrics.get("n_episodes", 1)), 1)
        total_overtakes = float(metrics.get("mean_overtakes", 0.0)) * n_eps
    ratio = min(1.0, float(total_overtakes) / lane_changes)
    return float(min(1.0, ratio / _V7_LANE_EFF_REF))


def _stationary_penalty_v7(mean_speed: float) -> float:
    if float(mean_speed) < _V7_STATIONARY_SPEED_MAX:
        return _V7_STATIONARY_PENALTY
    return 0.0


def _safety_penalty_v7(crash_rate: float) -> float:
    cr = float(crash_rate)
    return float(_V7_SAFETY_LAMBDA * (max(0.0, cr - _V7_SAFETY_C0) ** _V7_SAFETY_GAMMA))


def _near_miss_penalty_v7(min_ttc: float) -> float:
    if min_ttc < 0:
        return 0.0
    return float(_V7_NEAR_MISS_LAMBDA * max(0.0, 1.0 - float(min_ttc) / _V7_NEAR_MISS_TTC))


def _passive_penalty_v7(mean_speed: float, mean_overtakes: float, crash_rate: float) -> float:
    if float(crash_rate) > _V7_PASSIVE_CRASH_CEIL:
        return 0.0
    activity = _activity_product_v7(mean_speed, mean_overtakes)
    return float(_V7_PASSIVE_LAMBDA * max(0.0, _V7_PASSIVE_TARGET - activity))


def _trend_penalty_v7(metrics: dict[str, Any], prev_metrics: dict[str, Any] | None) -> float:
    if not prev_metrics:
        return 0.0
    delta_speed = float(metrics.get("mean_speed", 0.0)) - float(prev_metrics.get("mean_speed", 0.0))
    delta_crash = float(metrics.get("crash_rate", 0.0)) - float(prev_metrics.get("crash_rate", 0.0))
    if delta_speed >= 0.0 or delta_crash >= 0.0:
        return 0.0
    return float(_V7_TREND_LAMBDA * (-delta_speed))


def _curriculum_phase(generation: int, crash_rate: float) -> int:
    if float(crash_rate) > 0.40 or generation <= 1:
        return 0
    if generation <= 4 or float(crash_rate) > 0.15:
        return 1
    return 2


def _curriculum_ceiling_v7(crash_rate: float, phase: int) -> float:
    c_ceil, c_floor = _V7_CURRICULUM_BANDS[phase]
    cr = float(crash_rate)
    if cr >= c_ceil:
        return 0.0
    if cr <= c_floor:
        return 1.0
    span = max(c_ceil - c_floor, 1e-6)
    return float(((c_ceil - cr) / span) ** _V7_CURRICULUM_ETA)


# ── Public fitness functions ──────────────────────────────────────────────────


def compute_fitness_v6(metrics: dict[str, Any]) -> float:
    """Legacy v6 fitness (multiplicative safety × passive gates)."""
    crash_rate      = float(metrics.get("crash_rate", 0.5))
    mean_speed      = float(metrics.get("mean_speed", 0.0))
    mean_overtakes  = float(metrics.get("mean_overtakes", 0.0))
    mean_long_jerk  = float(metrics.get("mean_long_jerk", 0.0))
    mean_ttc        = float(metrics.get("mean_ttc", 30.0))
    p10_ttc         = float(metrics.get("p10_ttc", -1.0))
    min_ttc         = float(metrics.get("min_ttc", -1.0))
    completion_rate = float(metrics.get("completion_rate", 0.5))

    s_speed    = _speed_score(mean_speed)
    s_overtake = _overtake_score(mean_overtakes)
    s_comfort  = _comfort_score(mean_long_jerk)
    s_ttc      = _ttc_score(mean_ttc, p10_ttc, min_ttc)
    s_complete = float(max(0.0, min(1.0, completion_rate)))

    base = (
        _W["w_speed"]    * s_speed
        + _W["w_overtake"] * s_overtake
        + _W["w_comfort"]  * s_comfort
        + _W["w_ttc"]      * s_ttc
        + _W["w_complete"] * s_complete
    )

    safety = _safety_gate(crash_rate)
    passive = _passive_driving_gate(mean_speed, mean_overtakes, crash_rate)
    fitness = float(max(0.0, min(1.0, base * safety * passive)))
    return round(fitness, 4)


def compute_fitness_v7(
    metrics: dict[str, Any],
    *,
    generation: int = 0,
    prev_metrics: dict[str, Any] | None = None,
) -> float:
    """
    Fitness v7 — additive base with bounded penalties and curriculum ceiling.

    Addresses v6 failure modes: transition-ridge peak, slow-to-survive reward,
    and completion/crash double-counting.
    """
    crash_rate = float(metrics.get("crash_rate", 0.5))
    if crash_rate > _V7_HARD_CRASH_LIMIT:
        return 0.01

    mean_speed = float(metrics.get("mean_speed", 0.0))
    mean_overtakes = float(metrics.get("mean_overtakes", 0.0))
    mean_long_jerk = float(metrics.get("mean_long_jerk", 0.0))
    mean_ttc = float(metrics.get("mean_ttc", 30.0))
    p10_ttc = float(metrics.get("p10_ttc", -1.0))
    min_ttc = float(metrics.get("min_ttc", -1.0))

    s_activity = _activity_score_v7(mean_speed, mean_overtakes)
    s_speed = _speed_score_v7(mean_speed)
    s_overtake = _overtake_score_v7(mean_overtakes)
    s_lane = _lane_efficiency_score(metrics)
    s_ttc = _ttc_score(mean_ttc, p10_ttc, min_ttc)
    s_comfort = _comfort_score(mean_long_jerk)

    base = (
        _V7_W["activity"] * s_activity
        + _V7_W["speed"] * s_speed
        + _V7_W["overtake"] * s_overtake
        + _V7_W["lane_eff"] * s_lane
        + _V7_W["ttc"] * s_ttc
        + _V7_W["comfort"] * s_comfort
    )

    penalty = (
        _safety_penalty_v7(crash_rate)
        + _near_miss_penalty_v7(min_ttc)
        + _passive_penalty_v7(mean_speed, mean_overtakes, crash_rate)
        + _stationary_penalty_v7(mean_speed)
        + _trend_penalty_v7(metrics, prev_metrics)
    )

    phase = _curriculum_phase(generation, crash_rate)
    ceiling = _curriculum_ceiling_v7(crash_rate, phase)
    fitness = max(0.0, min(1.0, base - penalty)) * ceiling

    if crash_rate <= 0.05 and mean_speed < 22.0 and mean_overtakes < 0.3:
        fitness = min(fitness, _V7_PASSIVE_CAP)

    return round(float(fitness), 4)


def compute_fitness(
    metrics: dict[str, Any],
    *,
    generation: int = 0,
    prev_metrics: dict[str, Any] | None = None,
    version: int | None = None,
) -> float:
    """
    Dispatch to the configured fitness version (default: v7).

    Optional ``prev_metrics`` enables the slow-to-survive trend penalty in v7.
  """
    ver = FITNESS_VERSION_DEFAULT if version is None else version
    if ver == 6:
        return compute_fitness_v6(metrics)
    if ver == 7:
        return compute_fitness_v7(metrics, generation=generation, prev_metrics=prev_metrics)
    raise ValueError(f"Unsupported fitness version: {ver}")


# ── Structured failure mode detection (improvement #4) ───────────────────────

# Known failure-mode tag names (used for retrieval in improvement #3)
FAILURE_MODE_TAGS = frozenset({
    "tailgating",
    "passive_driving",
    "oscillatory_lane_changes",
    "acceleration_spam",
    "stationary_farming",
    "reward_hacking",
})

STRENGTH_MODE_TAGS = frozenset({
    "high_speed",
    "good_overtaking",
    "safe_driving",
    "smooth_driving",
})


def _filter_known_tags(tags: Any, allowed: frozenset[str]) -> list[str]:
    if not isinstance(tags, list):
        return []
    return [t for t in tags if isinstance(t, str) and t in allowed]


def _extract_critique_meta_json(critique_text: str) -> dict[str, Any] | None:
    """Parse CRITIQUE_META:{...} using balanced-brace extraction."""
    marker = "CRITIQUE_META:"
    if marker not in critique_text:
        return None
    raw = critique_text[critique_text.index(marker) + len(marker) :].strip()
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(raw)):
        ch = raw[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(raw[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def parse_structured_critique(critique_text: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Improvement #4: produce machine-readable critique metadata from the free-text
    critique + metrics.  This is called inside update_critique() and also used
    by the LLM-critique JSON path (reward_designer.py).

    Heuristic rules fire on metrics so we always have SOME structured data
    even when the LLM critique is unavailable.  The LLM JSON path overwrites
    these if a valid JSON block is present in critique_text.

    Returns a dict:
      {
        "failure_modes": list[str],
        "strengths":     list[str],
        "summary":       str,
      }
    """
    failure_modes: list[str] = []
    strengths:     list[str] = []

    mean_speed     = float(metrics.get("mean_speed", 0.0))
    crash_rate     = float(metrics.get("crash_rate", 0.5))
    mean_ttc       = float(metrics.get("mean_ttc", 30.0))
    min_ttc        = float(metrics.get("min_ttc", -1.0))
    mean_overtakes = float(metrics.get("mean_overtakes", 0.0))
    mean_jerk      = float(metrics.get("mean_long_jerk", 0.0))
    mean_accel     = float(metrics.get("mean_accel", 0.0))
    lc             = int(metrics.get("total_lane_changes", 0))
    ot             = float(metrics.get("mean_overtakes", 0.0))

    # Heuristic failure mode tagging
    if mean_speed < 5.0:
        failure_modes.append("stationary_farming")
    if is_passive_driving(metrics):
        if "stationary_farming" not in failure_modes:
            failure_modes.append("passive_driving")
    effective_ttc = min_ttc if min_ttc >= 0 else mean_ttc
    if effective_ttc < 2.0 and crash_rate < 0.2:
        failure_modes.append("tailgating")
    n_eps = max(int(metrics.get("n_episodes", 1)), 1)
    if lc > 0 and ot > 0 and lc / max(ot * n_eps, 1) > 5:
        failure_modes.append("oscillatory_lane_changes")
    if mean_jerk > 2.5 or mean_accel > 3.0:
        failure_modes.append("acceleration_spam")

    # Heuristic strength tagging
    if mean_speed >= 26.0:
        strengths.append("high_speed")
    if mean_overtakes >= 3.0:
        strengths.append("good_overtaking")
    if crash_rate <= 0.05:
        strengths.append("safe_driving")
    if mean_jerk <= 0.5:
        strengths.append("smooth_driving")

    # Try to extract JSON block if the LLM embedded one in the critique text
    # (reward_designer.py can inject a JSON block at the end of the critique)
    meta: dict[str, Any] = {
        "failure_modes": failure_modes,
        "strengths": strengths,
        "summary": "",
    }
    if critique_text:
        parsed = _extract_critique_meta_json(critique_text)
        if parsed:
            llm_failures = _filter_known_tags(parsed.get("failure_modes"), FAILURE_MODE_TAGS)
            if llm_failures:
                meta["failure_modes"] = llm_failures
            llm_strengths = _filter_known_tags(parsed.get("strengths"), STRENGTH_MODE_TAGS)
            if llm_strengths:
                meta["strengths"] = llm_strengths
            if isinstance(parsed.get("summary"), str):
                meta["summary"] = parsed["summary"]

    if not meta["summary"]:
        if failure_modes:
            meta["summary"] = f"Detected issues: {', '.join(failure_modes)}."
        elif strengths:
            meta["summary"] = f"Good performance: {', '.join(strengths)}."
        else:
            meta["summary"] = "No major issues detected."

    return meta


# ── Archive class ─────────────────────────────────────────────────────────────


class RewardArchive:
    """
    Persistent store for reward programs, metrics, fitness, and critiques.
    All writes are atomic (write-to-tmp then rename).

    Improvements #3 & #5: enriched retrieval API for archive-guided hill climbing.
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
            # Back-fill critique_meta for legacy entries that lack it
            for e in self.entries:
                if "critique_meta" not in e:
                    e["critique_meta"] = parse_structured_critique(
                        e.get("critique", ""), e.get("metrics", {})
                    )
            print(f"[archive] Loaded {len(self.entries)} entries from '{self.path}'")
        except Exception as ex:
            print(f"[archive] Failed to load '{self.path}': {ex} — starting fresh")
            self.entries = []

    def save(self) -> None:
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
        metrics: dict[str, Any],
        critique: str = "",
    ) -> dict[str, Any]:
        prev_metrics = self.entries[-1]["metrics"] if self.entries else None
        generation = len(self.entries)
        fitness = compute_fitness(
            metrics,
            generation=generation,
            prev_metrics=prev_metrics,
        )
        critique_meta = parse_structured_critique(critique, metrics)
        entry: dict[str, Any] = {
            "generation": generation,
            "reward_code": reward_code,
            "metrics": dict(metrics),
            "fitness": fitness,
            "fitness_version": FITNESS_VERSION_DEFAULT,
            "critique": critique,
            "critique_meta": critique_meta,          # improvement #4
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
        for entry in self.entries:
            if entry["generation"] == generation:
                entry["critique"] = critique
                # Improvement #4: re-parse structured metadata when critique updates
                entry["critique_meta"] = parse_structured_critique(
                    critique, entry.get("metrics", {})
                )
                self.save()
                return
        print(f"[archive] Warning: generation {generation} not found for critique update")

    def remove_generation(self, generation: int) -> bool:
        """Remove a corrupt/invalid archive entry (e.g. failed restore re-validation)."""
        before = len(self.entries)
        self.entries = [e for e in self.entries if e["generation"] != generation]
        if len(self.entries) < before:
            for i, entry in enumerate(self.entries):
                entry["generation"] = i
            self.save()
            print(f"[archive] Removed generation {generation} from archive")
            return True
        return False

    # ── Core retrieval ────────────────────────────────────────────────────────

    def get_top_k(self, k: int = 3) -> list[dict[str, Any]]:
        """Returns the k entries with highest fitness score."""
        return sorted(self.entries, key=lambda e: e["fitness"], reverse=True)[:k]

    def get_latest(self) -> dict[str, Any] | None:
        return self.entries[-1] if self.entries else None

    def get_by_generation(self, gen: int) -> dict[str, Any] | None:
        for entry in self.entries:
            if entry["generation"] == gen:
                return entry
        return None

    # ── Improvement #3: Richer retrieval API ─────────────────────────────────

    def get_top_rewards(self, k: int = 3) -> list[dict[str, Any]]:
        """Top-k by fitness (alias for get_top_k, exposed for hill-climbing)."""
        return self.get_top_k(k)

    def get_recent_rewards(self, k: int = 3) -> list[dict[str, Any]]:
        """Most recently archived k entries (newest first)."""
        return list(reversed(self.entries))[:k]

    def get_failed_rewards(self, k: int = 3, max_fitness: float = 0.15) -> list[dict[str, Any]]:
        """
        Entries with fitness below max_fitness — useful as negative examples
        so the LLM knows what NOT to replicate.

        Also includes "passive but safe" entries (low crash, low speed/overtakes)
        even when fitness is above max_fitness, so the LLM does not copy
        slow-to-survive strategies from the top-k list.
        """
        failed = [e for e in self.entries if e["fitness"] <= max_fitness]
        passive = [
            e for e in self.entries
            if e not in failed and is_passive_driving(e["metrics"])
        ]
        combined = sorted(failed, key=lambda e: e["fitness"]) + sorted(
            passive, key=lambda e: e["fitness"]
        )
        return combined[:k]

    def get_similar_failure_rewards(
        self, failure_mode: str, k: int = 3
    ) -> list[dict[str, Any]]:
        """
        Entries tagged with a specific failure mode in their critique_meta.

        failure_mode: one of the FAILURE_MODE_TAGS strings, e.g. "tailgating"
        """
        matched = [
            e for e in self.entries
            if failure_mode in e.get("critique_meta", {}).get("failure_modes", [])
        ]
        return sorted(matched, key=lambda e: e["fitness"], reverse=True)[:k]

    def get_entries_by_failure_modes(
        self, modes: list[str], k: int = 3
    ) -> list[dict[str, Any]]:
        """Entries that share ANY of the listed failure mode tags."""
        matched = [
            e for e in self.entries
            if any(m in e.get("critique_meta", {}).get("failure_modes", []) for m in modes)
        ]
        # Sort: highest fitness first so the LLM sees "least bad" examples
        return sorted(matched, key=lambda e: e["fitness"], reverse=True)[:k]

    # ── Improvement #5: Archive-guided hill-climbing context ─────────────────

    def format_for_llm(
        self,
        k: int = 3,
        current_failure_modes: list[str] | None = None,
    ) -> str:
        """
        Improvement #5: richly formatted context for archive-guided hill climbing.

        Sections:
          A) Top-k by fitness
          B) Most recent reward (trend context)
          C) Up to 2 known-failed rewards (negative examples)
          D) Rewards sharing current failure modes (targeted repair context)

        Parameters
        ──────────
        k                     : top-k entries to include in section A
        current_failure_modes : failure modes detected in the LATEST generation,
                                used to surface targeted repair examples (section D)
        """
        if not self.entries:
            return "No previous reward programs in archive."

        lines: list[str] = []

        # ── A) Top performers ────────────────────────────────────────────────
        top = self.get_top_rewards(k)
        lines.append("=== A) TOP REWARD PROGRAMS (by fitness) ===\n")
        for entry in top:
            lines.append(_format_entry(entry, show_code=True))

        # ── B) Most recent ───────────────────────────────────────────────────
        recent = self.get_recent_rewards(1)
        if recent and recent[0]["generation"] not in {e["generation"] for e in top}:
            lines.append("=== B) MOST RECENT REWARD (trend context) ===\n")
            lines.append(_format_entry(recent[0], show_code=True))

        # ── C) Failed rewards (negative examples) ────────────────────────────
        failed = self.get_failed_rewards(k=2)
        if failed:
            lines.append(
                "=== C) FAILED / PASSIVE REWARDS (do NOT repeat these patterns) ===\n"
                "(Includes low-fitness entries AND safe-but-slow passive-driving traps)\n"
            )
            for entry in failed:
                lines.append(_format_entry(entry, show_code=False))

        # ── D) Similar failure mode examples ─────────────────────────────────
        if current_failure_modes:
            similar = self.get_entries_by_failure_modes(current_failure_modes, k=2)
            # Exclude entries already shown in A/B/C
            shown_gens = {e["generation"] for e in top + failed + recent}
            similar = [e for e in similar if e["generation"] not in shown_gens]
            if similar:
                lines.append(
                    f"=== D) REWARDS WITH SIMILAR FAILURE MODES "
                    f"({', '.join(current_failure_modes)}) ===\n"
                )
                lines.append("These previously showed the same issues — study why they failed:\n")
                for entry in similar:
                    lines.append(_format_entry(entry, show_code=True))

        return "\n".join(lines)

    def format_latest_for_critique(self) -> str | None:
        entry = self.get_latest()
        if entry is None:
            return None
        m = entry["metrics"]
        cr = m.get("crash_rate", 0.5)
        ttc_score_val = _ttc_score(
            m.get("mean_ttc", 30.0),
            m.get("p10_ttc", -1.0),
            m.get("min_ttc", -1.0),
        )
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
            f"  p10_ttc         : {m.get('p10_ttc',        -1):.2f} s\n"
            f"  min_ttc         : {m.get('min_ttc',        -1):.2f} s\n"
            f"  mean_long_jerk  : {m.get('mean_long_jerk',  0):.3f} m/s³\n"
            f"  mean_lat_jerk   : {m.get('mean_lat_jerk',   0):.3f} m/s³\n"
            f"  mean_accel      : {m.get('mean_accel',      0):.3f} m/s²\n"
            f"  fitness         : {entry['fitness']:.4f}\n"
            f"\nFitness breakdown:\n"
            f"  speed_score     : {_speed_score(m.get('mean_speed', 0)):.3f}  "
            f"(sigmoid, mid={_SPEED_SIGMOID_MID} m/s)\n"
            f"  overtake_score  : {_overtake_score(m.get('mean_overtakes', 0)):.3f}\n"
            f"  comfort_score   : {_comfort_score(m.get('mean_long_jerk', 0)):.3f}\n"
            f"  ttc_score       : {ttc_score_val:.3f}  "
            f"(mean={_ttc_component_norm(m.get('mean_ttc',30)):.3f}, "
            f"p10={_ttc_component_norm(m.get('p10_ttc', 30)):.3f}, "
            f"min={_ttc_component_norm(m.get('min_ttc', 30)):.3f})\n"
            f"  safety_gate     : {_safety_gate(cr):.3f}  "
            f"({'HARD gate active' if cr > _CRASH_HARD_LIMIT else 'soft gate' if cr > _CRASH_THRESHOLD else 'no penalty'})\n"
            f"  passive_gate    : {_passive_driving_gate(m.get('mean_speed', 0), m.get('mean_overtakes', 0), cr):.3f}  "
            f"({'PASSIVE driving' if is_passive_driving(m) else 'active enough'})\n"
            f"\nStructured critique metadata:\n"
            f"  failure_modes   : {entry.get('critique_meta', {}).get('failure_modes', [])}\n"
            f"  strengths       : {entry.get('critique_meta', {}).get('strengths', [])}\n"
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


# ── Private formatting helper ─────────────────────────────────────────────────


def _format_entry(entry: dict[str, Any], show_code: bool) -> str:
    """Formats a single archive entry for LLM context."""
    m = entry["metrics"]
    cr = m.get("crash_rate", 0.5)
    meta = entry.get("critique_meta", {})
    lines = [
        f"--- Generation {entry['generation']} "
        f"(fitness={entry['fitness']:.4f}) ---\n"
        f"Metrics:\n"
        f"  mean_speed     : {m.get('mean_speed',      0):.2f} m/s\n"
        f"  crash_rate     : {m.get('crash_rate',      0):.1%}\n"
        f"  mean_overtakes : {m.get('mean_overtakes',  0):.2f}/ep\n"
        f"  mean_ttc       : {m.get('mean_ttc',        0):.2f} s  "
        f"p10={m.get('p10_ttc',-1):.1f}s  min={m.get('min_ttc',-1):.1f}s\n"
        f"  mean_long_jerk : {m.get('mean_long_jerk',  0):.3f} m/s³\n"
        f"  completion_rate: {m.get('completion_rate', 0):.1%}\n"
        f"  mean_steps     : {m.get('mean_steps',      0):.0f}\n"
        f"Fitness breakdown:\n"
        f"  speed_score    : {_speed_score(m.get('mean_speed', 0)):.3f}\n"
        f"  overtake_score : {_overtake_score(m.get('mean_overtakes', 0)):.3f}\n"
        f"  comfort_score  : {_comfort_score(m.get('mean_long_jerk', 0)):.3f}\n"
        f"  ttc_score      : {_ttc_score(m.get('mean_ttc',30), m.get('p10_ttc',-1), m.get('min_ttc',-1)):.3f}\n"
        f"  safety_gate    : {_safety_gate(cr):.3f}\n"
        f"  passive_gate   : {_passive_driving_gate(m.get('mean_speed', 0), m.get('mean_overtakes', 0), cr):.3f}"
        f"{'  [PASSIVE — do not copy]' if is_passive_driving(m) else ''}\n"
        f"Failure modes: {meta.get('failure_modes', [])}\n"
        f"Strengths    : {meta.get('strengths', [])}\n"
    ]
    if meta.get("summary"):
        lines.append(f"Summary      : {meta['summary']}\n")
    if entry.get("critique"):
        # Show only first 300 chars of free-text critique to keep context tight
        crit_snippet = entry["critique"][:300].replace("\n", " ")
        lines.append(f"Critique     : {crit_snippet}...\n")
    if show_code:
        lines.append(f"Reward Code:\n```python\n{entry['reward_code']}\n```\n")
    return "\n".join(lines)


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Fitness Function Self-Test (v4: sigmoid speed + robust TTC) ===\n")

    scenarios = [
        (
            "Perfect agent",
            {"mean_speed": 30.0, "crash_rate": 0.00, "mean_overtakes": 10.0,
             "mean_long_jerk": 0.5, "mean_ttc": 8.0, "p10_ttc": 6.0, "min_ttc": 4.0,
             "completion_rate": 1.00},
        ),
        (
            "Good agent",
            {"mean_speed": 25.0, "crash_rate": 0.05, "mean_overtakes": 5.0,
             "mean_long_jerk": 1.0, "mean_ttc": 6.0, "p10_ttc": 4.5, "min_ttc": 3.0,
             "completion_rate": 0.95},
        ),
        (
            "Tailgating (fast, TTC=1.4 mean, min=0.8)",
            {"mean_speed": 29.0, "crash_rate": 0.00, "mean_overtakes": 0.0,
             "mean_long_jerk": 0.15, "mean_ttc": 1.4, "p10_ttc": 0.9, "min_ttc": 0.8,
             "completion_rate": 1.00},
        ),
        (
            "Tailgating (legacy, no p10/min)",
            {"mean_speed": 29.0, "crash_rate": 0.00, "mean_overtakes": 0.0,
             "mean_long_jerk": 0.15, "mean_ttc": 1.4, "completion_rate": 1.00},
        ),
        (
            "Near-miss: 299 safe steps + 1 bad",
            {"mean_speed": 27.0, "crash_rate": 0.00, "mean_overtakes": 3.0,
             "mean_long_jerk": 0.3, "mean_ttc": 28.5, "p10_ttc": 5.0, "min_ttc": 0.2,
             "completion_rate": 1.00},
        ),
        (
            "Stationary/safe",
            {"mean_speed": 5.0, "crash_rate": 0.00, "mean_overtakes": 0.0,
             "mean_long_jerk": 0.1, "mean_ttc": 30.0, "p10_ttc": 30.0, "min_ttc": 30.0,
             "completion_rate": 1.00},
        ),
        (
            "Passive safe (20 m/s, 0 overtakes)",
            {"mean_speed": 20.0, "crash_rate": 0.00, "mean_overtakes": 0.0,
             "mean_long_jerk": 0.6, "mean_ttc": 2.4, "p10_ttc": 1.4, "min_ttc": 0.3,
             "completion_rate": 1.00},
        ),
        (
            "Fast but crashy",
            {"mean_speed": 28.0, "crash_rate": 0.50, "mean_overtakes": 8.0,
             "mean_long_jerk": 1.5, "mean_ttc": 3.5, "p10_ttc": 2.0, "min_ttc": 0.5,
             "completion_rate": 0.50},
        ),
    ]

    print(f"{'Scenario':<42} {'fitness':>8}  {'speed_s':>7}  {'overt_s':>7}  "
          f"{'comf_s':>7}  {'ttc_s':>7}  {'gate':>7}")
    print("─" * 105)

    for name, m in scenarios:
        f = compute_fitness(m)
        ss = _speed_score(m["mean_speed"])
        os_ = _overtake_score(m["mean_overtakes"])
        cs = _comfort_score(m["mean_long_jerk"])
        ts = _ttc_score(m["mean_ttc"], m.get("p10_ttc", -1), m.get("min_ttc", -1))
        g = _safety_gate(m["crash_rate"])
        print(f"{name:<42} {f:>8.4f}  {ss:>7.3f}  {os_:>7.3f}  "
              f"{cs:>7.3f}  {ts:>7.3f}  {g:>7.3f}")

    print("\n✓ All scenarios computed successfully.")

    # Speed score comparison: sigmoid vs old linear
    print("\n=== Speed Score Comparison (sigmoid v4 vs linear v3) ===")
    print(f"{'speed (m/s)':>12}  {'sigmoid v4':>11}  {'linear v3':>11}")
    print("-" * 38)
    for spd in [15, 18, 20, 22, 24, 25, 26, 28, 30, 32]:
        sig = _speed_score(float(spd))
        lin = max(0.0, min(1.0, (spd - 20.0) / 10.0))
        print(f"{spd:>12}  {sig:>11.3f}  {lin:>11.3f}")

    # Failure mode parsing smoke test
    print("\n=== Structured Failure Mode Parsing ===")
    test_metrics = {
        "mean_speed": 12.0, "crash_rate": 0.05, "mean_overtakes": 0.2,
        "mean_long_jerk": 0.3, "mean_ttc": 1.5, "min_ttc": 0.8,
        "total_lane_changes": 50, "n_episodes": 5,
    }
    meta = parse_structured_critique("", test_metrics)
    print(f"failure_modes : {meta['failure_modes']}")
    print(f"strengths     : {meta['strengths']}")
    print(f"summary       : {meta['summary']}")
