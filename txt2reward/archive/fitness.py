"""Multi-version fitness scoring for reward archive ranking.

Public API: ``compute_fitness()`` (default v8), ``is_passive_driving()``, and
version-specific scorers v6/v7/v8 for ablation. Version constant lives in
``config.fitness.FITNESS_VERSION_DEFAULT``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from txt2reward.archive.curriculum import infer_curriculum_phase
from txt2reward.config.fitness import FITNESS_VERSION_DEFAULT, PASSIVE_DRIVING_CRASH_CEILING
from txt2reward.core.metrics import lane_change_rate, near_miss_rate, safe_overtake_ratio
from txt2reward.core.types import CurriculumPhase, FitnessMetrics

# ── Fitness weights ────────────────────────────────────────────────────────────
_W = {
    "w_speed": 0.25,
    "w_overtake": 0.30,  # raised: overtaking is the clearest active-driving signal
    "w_comfort": 0.10,
    "w_ttc": 0.20,  # lowered to compensate; sum stays 1.0
    "w_complete": 0.15,
}
assert abs(sum(_W.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# ── Component normalisation references ────────────────────────────────────────
_SPEED_MIN = 20.0  # kept for backward-compat; sigmoid replaces linear use
_SPEED_REF = 30.0  # kept for backward-compat
_SPEED_SIGMOID_K = 0.5  # logistic steepness (#1)
_SPEED_SIGMOID_MID = 25.0  # midpoint of logistic (#1)

_OVERTAKE_REF = 5.0  # 3 overtakes/ep now gives 0.60 score, not 0.30
_COMFORT_K = 0.5
_TTC_SAFE = 5.0  # s — normalisation ceiling for all TTC components

# Robust TTC sub-weights (sum to 1.0)
_TTC_W_MEAN = 0.40
_TTC_W_P10 = 0.35
_TTC_W_MIN = 0.25
assert abs(_TTC_W_MEAN + _TTC_W_P10 + _TTC_W_MIN - 1.0) < 1e-9

# ── Safety gate parameters ─────────────────────────────────────────────────────
_CRASH_THRESHOLD = 0.30
_CRASH_K_SOFT = 5.0
_CRASH_HARD_LIMIT = 0.80
_HARD_PENALTY_SCALE = 0.10

# ── Passive-driving gate (v6) ─────────────────────────────────────────────────
# When the agent is already safe (low crash_rate), suppress fitness if it trades
# speed/overtaking for survival — the "slow down to stay safe" reward hack.
_PASSIVE_SPEED_MIN = 24.0  # m/s — raised: 20-22 m/s is passive, need 24+
_PASSIVE_OVERTAKE_MIN = 1.5  # overtakes/ep — raised: 0.5 was trivially easy
_PASSIVE_GATE_FLOOR = 0.10  # lowered: passive agent gets ≤10% of base fitness
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
    s_p10 = _ttc_component_norm(p10_ttc)
    s_min = _ttc_component_norm(min_ttc)
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


def is_passive_driving(metrics: Mapping[str, Any]) -> bool:
    """
    True when the agent is crash-free enough that we should expect active
    driving, but mean speed and/or overtakes are too low.

    Uses v7 thresholds when default fitness version is 7.
    """
    crash_rate = float(metrics.get("crash_rate", 1.0))
    if crash_rate > PASSIVE_DRIVING_CRASH_CEILING:
        return False
    speed_min = _V7_PASSIVE_SPEED_MIN if FITNESS_VERSION_DEFAULT >= 7 else _PASSIVE_SPEED_MIN
    ot_min = _V7_PASSIVE_OT_MIN if FITNESS_VERSION_DEFAULT >= 7 else _PASSIVE_OVERTAKE_MIN
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

    Inactive (returns 1.0) while crash_rate > PASSIVE_DRIVING_CRASH_CEILING — don't
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
    if float(crash_rate) > PASSIVE_DRIVING_CRASH_CEILING:
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

# ── Fitness v8 parameters ─────────────────────────────────────────────────────
# v8 removes the v7 hard flatline (fitness=0.01 when crash>50%) and replaces it
# with a continuous survival score so the archive retains gradient signal even
# when every episode crashes.  Lower crash_rate always yields higher survival;
# within the same crash band, speed/overtaking/comfort still differentiate.
_V8_CRASH_FLOOR = 0.02
_V8_CRASH_SURVIVAL_EXP = 1.35
_V8_DANGER_THRESHOLD = 0.50
_V8_DANGER_EXTRA_EXP = 2.5
_V8_W = {
    "activity": 0.30,
    "speed": 0.15,
    "overtake": 0.15,
    "lane_eff": 0.10,
    "ttc": 0.15,
    "comfort": 0.15,
}
assert abs(sum(_V8_W.values()) - 1.0) < 1e-9
_V8_JERK_SPAM_JERK_REF = 2.5
_V8_JERK_SPAM_ACCEL_REF = 3.0
_V8_JERK_SPAM_LAMBDA = 0.14
_V8_LANE_OSC_LAMBDA = 0.12
_V8_NEAR_MISS_LAMBDA = 0.12
_V8_NEAR_MISS_RATE_LAMBDA = 0.10
_V8_NEAR_MISS_RATE_THRESHOLD = 2.0
_V8_TIEBREAK_SCALE = 0.015
_V8_ELITE_FLOOR = 0.90
_V8_ELITE_SPAN = 0.10


def _speed_score_v7(mean_speed: float) -> float:
    x = _V7_SPEED_K * (float(mean_speed) - _V7_SPEED_MID)
    return float(1.0 / (1.0 + math.exp(-x)))


def _overtake_score_v7(mean_overtakes: float) -> float:
    ratio = float(mean_overtakes) / max(_V7_OVERTAKE_REF, 1e-6)
    return float(min(1.0, max(0.0, ratio) ** _V7_OVERTAKE_ALPHA))


def _activity_score_v7(mean_speed: float, mean_overtakes: float) -> float:
    s_v = _speed_score_v7(mean_speed)
    s_o = _overtake_score_v7(mean_overtakes)
    return float((s_v**_V7_ACTIVITY_SPEED_EXP) * (s_o**_V7_ACTIVITY_OT_EXP))


def _activity_product_v7(mean_speed: float, mean_overtakes: float) -> float:
    speed_factor = min(1.0, (float(mean_speed) / _V7_PASSIVE_SPEED_MIN) ** 2)
    overtake_factor = min(1.0, (float(mean_overtakes) / _V7_PASSIVE_OT_MIN) ** 0.5)
    return float(speed_factor * overtake_factor)


def _lane_efficiency_score(metrics: Mapping[str, Any]) -> float:
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
    if float(crash_rate) > PASSIVE_DRIVING_CRASH_CEILING:
        return 0.0
    activity = _activity_product_v7(mean_speed, mean_overtakes)
    return float(_V7_PASSIVE_LAMBDA * max(0.0, _V7_PASSIVE_TARGET - activity))


def _trend_penalty_v7(metrics: Mapping[str, Any], prev_metrics: Mapping[str, Any] | None) -> float:
    if not prev_metrics:
        return 0.0
    delta_speed = float(metrics.get("mean_speed", 0.0)) - float(prev_metrics.get("mean_speed", 0.0))
    delta_crash = float(metrics.get("crash_rate", 0.0)) - float(prev_metrics.get("crash_rate", 0.0))
    if delta_speed >= 0.0 or delta_crash >= 0.0:
        return 0.0
    return float(_V7_TREND_LAMBDA * (-delta_speed))


def _generation_curriculum_tier_v7(generation: int, crash_rate: float) -> int:
    """Legacy v7 tier index from generation + crash_rate (not metrics-driven phase)."""
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


def _survival_score_v8(crash_rate: float) -> float:
    """
    Continuous survival multiplier in [floor, 1].

    Below _V8_DANGER_THRESHOLD: identical to the original single power-curve
    (monotonic, smooth gradient — unchanged behavior).

    Above _V8_DANGER_THRESHOLD: an additional multiplicative penalty kicks in,
    so a 90-100% crash-rate agent falls off much faster than the base curve
    alone would produce, preventing reckless-but-fast agents from scoring
    close enough to genuinely safe agents to be mistaken for "top performers"
    in archive retrieval.
    """
    cr = min(1.0, max(0.0, float(crash_rate)))
    base = _V8_CRASH_FLOOR + (1.0 - _V8_CRASH_FLOOR) * ((1.0 - cr) ** _V8_CRASH_SURVIVAL_EXP)
    if cr > _V8_DANGER_THRESHOLD:
        danger_frac = (cr - _V8_DANGER_THRESHOLD) / (1.0 - _V8_DANGER_THRESHOLD)
        base *= (1.0 - danger_frac) ** _V8_DANGER_EXTRA_EXP
    return float(max(_V8_CRASH_FLOOR * 0.1, base))


def _jerk_spam_penalty_v8(
    mean_long_jerk: float,
    mean_accel: float,
    mean_speed: float,
) -> float:
    """High jerk/accel without proportional speed — acceleration spam."""
    jerk_excess = max(0.0, float(mean_long_jerk) - _V8_JERK_SPAM_JERK_REF) / 4.0
    accel_excess = max(0.0, float(mean_accel) - _V8_JERK_SPAM_ACCEL_REF) / 4.0
    if jerk_excess <= 0.0 and accel_excess <= 0.0:
        return 0.0
    speed_cover = min(1.0, float(mean_speed) / 26.0)
    spam = max(jerk_excess, accel_excess)
    return float(_V8_JERK_SPAM_LAMBDA * spam * (1.0 - 0.6 * speed_cover))


def _lane_oscillation_penalty_v8(metrics: Mapping[str, Any]) -> float:
    """Many lane changes with few overtakes — lane thrashing."""
    ratio = safe_overtake_ratio(metrics)
    if ratio >= 0.35:
        return 0.0
    lc_rate = lane_change_rate(metrics)
    if lc_rate < 2.0:
        return 0.0
    thrash = min(1.0, lc_rate / 8.0)
    return float(_V8_LANE_OSC_LAMBDA * thrash * (1.0 - ratio))


def _near_miss_rate_penalty_v8(near_miss: float, crash_rate: float) -> float:
    """Penalise sustained near-miss steps (TTC < 2 s) when not yet crashing."""
    if float(crash_rate) > 0.25:
        return 0.0
    if near_miss <= 0.05:
        return 0.0
    excess = min(1.0, (near_miss - 0.05) / 0.35)
    return float(_V8_NEAR_MISS_RATE_LAMBDA * excess)


def _tailgate_penalty_v8(min_ttc: float, p10_ttc: float, crash_rate: float) -> float:
    """Sustained low TTC without crashing yet — tailgating."""
    if float(crash_rate) > 0.25:
        return 0.0
    if min_ttc < 0 or p10_ttc < 0:
        return 0.0
    effective = min(float(min_ttc), float(p10_ttc))
    if effective >= 2.0:
        return 0.0
    return float(_V8_NEAR_MISS_LAMBDA * max(0.0, 1.0 - effective / 2.0))


def _curriculum_quality_weights(phase: str) -> dict[str, float]:
    """Re-weight quality components by observed curriculum phase."""
    base = dict(_V8_W)
    if phase == "survive":
        base["ttc"] += 0.10
        base["comfort"] += 0.05
        base["speed"] -= 0.08
        base["overtake"] -= 0.07
    elif phase == "speed":
        base["speed"] += 0.10
        base["activity"] += 0.05
        base["ttc"] -= 0.08
        base["lane_eff"] -= 0.07
    elif phase == "overtake":
        base["overtake"] += 0.12
        base["activity"] += 0.08
        base["speed"] -= 0.10
        base["comfort"] -= 0.10
    total = sum(base.values())
    return {k: v / total for k, v in base.items()}


_SAMPLE_CONFIDENCE_MIN_RELIABLE = 30
_SAMPLE_CONFIDENCE_MAX_PENALTY = 0.05


def _sample_confidence_penalty(n_episodes: int, min_reliable: int = _SAMPLE_CONFIDENCE_MIN_RELIABLE) -> float:
    """
    Small additive penalty (0 to _SAMPLE_CONFIDENCE_MAX_PENALTY) applied when
    n_episodes is below min_reliable, so noisy small-sample fitness scores are
    shrunk slightly rather than treated as equally authoritative as
    well-sampled ones. Zero penalty once n_episodes reaches min_reliable.
    """
    n = max(int(n_episodes), 0)
    if n >= min_reliable:
        return 0.0
    return float(_SAMPLE_CONFIDENCE_MAX_PENALTY * (1.0 - n / min_reliable))


def compute_fitness_v8(
    metrics: FitnessMetrics | Mapping[str, Any],
    *,
    generation: int = 0,
    prev_metrics: FitnessMetrics | Mapping[str, Any] | None = None,
    curriculum_phase: CurriculumPhase | str | None = None,
) -> float:
    """
    Fitness v8 — continuous survival ranking + metrics-driven curriculum.

    Replaces the v7 hard flatline at crash>50% with ``_survival_score_v8`` so
    100% crash, 90% crash, and 80% crash produce distinct fitness values while
    still ranking below safer behaviour.
    """
    crash_rate = float(metrics.get("crash_rate", 0.5))
    mean_speed = float(metrics.get("mean_speed", 0.0))
    mean_overtakes = float(metrics.get("mean_overtakes", 0.0))
    mean_long_jerk = float(metrics.get("mean_long_jerk", 0.0))
    mean_accel = float(metrics.get("mean_accel", 0.0))
    mean_ttc = float(metrics.get("mean_ttc", 30.0))
    p10_ttc = float(metrics.get("p10_ttc", -1.0))
    min_ttc = float(metrics.get("min_ttc", -1.0))
    near_miss = near_miss_rate(metrics)

    survival = _survival_score_v8(crash_rate)
    phase = curriculum_phase or infer_curriculum_phase(metrics)
    weights = _curriculum_quality_weights(phase)

    s_activity = _activity_score_v7(mean_speed, mean_overtakes)
    s_speed = _speed_score_v7(mean_speed)
    s_overtake = _overtake_score_v7(mean_overtakes)
    s_lane = _lane_efficiency_score(metrics)
    s_ttc = _ttc_score(mean_ttc, p10_ttc, min_ttc)
    s_comfort = _comfort_score(mean_long_jerk)
    s_safe_ot = safe_overtake_ratio(metrics)

    quality = (
        weights["activity"] * s_activity
        + weights["speed"] * s_speed
        + weights["overtake"] * s_overtake
        + weights["lane_eff"] * (0.7 * s_lane + 0.3 * s_safe_ot)
        + weights["ttc"] * s_ttc
        + weights["comfort"] * s_comfort
    )

    behavioral_penalty = (
        _safety_penalty_v7(crash_rate)
        + _tailgate_penalty_v8(min_ttc, p10_ttc, crash_rate)
        + _near_miss_rate_penalty_v8(near_miss, crash_rate)
        + _jerk_spam_penalty_v8(mean_long_jerk, mean_accel, mean_speed)
        + _lane_oscillation_penalty_v8(metrics)
        + _passive_penalty_v7(mean_speed, mean_overtakes, crash_rate)
        + _stationary_penalty_v7(mean_speed)
        + _trend_penalty_v7(metrics, prev_metrics)
        + _sample_confidence_penalty(int(metrics.get("n_episodes", _SAMPLE_CONFIDENCE_MIN_RELIABLE)))
    )

    quality_adj = max(0.0, min(1.0, quality - behavioral_penalty))

    # Survival anchors fitness in high-crash regimes; quality modulates within each
    # crash band so 100% != 90% != 80%, and faster/safer behaviour ranks higher
    # at the same crash rate.
    raw_fitness = survival + survival * quality_adj * 0.75 + _V8_TIEBREAK_SCALE * quality_adj * (1.0 - crash_rate)

    if raw_fitness >= 1.0:
        # Low-crash agents often exceed 1.0 before clamping; map quality_adj into
        # [elite_floor, 1.0] so tailgating / lane thrashing still ranks below
        # clean active driving at the same crash rate.
        fitness = _V8_ELITE_FLOOR + _V8_ELITE_SPAN * max(0.0, min(1.0, quality_adj))
    else:
        fitness = raw_fitness

    if crash_rate <= PASSIVE_DRIVING_CRASH_CEILING and mean_speed < 22.0 and mean_overtakes < 0.3:
        fitness = min(fitness, _V7_PASSIVE_CAP)

    return round(float(max(0.001, min(1.0, fitness))), 4)


# ── Public fitness functions ──────────────────────────────────────────────────


def compute_fitness_v6(metrics: Mapping[str, Any]) -> float:
    """Legacy v6 fitness (multiplicative safety × passive gates)."""
    crash_rate = float(metrics.get("crash_rate", 0.5))
    mean_speed = float(metrics.get("mean_speed", 0.0))
    mean_overtakes = float(metrics.get("mean_overtakes", 0.0))
    mean_long_jerk = float(metrics.get("mean_long_jerk", 0.0))
    mean_ttc = float(metrics.get("mean_ttc", 30.0))
    p10_ttc = float(metrics.get("p10_ttc", -1.0))
    min_ttc = float(metrics.get("min_ttc", -1.0))
    completion_rate = float(metrics.get("completion_rate", 0.5))

    s_speed = _speed_score(mean_speed)
    s_overtake = _overtake_score(mean_overtakes)
    s_comfort = _comfort_score(mean_long_jerk)
    s_ttc = _ttc_score(mean_ttc, p10_ttc, min_ttc)
    s_complete = float(max(0.0, min(1.0, completion_rate)))

    base = (
        _W["w_speed"] * s_speed
        + _W["w_overtake"] * s_overtake
        + _W["w_comfort"] * s_comfort
        + _W["w_ttc"] * s_ttc
        + _W["w_complete"] * s_complete
    )

    safety = _safety_gate(crash_rate)
    passive = _passive_driving_gate(mean_speed, mean_overtakes, crash_rate)
    fitness = float(max(0.0, min(1.0, base * safety * passive)))
    return round(fitness, 4)


def compute_fitness_v7(
    metrics: Mapping[str, Any],
    *,
    generation: int = 0,
    prev_metrics: Mapping[str, Any] | None = None,
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

    phase = _generation_curriculum_tier_v7(generation, crash_rate)
    ceiling = _curriculum_ceiling_v7(crash_rate, phase)
    fitness = max(0.0, min(1.0, base - penalty)) * ceiling

    if crash_rate <= PASSIVE_DRIVING_CRASH_CEILING and mean_speed < 22.0 and mean_overtakes < 0.3:
        fitness = min(fitness, _V7_PASSIVE_CAP)

    return round(float(fitness), 4)


def compute_fitness(
    metrics: FitnessMetrics | Mapping[str, Any],
    *,
    generation: int = 0,
    prev_metrics: FitnessMetrics | Mapping[str, Any] | None = None,
    version: int | None = None,
) -> float:
    """Dispatch to fitness v6, v7, or v8.

    Args:
        metrics: Episode-window or evaluation metrics (enriched when possible).
        generation: Archive generation index (curriculum ceiling in v7/v8).
        prev_metrics: Prior generation metrics for trend penalty (v7/v8).
        version: Override ``FITNESS_VERSION_DEFAULT`` for ablation.

    Returns:
        Scalar fitness in ``[0, 1]`` (rounded to four decimals).
    """
    ver = FITNESS_VERSION_DEFAULT if version is None else version
    if ver == 6:
        return compute_fitness_v6(metrics)
    if ver == 7:
        return compute_fitness_v7(metrics, generation=generation, prev_metrics=prev_metrics)
    if ver == 8:
        return compute_fitness_v8(
            metrics,
            generation=generation,
            prev_metrics=prev_metrics,
        )
    raise ValueError(f"Unsupported fitness version: {ver}")


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from txt2reward.archive.critique import parse_structured_critique
    from txt2reward.core.log import configure_logging, get_logger

    configure_logging()
    log = get_logger("fitness")

    log.info("=== Fitness Function Self-Test (v4: sigmoid speed + robust TTC) ===\n")

    scenarios = [
        (
            "Perfect agent",
            {
                "mean_speed": 30.0,
                "crash_rate": 0.00,
                "mean_overtakes": 10.0,
                "mean_long_jerk": 0.5,
                "mean_ttc": 8.0,
                "p10_ttc": 6.0,
                "min_ttc": 4.0,
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
                "p10_ttc": 4.5,
                "min_ttc": 3.0,
                "completion_rate": 0.95,
            },
        ),
        (
            "Tailgating (fast, TTC=1.4 mean, min=0.8)",
            {
                "mean_speed": 29.0,
                "crash_rate": 0.00,
                "mean_overtakes": 0.0,
                "mean_long_jerk": 0.15,
                "mean_ttc": 1.4,
                "p10_ttc": 0.9,
                "min_ttc": 0.8,
                "completion_rate": 1.00,
            },
        ),
        (
            "Tailgating (legacy, no p10/min)",
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
            "Near-miss: 299 safe steps + 1 bad",
            {
                "mean_speed": 27.0,
                "crash_rate": 0.00,
                "mean_overtakes": 3.0,
                "mean_long_jerk": 0.3,
                "mean_ttc": 28.5,
                "p10_ttc": 5.0,
                "min_ttc": 0.2,
                "completion_rate": 1.00,
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
                "p10_ttc": 30.0,
                "min_ttc": 30.0,
                "completion_rate": 1.00,
            },
        ),
        (
            "Passive safe (20 m/s, 0 overtakes)",
            {
                "mean_speed": 20.0,
                "crash_rate": 0.00,
                "mean_overtakes": 0.0,
                "mean_long_jerk": 0.6,
                "mean_ttc": 2.4,
                "p10_ttc": 1.4,
                "min_ttc": 0.3,
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
                "p10_ttc": 2.0,
                "min_ttc": 0.5,
                "completion_rate": 0.50,
            },
        ),
    ]

    log.info(
        "%-42s %8s  %7s  %7s  %7s  %7s  %7s", "Scenario", "fitness", "speed_s", "overt_s", "comf_s", "ttc_s", "gate"
    )
    log.info("─" * 105)

    for name, m in scenarios:
        f = compute_fitness(m)
        ss = _speed_score(m["mean_speed"])
        os_ = _overtake_score(m["mean_overtakes"])
        cs = _comfort_score(m["mean_long_jerk"])
        ts = _ttc_score(m["mean_ttc"], m.get("p10_ttc", -1), m.get("min_ttc", -1))
        g = _safety_gate(m["crash_rate"])
        log.info(
            "%-42s %8.4f  %7.3f  %7.3f  %7.3f  %7.3f  %7.3f",
            name,
            f,
            ss,
            os_,
            cs,
            ts,
            g,
        )

    log.info("\n✓ All scenarios computed successfully.")

    log.info("\n=== Speed Score Comparison (sigmoid v4 vs linear v3) ===")
    log.info("%12s  %11s  %11s", "speed (m/s)", "sigmoid v4", "linear v3")
    log.info("-" * 38)
    for spd in [15, 18, 20, 22, 24, 25, 26, 28, 30, 32]:
        sig = _speed_score(float(spd))
        lin = max(0.0, min(1.0, (spd - 20.0) / 10.0))
        log.info("%12d  %11.3f  %11.3f", spd, sig, lin)

    log.info("\n=== Structured Failure Mode Parsing ===")
    test_metrics = {
        "mean_speed": 12.0,
        "crash_rate": 0.05,
        "mean_overtakes": 0.2,
        "mean_long_jerk": 0.3,
        "mean_ttc": 1.5,
        "min_ttc": 0.8,
        "total_lane_changes": 50,
        "n_episodes": 5,
    }
    meta = parse_structured_critique("", test_metrics)
    log.info("failure_modes : %s", meta["failure_modes"])
    log.info("strengths     : %s", meta["strengths"])
    log.info("summary       : %s", meta["summary"])
