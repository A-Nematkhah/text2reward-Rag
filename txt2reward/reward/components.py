"""Legacy weight-based reward components for ``evaluate.py --no-shaped``.

New code should use ``reward_program.py`` and ``txt2reward.sandbox``.
"""

from __future__ import annotations

import json
import math
import os

from txt2reward.config.paths import WEIGHTS_FILE
from txt2reward.core.log import get_logger

log = get_logger("weights")

DEFAULT_WEIGHTS: dict[str, float] = {
    "w_env": 0.2,
    "w_speed": 0.8,
    "speed_target": 28.0,
    "speed_min": 10.0,
    "w_safety": 0.0,
    "safe_dist": 10.0,
    "w_lane": 0.0,
    "w_collision": 20.0,
    "w_ttc": 0.3,
    "ttc_threshold": 3.0,
    "w_rel_vel": 0.0,
    "w_comfort": 0.02,
    "jerk_threshold": 2.0,
    "w_jerk": 0.02,
    "w_accel": 0.02,
    "accel_threshold": 3.0,
    "w_density": 0.0,
    "density_radius": 30.0,
    "density_max": 5.0,
    "w_overtake": 2.0,
    "w_lc_quality": 0.05,
    "w_progress": 0.5,
    "max_speed": 40.0,
}

WEIGHT_BOUNDS: dict[str, tuple[float, float]] = {k: (v * 0.1, v * 5.0 + 1.0) for k, v in DEFAULT_WEIGHTS.items()}


def load_weights(path: str = WEIGHTS_FILE) -> dict[str, float]:
    """Loads weights from JSON; returns defaults if missing."""
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            w = dict(DEFAULT_WEIGHTS)
            w.update({k: float(v) for k, v in data.items() if k in w})
            return w
        except Exception as exc:
            log.warning("Failed to load '%s': %s — using defaults", path, exc)
    return dict(DEFAULT_WEIGHTS)


def save_weights(weights: dict[str, float], path: str = WEIGHTS_FILE) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)
    os.replace(tmp, path)


def clamp_weights(weights: dict[str, float]) -> dict[str, float]:
    out = {}
    for k, v in weights.items():
        lo, hi = WEIGHT_BOUNDS.get(k, (-1e9, 1e9))
        out[k] = float(max(lo, min(hi, v)))
    return out


def _speed_component(speed_ms: float, target: float, speed_min: float) -> float:
    span = max(target - speed_min, 1e-6)
    return float(max(0.0, min(1.0, (speed_ms - speed_min) / span)))


def _safety_component(front_dist: float, safe_dist: float) -> float:
    return -1.0 if front_dist < safe_dist else 0.0


def _lane_component(lane: int, num_lanes: int = 4) -> float:
    middle_lanes = set(range(1, num_lanes - 1))
    return 1.0 if lane in middle_lanes else -0.5


def _ttc_component(ttc: float, ttc_threshold: float) -> float:
    if ttc <= 0.0:
        return -1.0
    if ttc >= ttc_threshold:
        return 0.0
    return -(1.0 - ttc / ttc_threshold)


def _rel_vel_component(rel_vel_ms: float) -> float:
    return float(max(-1.0, min(1.0, rel_vel_ms / 20.0)))


def _comfort_component(long_jerk: float, lat_jerk: float, jerk_threshold: float) -> float:
    combined = math.sqrt(long_jerk**2 + lat_jerk**2)
    if combined <= jerk_threshold:
        return 0.0
    excess = (combined - jerk_threshold) / max(jerk_threshold, 1e-6)
    return -float(min(1.0, excess))


def _jerk_component(long_jerk: float, jerk_threshold: float) -> float:
    abs_jerk = abs(long_jerk)
    if abs_jerk <= jerk_threshold:
        return 0.0
    excess = (abs_jerk - jerk_threshold) / max(jerk_threshold, 1e-6)
    return -float(min(1.0, excess))


def _accel_component(accel_ms2: float, accel_threshold: float) -> float:
    abs_accel = abs(accel_ms2)
    if abs_accel <= accel_threshold:
        return 0.0
    excess = (abs_accel - accel_threshold) / max(accel_threshold, 1e-6)
    return -float(min(1.0, excess))


def _density_component(nearby_vehicles: int, density_max: float) -> float:
    return -float(min(1.0, nearby_vehicles / max(density_max, 1.0)))


def _overtake_component(overtook: bool) -> float:
    return 1.0 if overtook else 0.0


def _progress_component(speed_ms: float, max_speed: float) -> float:
    return float(max(0.0, min(1.0, speed_ms / max(max_speed, 1e-6))))


def _lc_quality_component(lane_changed: bool, rel_vel_ms: float, front_dist: float, safe_dist: float) -> float:
    if not lane_changed:
        return 0.0
    justified = (front_dist < safe_dist * 1.5) or (rel_vel_ms < 0.0)
    return 0.0 if justified else -0.5


def compute_shaped_reward(
    weights: dict[str, float],
    env_reward: float,
    speed_ms: float,
    lane: int,
    front_dist: float,
    collided: bool,
    rel_vel_ms: float = 0.0,
    ttc: float = 30.0,
    long_jerk: float = 0.0,
    lat_jerk: float = 0.0,
    accel_ms2: float = 0.0,
    nearby_vehicles: int = 0,
    overtook: bool = False,
    lane_changed: bool = False,
    num_lanes: int = 4,
) -> float:
    """LEGACY weight-based shaped reward used only by evaluate.py --no-shaped."""
    w = weights
    r = w["w_env"] * env_reward
    r += w["w_speed"] * _speed_component(speed_ms, w["speed_target"], w["speed_min"])
    r += w["w_safety"] * _safety_component(front_dist, w["safe_dist"])
    r += w["w_lane"] * _lane_component(lane, num_lanes)
    if collided:
        r -= w["w_collision"]
    r += w["w_ttc"] * _ttc_component(ttc, w["ttc_threshold"])
    r += w["w_rel_vel"] * _rel_vel_component(rel_vel_ms)
    r += w["w_comfort"] * _comfort_component(long_jerk, lat_jerk, w["jerk_threshold"])
    r += w["w_jerk"] * _jerk_component(long_jerk, w["jerk_threshold"])
    r += w["w_accel"] * _accel_component(accel_ms2, w["accel_threshold"])
    r += w["w_density"] * _density_component(nearby_vehicles, w["density_max"])
    r += w["w_overtake"] * _overtake_component(overtook)
    r += w["w_lc_quality"] * _lc_quality_component(lane_changed, rel_vel_ms, front_dist, w["safe_dist"])
    r += w["w_progress"] * _progress_component(speed_ms, w["max_speed"])
    return float(r)
