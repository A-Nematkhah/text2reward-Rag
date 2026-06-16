"""
reward_program.py
─────────────────
Default reward program used before the first LLM generation.

This file is REPLACED at runtime by the LLM-generated version.
It is also the template shown to the LLM as an example.

The function signature is fixed:
  compute_reward(state: dict) -> float

State keys available:
  speed_ms        : float   ego speed in m/s (typical range 0–40)
  front_dist      : float   distance to front vehicle in metres (0–200)
  ttc             : float   time-to-collision in seconds (0–30, 30 = no vehicle)
  rel_vel_ms      : float   v_front - v_ego in m/s (negative = approaching)
  lane            : int     current lane index, 0 = rightmost
  overtook        : bool    ego completed an overtake this step
  lane_changed    : bool    ego changed lane this step
  collided        : bool    collision detected
  nearby_vehicles : int     vehicles within ~30 m
  accel_ms2       : float   longitudinal acceleration m/s²
  long_jerk       : float   longitudinal jerk m/s³
  lat_jerk        : float   lateral jerk m/s³

Safe math available (no imports needed):
  min, max, abs, round, float, int, bool
  sqrt, exp, log, sin, cos, tan, atan, atan2
  floor, ceil, clip(val, lo, hi), pi, e, inf
"""


def compute_reward(state):
    # ── Unpack state ──────────────────────────────────────────────────────────
    speed_ms        = state["speed_ms"]
    front_dist      = state["front_dist"]
    ttc             = state["ttc"]
    rel_vel_ms      = state["rel_vel_ms"]
    collided        = state["collided"]
    overtook        = state["overtook"]
    lane_changed    = state["lane_changed"]
    accel_ms2       = state["accel_ms2"]
    long_jerk       = state["long_jerk"]
    lat_jerk        = state["lat_jerk"]
    nearby_vehicles = state["nearby_vehicles"]

    reward = 0.0

    # ── Speed reward: linear ramp 0→1 between 10 and 28 m/s ──────────────────
    speed_norm = clip((speed_ms - 10.0) / 18.0, 0.0, 1.0)
    reward = reward + 0.8 * speed_norm

    # ── Progress: always reward forward motion ────────────────────────────────
    reward = reward + 0.4 * clip(speed_ms / 40.0, 0.0, 1.0)

    # ── TTC safety penalty: linear -1 below 3 s ──────────────────────────────
    if ttc < 3.0:
        ttc_penalty = -(1.0 - ttc / 3.0)
        reward = reward + 0.3 * ttc_penalty

    # ── Collision: hard penalty ───────────────────────────────────────────────
    if collided:
        reward = reward - 20.0

    # ── Overtake bonus ────────────────────────────────────────────────────────
    if overtook:
        reward = reward + 2.0

    # ── Comfort: penalise harsh jerk and acceleration ─────────────────────────
    abs_jerk = abs(long_jerk)
    if abs_jerk > 2.0:
        jerk_pen = -clip((abs_jerk - 2.0) / 2.0, 0.0, 1.0)
        reward = reward + 0.02 * jerk_pen

    abs_accel = abs(accel_ms2)
    if abs_accel > 3.0:
        accel_pen = -clip((abs_accel - 3.0) / 3.0, 0.0, 1.0)
        reward = reward + 0.02 * accel_pen

    return reward
