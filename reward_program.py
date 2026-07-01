"""
reward_program.py — active reward (hot-reloaded at runtime).
"""

def compute_reward(state):
    if state["collided"]:
        return -80.0
    speed = state["speed_ms"]
    target_speed = 28.0
    open_road = state["front_dist"] > 41.0 and state["ttc"] > 6.0
    speed_reward = clip(speed * 0.09, 0.0, 2.5)
    speed_gap = clip((target_speed - speed) / target_speed, 0.0, 1.0)
    cruise_tax = -2.0 * speed_gap if open_road and not state["overtook"] else 0.0
    above_target_passive = (
        -3.5 * clip((speed - 22.0) / 8.0, 0.0, 1.0)
        if open_road and not state["overtook"] and speed > 22.0
        else 0.0
    )
    if open_road and not state["overtook"]:
        no_overtake_tax = -0.85 * speed_gap if state["lane_changed"] else -1.2 * speed_gap
    else:
        no_overtake_tax = 0.0
    static_passive = -0.50 if open_road and not state["overtook"] and not state["lane_changed"] else 0.0
    ttc_penalty = (
        -5.0 if state["ttc"] < 1.0
        else -2.5 if state["ttc"] < 3.0
        else -1.0 if state["ttc"] < 5.0
        else 0.0
    )
    tailgate_penalty = -2.2 if state["front_dist"] < 22.0 and state["ttc"] < 4.5 else 0.0
    overtake_bonus = 3.0 if state["overtook"] else 0.0
    harsh_jerk = max(0.0, abs(state["long_jerk"]) - 2.0) + max(0.0, abs(state["lat_jerk"]) - 2.0)
    harsh_accel = max(0.0, abs(state["accel_ms2"]) - 2.5)
    jerk_penalty = -0.90 * harsh_jerk
    accel_penalty = -0.50 * harsh_accel
    lc_penalty = -0.55 if state["lane_changed"] and not state["overtook"] else 0.0
    return (
        speed_reward
        + cruise_tax
        + above_target_passive
        + no_overtake_tax
        + static_passive
        + ttc_penalty
        + tailgate_penalty
        + overtake_bonus
        + jerk_penalty
        + accel_penalty
        + lc_penalty
    )
