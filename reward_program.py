"""
reward_program.py — generation 2
"""

def compute_reward(state):
    if state["collided"]:
        return -90.0
    speed = state["speed_ms"]
    target_speed = 28.0
    speed_reward = clip(speed * 0.07, 0.0, 2.0)
    speed_gap = clip((target_speed - speed) / target_speed, 0.0, 1.0)
    clear_no_target = (
        state["front_dist"] > 41.0
        and state["ttc"] > 6.0
        and state["rel_vel_ms"] >= -1.0
    )
    overtake_opportunity = (
        state["front_dist"] < 45.0
        and state["rel_vel_ms"] < -1.0
        and state["ttc"] > 3.0
    )
    cruise_tax = -2.5 * speed_gap if clear_no_target and not state["overtook"] else 0.0
    above_target_passive = (
        -3.5 * clip((speed - 22.0) / 8.0, 0.0, 1.0)
        if clear_no_target and not state["overtook"] and speed > 22.0
        else 0.0
    )
    no_overtake_tax = -1.0 * speed_gap if clear_no_target and not state["overtook"] else 0.0
    static_passive = -0.75 if clear_no_target and not state["overtook"] and not state["lane_changed"] else 0.0
    missed_overtake_tax = -1.2 if overtake_opportunity and not state["overtook"] else 0.0
    ttc_penalty = (
        -6.0 if state["ttc"] < 1.0
        else -3.5 if state["ttc"] < 3.0
        else -1.5 if state["ttc"] < 5.0
        else 0.0
    )
    tailgate_penalty = -3.0 if state["front_dist"] < 23.0 and state["ttc"] < 4.5 else 0.0
    overtake_bonus = 3.5 if state["overtook"] else 0.0
    harsh_jerk = max(0.0, abs(state["long_jerk"]) - 2.5) + max(0.0, abs(state["lat_jerk"]) - 2.5)
    harsh_accel = max(0.0, abs(state["accel_ms2"]) - 3.0)
    jerk_penalty = -1.2 * harsh_jerk
    accel_penalty = -0.75 * harsh_accel
    lc_penalty = -0.85 if state["lane_changed"] and not state["overtook"] else 0.0
    passive_driving_penalty = -1.5 if speed < 24.0 and clear_no_target else 0.0
    return (
        speed_reward
        + cruise_tax
        + above_target_passive
        + no_overtake_tax
        + static_passive
        + missed_overtake_tax
        + ttc_penalty
        + tailgate_penalty
        + overtake_bonus
        + jerk_penalty
        + accel_penalty
        + lc_penalty
        + passive_driving_penalty
    )