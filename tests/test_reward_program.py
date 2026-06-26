from reward_program import compute_reward


def test_collision_penalty():
    # Provide full canonical state keys to avoid KeyError in generated code
    state = {
        "speed_ms": 20.0,
        "front_dist": 0.0,
        "ttc": 0.0,
        "rel_vel_ms": 0.0,
        "lane": 0,
        "overtook": False,
        "lane_changed": False,
        "collided": True,
        "nearby_vehicles": 0,
        "accel_ms2": 0.0,
        "long_jerk": 0.0,
        "lat_jerk": 0.0,
    }
    assert compute_reward(state) <= -40.0
