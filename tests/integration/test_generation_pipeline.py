from reward_designer import _full_validation_pipeline


def test_generation_pipeline_passes_full_validation():
    code = (
        "def compute_reward(state):\n"
        '    if state["collided"]:\n'
        "        return -30.0\n"
        "    reward = 0.0\n"
        '    reward += 0.2 * (state["speed_ms"] / 30.0) ** 2\n'
        '    reward += 3.5 if state["overtook"] else 0.0\n'
        '    reward += 0 if state["ttc"] > 3 else -0.2 * (3 - state["ttc"])\n'
        '    reward -= 0.02 * abs(state["long_jerk"])\n'
        '    reward -= 0.02 * abs(state["lat_jerk"])\n'
        "    return reward\n"
    )
    ok, err = _full_validation_pipeline(code)
    assert ok, err
