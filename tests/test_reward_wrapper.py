from txt2reward.reward.wrapper import _fallback_reward


def test_fallback_collision_penalty():
    state = {"speed_ms": 20.0, "collided": True}
    assert _fallback_reward(state) < 0
