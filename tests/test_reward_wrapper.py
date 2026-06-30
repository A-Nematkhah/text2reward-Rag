from txt2reward.config.training import REWARD_STEP_CLIP_MAX, REWARD_STEP_CLIP_MIN
from txt2reward.reward.wrapper import _clip_shaped_reward, _fallback_reward


def test_fallback_collision_penalty():
    state = {"speed_ms": 20.0, "collided": True}
    assert _fallback_reward(state) < 0


def test_clip_shaped_reward_bounds():
    assert _clip_shaped_reward(-100.0) == REWARD_STEP_CLIP_MIN
    assert _clip_shaped_reward(100.0) == REWARD_STEP_CLIP_MAX
    assert _clip_shaped_reward(3.5) == 3.5


def test_clip_shaped_reward_preserves_sign_in_range():
    assert _clip_shaped_reward(-5.0) == -5.0
    assert _clip_shaped_reward(7.0) == 7.0
