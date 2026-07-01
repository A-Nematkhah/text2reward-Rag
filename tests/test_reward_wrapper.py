from txt2reward.config.training import (
    REWARD_COLLISION_CLIP_MIN,
    REWARD_STEP_CLIP_MAX,
    REWARD_STEP_CLIP_MIN,
)
from txt2reward.reward.clip import clip_shaped_reward
from txt2reward.reward.wrapper import _clip_shaped_reward, _fallback_reward


def test_fallback_collision_penalty():
    state = {"speed_ms": 20.0, "collided": True}
    assert _fallback_reward(state) < 0


def test_clip_shaped_reward_bounds_non_collision():
    assert _clip_shaped_reward(-100.0) == REWARD_STEP_CLIP_MIN
    assert _clip_shaped_reward(100.0) == REWARD_STEP_CLIP_MAX
    assert _clip_shaped_reward(3.5) == 3.5


def test_clip_shaped_reward_preserves_sign_in_range():
    assert _clip_shaped_reward(-5.0) == -5.0
    assert _clip_shaped_reward(7.0) == 7.0


def test_collision_clip_preserves_full_penalty():
    assert clip_shaped_reward(-80.0, collided=True) == -80.0
    assert clip_shaped_reward(-90.0, collided=True) == -90.0
    assert clip_shaped_reward(5.0, collided=True) == 0.0


def test_symmetric_clip_would_hide_collision_penalty():
    """Regression guard: old symmetric [-10,10] clip collapsed -80 to -10."""
    old = max(REWARD_STEP_CLIP_MIN, min(REWARD_STEP_CLIP_MAX, -80.0))
    assert old == REWARD_STEP_CLIP_MIN
    assert clip_shaped_reward(-80.0, collided=True) < old


def test_collision_clip_respects_floor():
    assert clip_shaped_reward(-200.0, collided=True) == REWARD_COLLISION_CLIP_MIN


def test_debug_reward_logs_raw_vs_clipped_on_collision():
    import os
    from unittest.mock import MagicMock, patch

    from txt2reward.reward import wrapper as wrapper_mod
    from txt2reward.reward.wrapper import LLMRewardWrapper

    base_state = {
        "speed_ms": 28.0,
        "front_dist": 50.0,
        "ttc": 12.0,
        "rel_vel_ms": 0.0,
        "lane": 1,
        "overtook": False,
        "lane_changed": False,
        "collided": True,
        "nearby_vehicles": 2,
        "accel_ms2": 0.0,
        "long_jerk": 0.0,
        "lat_jerk": 0.0,
    }
    parsed = {
        "speed_ms": 28.0,
        "lat_vel_ms": 0.0,
        "overtake_tracks": [],
        "next_track_id": 0,
        "lane": 1,
        "front_dist": 50.0,
        "rel_vel_ms": 0.0,
        "ttc": 12.0,
        "accel_ms2": 0.0,
        "long_jerk": 0.0,
        "lat_jerk": 0.0,
        "nearby_vehicles": 2,
        "overtook": False,
        "lane_changed": False,
    }

    env = MagicMock()
    env.reset.return_value = (None, {})
    env.step.return_value = (None, 0.0, True, False, {"crashed": True})

    wrapper = LLMRewardWrapper(env, apply_shaped_reward=True)
    wrapper.reload_interval = 10_000
    wrapper._global_step = 1

    with (
        patch.dict(os.environ, {"DEBUG_REWARD": "1"}),
        patch.object(wrapper_mod, "_parse_full_obs", return_value=parsed),
        patch.object(wrapper_mod, "build_state", return_value=base_state),
        patch.object(wrapper_mod.log, "debug") as mock_debug,
    ):
        wrapper._reward_fn = lambda _state: -90.0
        wrapper.step(0)
        collision_msg = mock_debug.call_args[0][0]
        assert "collided=True" in collision_msg
        assert "reward=-90.000" in collision_msg

        wrapper._global_step = 999
        wrapper._reward_fn = lambda _state: 50.0
        base_state["collided"] = False
        env.step.return_value = (None, 0.0, True, False, {"crashed": False})
        wrapper.step(0)
        clip_msg = mock_debug.call_args[0][0]
        assert "raw=50.000 -> clipped=10.000" in clip_msg
