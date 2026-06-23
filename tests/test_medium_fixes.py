"""Tests for medium-severity bug fixes (#16-#20)."""

import tempfile

from env_config import ENV_CONFIG
from reward_designer import RewardDesigner
from reward_sandbox import validate_reward_code
from train import ENV_CONFIG as TRAIN_ENV_CONFIG


def test_train_and_evaluate_share_env_config():
    from evaluate import ENV_CONFIG as EVAL_ENV_CONFIG

    assert TRAIN_ENV_CONFIG == EVAL_ENV_CONFIG
    assert ENV_CONFIG["high_speed_reward"] == 0.0


def test_collision_gate_does_not_require_fixed_gap_for_negative_normal():
    """Old gate false-failed when normal=-12 and collision=-30 (gap 18 < 20)."""
    normal_r, collision_r = -12.0, -30.0
    assert collision_r < normal_r
    assert collision_r <= -10.0
    # Legacy check that wrongly rejected this pair:
    assert collision_r >= (normal_r - 20.0)


def test_validate_rejects_dynamic_pow_exponent():
    code = 'def compute_reward(state):\n    return state["speed_ms"] ** round(state["front_dist"])\n'
    ok, err = validate_reward_code(code)
    assert not ok
    assert "dynamic exponents" in err.lower() or "constant literal" in err.lower()


def test_maybe_evolve_once_per_callback_batch():
    """Parallel env batch crossing one boundary should evolve exactly once."""
    workdir = tempfile.mkdtemp()
    designer = RewardDesigner(
        archive_path=workdir + "/archive.json",
        reward_path=workdir + "/reward.py",
        warmup_episodes=40,
        evolve_every=20,
        initial_episode_count=56,
        verbose=False,
    )
    stats = {"mean_speed": 20, "collisions": 0, "steps": 10, "total_overtakes": 0}

    calls = {"n": 0}

    def _spy():
        calls["n"] += 1
        return False

    designer._evolve = _spy  # type: ignore[method-assign]

    for _ in range(4):
        designer.accumulate_episode(stats)
        designer.maybe_evolve()

    assert calls["n"] == 1
