"""Vectorized highway-env factory for PPO training."""

from __future__ import annotations

import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from txt2reward.config.env import ENV_CONFIG
from txt2reward.config.paths import REWARD_PROGRAM_PATH
from txt2reward.config.training import DEFAULT_RELOAD_INTERVAL
from txt2reward.core.log import get_logger
from txt2reward.reward.wrapper import LLMRewardWrapper

log = get_logger("train")


def make_highway_env(
    *,
    render_mode: str | None = None,
    use_shaped: bool = True,
    reload_interval: int = DEFAULT_RELOAD_INTERVAL,
    reward_path: str = REWARD_PROGRAM_PATH,
    monitor: bool = True,
) -> gym.Env:
    """Create a single highway-v0 env with optional shaped reward and Monitor."""
    config = dict(ENV_CONFIG)
    kwargs: dict = {"config": config}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    env = gym.make("highway-v0", **kwargs)
    if use_shaped:
        env = LLMRewardWrapper(
            env,
            reload_interval=reload_interval,
            num_lanes=ENV_CONFIG["lanes_count"],
            reward_path=reward_path,
        )
    if monitor:
        env = Monitor(env)
    return env


def make_env(rank: int = 0, reload_interval: int = DEFAULT_RELOAD_INTERVAL, reward_path: str = REWARD_PROGRAM_PATH):
    """Factory for SubprocVecEnv workers (rank is unused; kept for SB3 convention)."""

    def _init():
        return make_highway_env(
            reload_interval=reload_interval,
            reward_path=reward_path,
            use_shaped=True,
            monitor=True,
        )

    return _init


def build_vec_env(env_fns, *, allow_dummy_env: bool = False):
    """Create a vectorized env; optionally fall back to DummyVecEnv on failure.

    Args:
        env_fns: List of zero-argument callables returning ``gym.Env``.
        allow_dummy_env: If True, use single-process DummyVecEnv when
            SubprocVecEnv cannot start worker processes.

    Returns:
        ``SubprocVecEnv`` or ``DummyVecEnv``.

    Side effects:
        Logs the chosen backend; may call ``SystemExit`` when subprocess env
        fails and ``allow_dummy_env`` is False.
    """
    try:
        vec_env = SubprocVecEnv(env_fns)
        log.info("[train] Using SubprocVecEnv with %s workers", len(env_fns))
        return vec_env
    except Exception as e:
        if allow_dummy_env:
            log.warning("[train] SubprocVecEnv failed (%s), falling back to DummyVecEnv", e)
            return DummyVecEnv(env_fns)
        raise SystemExit(
            f"[train] SubprocVecEnv failed ({e}). "
            "Parallel workers require a working subprocess environment. "
            "Pass --allow-dummy-env to fall back to single-process DummyVecEnv."
        ) from e
