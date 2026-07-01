"""Per-step shaped reward clipping (shared by wrapper and validation gates)."""

from __future__ import annotations

from typing import Any, Mapping

from txt2reward.config.training import (
    REWARD_COLLISION_CLIP_MAX,
    REWARD_COLLISION_CLIP_MIN,
    REWARD_STEP_CLIP_MAX,
    REWARD_STEP_CLIP_MIN,
)


def clip_shaped_reward(reward: float, *, collided: bool = False) -> float:
    """Match runtime PPO reward scaling; preserve full collision penalty magnitude."""
    if collided:
        return float(max(REWARD_COLLISION_CLIP_MIN, min(REWARD_COLLISION_CLIP_MAX, reward)))
    return float(max(REWARD_STEP_CLIP_MIN, min(REWARD_STEP_CLIP_MAX, reward)))


def clip_reward_for_state(reward: float, state: Mapping[str, Any]) -> float:
    """Clip a raw compute_reward return using the state's collided flag."""
    return clip_shaped_reward(reward, collided=bool(state.get("collided", False)))
