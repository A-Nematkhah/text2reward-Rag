"""Shared highway-v0 configuration for training and evaluation.

``ENV_CONFIG`` is passed to ``gym.make("highway-v0", config=...)``. Intrinsic
env reward terms are zeroed because shaping comes from ``reward_program.py``.
"""

ENV_CONFIG = {
    "vehicles_count": 30,
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 60,
    "lanes_count": 4,
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": True,
        "absolute": False,
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "reward_speed_range": [20, 30],
    "collision_reward": -1.0,
    # Shaped reward comes from reward_program.py — keep intrinsic env speed term off.
    "high_speed_reward": 0.0,
    "right_lane_reward": 0.0,
    "lane_change_reward": 0.0,
}
