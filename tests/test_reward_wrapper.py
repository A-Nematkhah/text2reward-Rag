import gymnasium as gym
from reward_wrapper import RewardWrapper

def test_wrapper_reset():
    env=RewardWrapper(gym.make('highway-v0'))
    obs,info=env.reset()
    assert obs is not None
