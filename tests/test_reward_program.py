import pytest
from reward_program import compute_reward

def test_collision_penalty():
    state={'collision':True,'speed_ms':20.0}
    assert compute_reward(state) < -5
