"""Reward sandbox: AST validation and timed execution."""

import pytest
from txt2reward.sandbox.sandbox import execute_reward, validate_reward_code


def test_validate_disallows_imports():
    code = "import os\n\ndef compute_reward(state):\n    return 1.0"
    ok, err = validate_reward_code(code)
    assert not ok
    assert "Import" in err or "Forbidden" in err or "import" in err.lower()


def test_validate_disallows_attribute_access():
    code = "def compute_reward(state):\n    return state.__class__"
    ok, err = validate_reward_code(code)
    assert not ok


def test_execute_reward_returns_float():
    code = "def compute_reward(state):\n    return state['speed_ms'] * 0.1\n"
    ok, err = validate_reward_code(code)
    assert ok, err
    val = execute_reward(code, {"speed_ms": 20.0})
    assert isinstance(val, float)
    assert abs(val - 2.0) < 1e-6


def test_execute_reward_non_numeric_raises():
    code = "def compute_reward(state):\n    return 'bad'"
    ok, err = validate_reward_code(code)
    assert ok
    with pytest.raises(TypeError):
        execute_reward(code, {})


def test_execute_reward_timeout():
    code = "def compute_reward(state):\n    x = 0\n    for i in range(1000000):\n        x += i\n    return float(x)"
    ok, err = validate_reward_code(code)
    assert not ok
