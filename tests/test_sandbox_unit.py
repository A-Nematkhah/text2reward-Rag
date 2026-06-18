import pytest

from reward_sandbox import validate_reward_code, execute_reward


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
    code = "def compute_reward(state):\n" "    return state['speed_ms'] * 0.1\n"
    # execute_reward expects validated code; validate first
    ok, err = validate_reward_code(code)
    assert ok, err
    val = execute_reward(code, {"speed_ms": 20.0})
    assert isinstance(val, float)
    assert abs(val - 2.0) < 1e-6


def test_execute_reward_non_numeric_raises():
    code = "def compute_reward(state):\n    return 'bad'"
    ok, err = validate_reward_code(code)
    # Even if AST allows string constant, execute_reward should raise on non-numeric
    assert ok
    with pytest.raises(TypeError):
        execute_reward(code, {})


def test_execute_reward_timeout():
    # A busy computation (but loops are forbidden) — emulate heavy op via recursion depth
    code = (
        "def compute_reward(state):\n"
        "    x = 0\n"
        "    for i in range(1000000):\n"
        "        x += i\n"
        "    return float(x)"
    )
    # This should be rejected by AST because loops are forbidden
    ok, err = validate_reward_code(code)
    assert not ok
