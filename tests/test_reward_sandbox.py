from reward_sandbox import validate_reward_code


def test_import_blocked():
    code = "import os\n\ndef compute_reward(state):\n    return 1"
    ok, err = validate_reward_code(code)
    assert ok is False
