from reward_sandbox import validate_reward_code

def test_import_blocked():
    code='import os\n\ndef compute_reward(state):\n    return 1'
    assert validate_reward_code(code).valid is False
