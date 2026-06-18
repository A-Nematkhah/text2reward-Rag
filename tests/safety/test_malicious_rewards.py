from reward_sandbox import validate_reward_code

def test_os_attack_rejected():
    code='import os\n\ndef compute_reward(state):\n    return 0'
    assert validate_reward_code(code).valid is False
