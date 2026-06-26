"""Training loop: env factory, vec env fallback, and training logger."""

import os
import tempfile

import pytest
from txt2reward.config.env import ENV_CONFIG
from txt2reward.training.env_factory import build_vec_env, make_env
from txt2reward.training.logger import TrainingLogger


def test_train_and_evaluate_share_env_config():
    from txt2reward.training.train import ENV_CONFIG as TRAIN_ENV_CONFIG

    assert TRAIN_ENV_CONFIG == ENV_CONFIG
    assert ENV_CONFIG["high_speed_reward"] == 0.0


def test_build_vec_env_falls_back_with_allow_dummy_env(monkeypatch):
    class _FailSubproc:
        def __init__(self, env_fns):
            raise OSError("subprocess unavailable")

    created = {"dummy": False}

    class _OkDummy:
        def __init__(self, env_fns):
            created["dummy"] = True
            self.env_fns = env_fns

    monkeypatch.setattr("txt2reward.training.env_factory.SubprocVecEnv", _FailSubproc)
    monkeypatch.setattr("txt2reward.training.env_factory.DummyVecEnv", _OkDummy)

    env = build_vec_env([lambda: None], allow_dummy_env=True)
    assert created["dummy"] is True
    assert len(env.env_fns) == 1


def test_build_vec_env_raises_without_allow_dummy_env(monkeypatch):
    class _FailSubproc:
        def __init__(self, env_fns):
            raise OSError("subprocess unavailable")

    monkeypatch.setattr("txt2reward.training.env_factory.SubprocVecEnv", _FailSubproc)

    with pytest.raises(SystemExit, match="--allow-dummy-env"):
        build_vec_env([lambda: None], allow_dummy_env=False)


def test_completed_evolution_index_uses_llm_updates():
    log = TrainingLogger(log_path=os.devnull)
    log._episode_n = 63
    log.llm_updates = [{"episode": 60}]
    assert log.completed_evolution_index(warmup_episodes=40, evolve_every=20) == 1


def test_completed_evolution_index_without_llm_updates():
    log = TrainingLogger(log_path=os.devnull)
    log._episode_n = 63
    assert log.completed_evolution_index(warmup_episodes=40, evolve_every=20) == -1


def test_make_env_preserves_absolute_reward_path():
    reward_path = os.path.abspath(os.path.join(tempfile.mkdtemp(), "reward_program.py"))
    with open(reward_path, "w", encoding="utf-8") as f:
        f.write("def compute_reward(state):\n    return 1.0\n")

    env = make_env(rank=0, reward_path=reward_path)()
    wrapper = env
    while hasattr(wrapper, "env"):
        if hasattr(wrapper, "reward_path"):
            break
        wrapper = wrapper.env
    assert os.path.isabs(wrapper.reward_path)
    assert wrapper.reward_path == reward_path
    env.close()
