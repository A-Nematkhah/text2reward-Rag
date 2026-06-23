"""Tests for low-severity bug fixes."""

import os
import tempfile

import pytest

from reward_archive import FAILURE_MODE_TAGS, STRENGTH_MODE_TAGS, parse_structured_critique
from train import build_vec_env


def test_parse_structured_critique_filters_unknown_llm_tags():
    critique = (
        'CRITIQUE_META:{"failure_modes": ["passive_driving", "made_up_mode"], '
        '"strengths": ["high_speed", "fake_strength"], "summary": "ok"}'
    )
    meta = parse_structured_critique(critique, {"mean_speed": 12.0, "crash_rate": 0.0})
    assert meta["failure_modes"] == ["passive_driving"]
    assert meta["strengths"] == ["high_speed"]
    assert "made_up_mode" not in meta["failure_modes"]
    assert "fake_strength" not in meta["strengths"]
    assert "passive_driving" in FAILURE_MODE_TAGS
    assert "high_speed" in STRENGTH_MODE_TAGS


def test_build_vec_env_falls_back_with_allow_dummy_env(monkeypatch):
    class _FailSubproc:
        def __init__(self, env_fns):
            raise OSError("subprocess unavailable")

    created = {"dummy": False}

    class _OkDummy:
        def __init__(self, env_fns):
            created["dummy"] = True
            self.env_fns = env_fns

    monkeypatch.setattr("train.SubprocVecEnv", _FailSubproc)
    monkeypatch.setattr("train.DummyVecEnv", _OkDummy)

    env_fns = [lambda: None]
    env = build_vec_env(env_fns, allow_dummy_env=True)
    assert created["dummy"] is True
    assert len(env.env_fns) == 1


def test_build_vec_env_raises_without_allow_dummy_env(monkeypatch):
    class _FailSubproc:
        def __init__(self, env_fns):
            raise OSError("subprocess unavailable")

    monkeypatch.setattr("train.SubprocVecEnv", _FailSubproc)

    with pytest.raises(SystemExit, match="--allow-dummy-env"):
        build_vec_env([lambda: None], allow_dummy_env=False)
