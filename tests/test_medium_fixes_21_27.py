"""Tests for medium-severity bug fixes (#21-#27)."""

import json
import os
import tempfile

from reward_archive import parse_structured_critique
from reward_designer import RewardDesigner, _smoke_test_reward_code
from training_logger import TrainingLogger


def test_designer_resumes_last_evolution_index():
    designer = RewardDesigner(
        initial_episode_count=60,
        initial_last_evolution_index=4,
        warmup_episodes=20,
        evolve_every=10,
        verbose=False,
    )
    assert designer._episode_count == 60
    assert designer._last_evolution_index == 4
    assert designer.maybe_evolve() is False

    for _ in range(10):
        designer.accumulate_episode({"mean_speed": 20, "collisions": 0, "steps": 10})

    calls = {"n": 0}
    original = designer._evolve

    def _spy():
        calls["n"] += 1
        return False

    designer._evolve = _spy  # type: ignore[method-assign]
    assert designer.maybe_evolve() is False
    assert calls["n"] == 1


def test_completed_evolution_index_uses_llm_updates():
    log = TrainingLogger(log_path=os.devnull)
    log._episode_n = 63
    log.llm_updates = [{"episode": 60}]
    assert log.completed_evolution_index(warmup_episodes=40, evolve_every=20) == 1


def test_completed_evolution_index_without_llm_updates():
    log = TrainingLogger(log_path=os.devnull)
    log._episode_n = 63
    assert log.completed_evolution_index(warmup_episodes=40, evolve_every=20) == -1


def test_maybe_evolve_catches_skipped_boundary_on_single_batch_call():
    designer = RewardDesigner(
        initial_episode_count=58,
        warmup_episodes=40,
        evolve_every=20,
        verbose=False,
    )
    stats = {"mean_speed": 20, "collisions": 0, "steps": 10, "total_overtakes": 0}

    for _ in range(4):
        designer.accumulate_episode(stats)

    calls = {"n": 0}

    def _spy():
        calls["n"] += 1
        return False

    designer._evolve = _spy  # type: ignore[method-assign]
    assert designer.maybe_evolve() is False
    assert calls["n"] == 1
    assert designer._last_evolution_index == 1


def test_resume_after_skipped_boundary_retries_evolution():
    log = TrainingLogger(log_path=os.devnull)
    log._episode_n = 63
    log.llm_updates = []

    designer = RewardDesigner(
        initial_episode_count=log.episode_count(),
        initial_last_evolution_index=log.completed_evolution_index(40, 20),
        warmup_episodes=40,
        evolve_every=20,
        verbose=False,
    )
    assert designer._last_evolution_index == -1

    calls = {"n": 0}

    def _spy():
        calls["n"] += 1
        return False

    designer._evolve = _spy  # type: ignore[method-assign]
    assert designer.maybe_evolve() is False
    assert calls["n"] == 1


def test_aggregate_metrics_pools_step_ttc_values():
    stats = [
        {"ttc_vals": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], "collisions": 0},
        {"ttc_vals": [20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0], "collisions": 0},
    ]
    metrics = RewardDesigner._aggregate_metrics(stats)
    assert metrics["min_ttc"] == 1.0
    # 20 pooled steps -> 10th percentile is not the mean of per-episode p10s (6.9 and 16.9).
    assert metrics["p10_ttc"] < 6.0


def test_parse_structured_critique_extracts_nested_json():
    critique = (
        "The agent is too passive.\n"
        'CRITIQUE_META:{"failure_modes": ["passive_driving"], '
        '"strengths": ["low crash"], "summary": "speed up"}'
    )
    meta = parse_structured_critique(critique, {"mean_speed": 12.0, "crash_rate": 0.0})
    assert meta["failure_modes"] == ["passive_driving"]
    assert meta["summary"] == "speed up"


def test_smoke_test_uses_execute_reward_path():
    code = (
        "def compute_reward(state):\n"
        "    if state['collided']:\n"
        "        return -30.0\n"
        "    return float(state['speed_ms']) * 0.01\n"
    )
    ok, err = _smoke_test_reward_code(code)
    assert ok, err


def test_make_env_preserves_absolute_reward_path():
    from train import make_env

    reward_path = os.path.abspath(os.path.join(tempfile.mkdtemp(), "reward_program.py"))
    with open(reward_path, "w", encoding="utf-8") as f:
        f.write('def compute_reward(state):\n    return 1.0\n')

    env = make_env(rank=0, reward_path=reward_path)()
    wrapper = env
    while hasattr(wrapper, "env"):
        if hasattr(wrapper, "reward_path"):
            break
        wrapper = wrapper.env
    assert os.path.isabs(wrapper.reward_path)
    assert wrapper.reward_path == reward_path
    env.close()
