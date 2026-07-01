"""Evaluation pipeline: TTC pooling and validated archive reward restore."""

import os
import tempfile

import pytest
from stable_baselines3.common.monitor import Monitor
from txt2reward.archive.archive import RewardArchive, compute_fitness
from txt2reward.core.metrics import aggregate_eval_fitness_metrics, pool_ttc_p10_min
from txt2reward.llm.validation import write_validated_reward_tempfile
from txt2reward.reward.wrapper import LLMRewardWrapper
from txt2reward.sandbox.sandbox import validate_reward_code
from txt2reward.training.env_factory import make_highway_env

from tests.helpers import passing_reward_code


def _unwrap_wrapper(env):
    while isinstance(env, Monitor):
        env = env.env
    return env


def test_pool_ttc_metrics_uses_step_values():
    results = [
        {"ttc_vals": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]},
        {"ttc_vals": [20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0]},
    ]
    p10, min_ttc = pool_ttc_p10_min(results)
    assert min_ttc == 1.0
    assert p10 < 6.0


def test_write_validated_reward_rejects_invalid_code():
    workdir = tempfile.mkdtemp()
    bad_code = "import os\n\ndef compute_reward(state):\n    return 1.0\n"
    archive = RewardArchive(os.path.join(workdir, "archive.json"))
    archive.add_entry(
        reward_code=bad_code,
        metrics={"mean_speed": 20.0, "crash_rate": 0.0},
        critique="",
    )
    assert not validate_reward_code(bad_code)[0]

    with pytest.raises(ValueError, match="failed validation"):
        write_validated_reward_tempfile(bad_code, 0)


def test_write_validated_reward_accepts_safe_code():
    code = passing_reward_code()
    workdir = tempfile.mkdtemp()
    archive = RewardArchive(os.path.join(workdir, "archive.json"))
    archive.add_entry(
        reward_code=code,
        metrics={"mean_speed": 20.0, "crash_rate": 0.0},
        critique="",
    )
    entry = archive.get_by_generation(0)
    path = write_validated_reward_tempfile(entry["reward_code"], 0)
    try:
        with open(path, encoding="utf-8") as f:
            assert "compute_reward" in f.read()
    finally:
        os.remove(path)


def test_no_shaped_env_still_collects_stats_via_wrapper():
    env = make_highway_env(use_shaped=False)
    wrapper = _unwrap_wrapper(env)
    assert isinstance(wrapper, LLMRewardWrapper)
    assert wrapper.apply_shaped_reward is False
    env.close()


def test_aggregate_eval_fitness_metrics_includes_lane_overtake_totals():
    results = [
        {"crashed": False, "mean_speed": 26.0, "overtakes": 2, "lane_changes": 3, "steps": 40},
        {"crashed": True, "mean_speed": 24.0, "overtakes": 1, "lane_changes": 1, "steps": 20},
        {"crashed": False, "mean_speed": 28.0, "overtakes": 0, "lane_changes": 2, "steps": 50},
    ]
    metrics = aggregate_eval_fitness_metrics(results)
    assert metrics["n_episodes"] == 3
    assert metrics["total_overtakes"] == 3
    assert metrics["total_lane_changes"] == 6
    assert metrics["mean_overtakes"] == pytest.approx(1.0)
    assert metrics["safe_overtake_ratio"] == pytest.approx(0.5)
    assert metrics["lane_change_rate"] == pytest.approx(2.0)
    assert metrics["crash_rate"] == pytest.approx(1 / 3)


def test_eval_fitness_uses_lane_overtake_subscores_not_trivial_defaults():
    efficient = aggregate_eval_fitness_metrics(
        [
            {"crashed": False, "mean_speed": 27.0, "overtakes": 5, "lane_changes": 6, "steps": 40},
            {"crashed": False, "mean_speed": 27.0, "overtakes": 5, "lane_changes": 6, "steps": 40},
        ]
    )
    thrash = aggregate_eval_fitness_metrics(
        [
            {"crashed": False, "mean_speed": 27.0, "overtakes": 1, "lane_changes": 20, "steps": 40},
            {"crashed": False, "mean_speed": 27.0, "overtakes": 1, "lane_changes": 20, "steps": 40},
        ]
    )
    assert efficient["safe_overtake_ratio"] > thrash["safe_overtake_ratio"]
    assert compute_fitness(efficient) > compute_fitness(thrash)
