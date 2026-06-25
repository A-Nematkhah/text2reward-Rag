"""Tests for high-severity bug fixes (#5-#15)."""

import json
import os
import tempfile

from reward_archive import RewardArchive
from reward_designer import RewardDesigner, _is_placeholder_code, DEFAULT_BOOTSTRAP_REWARD_BODY
from trajectory_bank import build_trajectory_bank, evaluate_consistency


def _passing_reward_code() -> str:
    return DEFAULT_BOOTSTRAP_REWARD_BODY.strip()


def test_trajectory_metrics_include_robust_ttc():
    from trajectory_bank import _aggregate_trajectory_metrics

    states = [
        {"speed_ms": 20.0, "collided": False, "overtook": False, "long_jerk": 0.0, "ttc": 30.0},
        {"speed_ms": 20.0, "collided": False, "overtook": False, "long_jerk": 0.0, "ttc": 1.0},
    ]
    m = _aggregate_trajectory_metrics(states)
    assert "p10_ttc" in m
    assert "min_ttc" in m
    assert m["min_ttc"] == 1.0


def test_hard_violations_fail_consistency_gate():
    bank = build_trajectory_bank()

    def bad_reward(state):
        return 100.0 if state["collided"] else 0.0

    ok, report, _ = evaluate_consistency(bad_reward, bank=bank, max_violation_rate=1.0)
    assert not ok
    assert "hard safety violations" in report


def test_active_generation_unchanged_when_llm_fails(monkeypatch):
    workdir = tempfile.mkdtemp()
    archive_path = os.path.join(workdir, "reward_archive.json")
    reward_path = os.path.join(workdir, "reward_program.py")
    code = _passing_reward_code()
    with open(reward_path, "w", encoding="utf-8") as f:
        f.write(code)

    designer = RewardDesigner(archive_path=archive_path, reward_path=reward_path, verbose=False)
    before = designer.get_weights()["generation"]

    monkeypatch.setattr(designer, "_call_generate_with_repair", lambda _ctx: None)
    designer._episode_stats = [{"mean_speed": 20, "collisions": 0, "steps": 10, "total_overtakes": 0}]
    designer._episode_count = designer.warmup_episodes
    designer._evolve()

    assert designer.get_weights()["generation"] == before
    assert len(designer.archive.entries) == 1


def test_placeholder_not_archived(monkeypatch):
    workdir = tempfile.mkdtemp()
    archive_path = os.path.join(workdir, "reward_archive.json")
    reward_path = os.path.join(workdir, "reward_program.py")

    designer = RewardDesigner(archive_path=archive_path, reward_path=reward_path, verbose=False)
    monkeypatch.setattr(
        designer,
        "_call_generate_with_repair",
        lambda _ctx: 'def compute_reward(state):\n    return 1.0\n',
    )
    designer._episode_stats = [{"mean_speed": 20, "collisions": 0, "steps": 10, "total_overtakes": 0}]
    designer._episode_count = designer.warmup_episodes
    designer._evolve()

    assert len(designer.archive.entries) == 0
    assert not _is_placeholder_code(designer._current_code)


def test_archive_remove_generation():
    workdir = tempfile.mkdtemp()
    path = os.path.join(workdir, "reward_archive.json")
    archive = RewardArchive(path)
    archive.add_entry("def compute_reward(state):\n    return 1.0\n", {"mean_speed": 20, "crash_rate": 0.1})
    archive.add_entry("def compute_reward(state):\n    return 2.0\n", {"mean_speed": 22, "crash_rate": 0.1})
    assert archive.remove_generation(0)
    assert len(archive.entries) == 1
    assert archive.entries[0]["generation"] == 0

    archive.add_entry("def compute_reward(state):\n    return 3.0\n", {"mean_speed": 24, "crash_rate": 0.1})
    generations = [e["generation"] for e in archive.entries]
    assert generations == [0, 1]
    assert len(set(generations)) == len(generations)


def test_generation_pipeline_validation():
    from reward_designer import _full_validation_pipeline

    ok, err, _ = _full_validation_pipeline(_passing_reward_code())
    assert ok, err
