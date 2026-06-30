"""RewardDesigner evolution, resume, critique parsing, and archive safety."""

import json
import os
import tempfile

import pytest
from txt2reward.archive.archive import FAILURE_MODE_TAGS, STRENGTH_MODE_TAGS, RewardArchive, parse_structured_critique
from txt2reward.llm.designer import RewardDesigner, _is_placeholder_code, write_default_reward_program

from tests.helpers import passing_reward_code


def test_is_placeholder_code():
    assert _is_placeholder_code("")
    assert _is_placeholder_code("def compute_reward(state):\n    return 0.0  # placeholder\n")
    assert not _is_placeholder_code("def compute_reward(state):\n    return 1.0\n")


def test_disk_reward_not_clobbered_by_stale_archive():
    workdir = tempfile.mkdtemp()
    archive_path = os.path.join(workdir, "reward_archive.json")
    reward_path = os.path.join(workdir, "reward_program.py")

    new_code = "def compute_reward(state):\n    return 42.0\n"
    with open(reward_path, "w", encoding="utf-8") as f:
        f.write(new_code)

    old_code = "def compute_reward(state):\n    return 1.0\n"
    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {},
                "entries": [
                    {
                        "generation": 0,
                        "reward_code": old_code,
                        "metrics": {"mean_speed": 20.0, "crash_rate": 0.1},
                        "fitness": 0.1,
                        "critique": "",
                        "timestamp": "2026-01-01T00:00:00",
                    }
                ],
            },
            f,
        )

    designer = RewardDesigner(archive_path=archive_path, reward_path=reward_path, verbose=False)
    with open(reward_path, encoding="utf-8") as f:
        assert "42.0" in f.read()
    assert designer._current_code == new_code


def test_designer_resumes_episode_count():
    designer = RewardDesigner(initial_episode_count=170, verbose=False)
    assert designer._episode_count == 170


def test_write_default_reward_program():
    workdir = tempfile.mkdtemp()
    path = os.path.join(workdir, "reward_program.py")
    write_default_reward_program(path)
    with open(path, encoding="utf-8") as f:
        body = f.read()
    assert "def compute_reward" in body
    assert "clip(" in body


def test_parse_structured_critique_filters_unknown_llm_tags():
    critique = (
        'CRITIQUE_META:{"failure_modes": ["passive_driving", "made_up_mode"], '
        '"strengths": ["high_speed", "fake_strength"], "summary": "ok"}'
    )
    meta = parse_structured_critique(critique, {"mean_speed": 12.0, "crash_rate": 0.0})
    assert meta["failure_modes"] == ["passive_driving"]
    assert meta["strengths"] == ["high_speed"]
    assert "passive_driving" in FAILURE_MODE_TAGS
    assert "high_speed" in STRENGTH_MODE_TAGS


def test_parse_structured_critique_extracts_nested_json():
    critique = (
        "The agent is too passive.\n"
        'CRITIQUE_META:{"failure_modes": ["passive_driving"], '
        '"strengths": ["low crash"], "summary": "speed up"}'
    )
    meta = parse_structured_critique(critique, {"mean_speed": 12.0, "crash_rate": 0.0})
    assert meta["failure_modes"] == ["passive_driving"]
    assert meta["summary"] == "speed up"


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
    designer._evolve = lambda: calls.__setitem__("n", calls["n"] + 1) or False  # type: ignore[method-assign]
    assert designer.maybe_evolve() is False
    assert calls["n"] == 1


@pytest.mark.parametrize(
    "initial_episode_count,batch_size,expected_calls,expected_last_index",
    [
        (56, 4, 1, None),
        (58, 4, 1, 1),
    ],
)
def test_maybe_evolve_fires_once_per_boundary(initial_episode_count, batch_size, expected_calls, expected_last_index):
    designer = RewardDesigner(
        initial_episode_count=initial_episode_count,
        warmup_episodes=40,
        evolve_every=20,
        verbose=False,
    )
    stats = {"mean_speed": 20, "collisions": 0, "steps": 10, "total_overtakes": 0}
    calls = {"n": 0}
    designer._evolve = lambda: calls.__setitem__("n", calls["n"] + 1) or False  # type: ignore[method-assign]

    for _ in range(batch_size):
        designer.accumulate_episode(stats)
        designer.maybe_evolve()

    assert calls["n"] == expected_calls
    if expected_last_index is not None:
        assert designer._last_evolution_index == expected_last_index


def test_resume_after_skipped_boundary_retries_evolution():
    from txt2reward.training.logger import TrainingLogger

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
    designer._evolve = lambda: calls.__setitem__("n", calls["n"] + 1) or False  # type: ignore[method-assign]
    assert designer.maybe_evolve() is False
    assert calls["n"] == 1


def test_active_generation_unchanged_when_llm_fails(monkeypatch):
    workdir = tempfile.mkdtemp()
    archive_path = os.path.join(workdir, "reward_archive.json")
    reward_path = os.path.join(workdir, "reward_program.py")
    code = passing_reward_code()
    with open(reward_path, "w", encoding="utf-8") as f:
        f.write(code)

    designer = RewardDesigner(archive_path=archive_path, reward_path=reward_path, verbose=False)
    before = designer.get_weights()["generation"]

    monkeypatch.setattr(designer, "_call_generate_with_repair", lambda _ctx, **_: None)
    designer._episode_stats = [{"mean_speed": 20, "collisions": 0, "steps": 10, "total_overtakes": 0}]
    designer._episode_count = designer.warmup_episodes
    designer._evolve()

    assert designer.get_weights()["generation"] == before
    assert len(designer.archive.entries) == 1


def test_placeholder_not_archived(monkeypatch):
    workdir = tempfile.mkdtemp()
    designer = RewardDesigner(
        archive_path=os.path.join(workdir, "reward_archive.json"),
        reward_path=os.path.join(workdir, "reward_program.py"),
        verbose=False,
    )
    monkeypatch.setattr(
        designer,
        "_call_generate_with_repair",
        lambda _ctx, **_: "def compute_reward(state):\n    return 1.0\n",
    )
    designer._episode_stats = [{"mean_speed": 20, "collisions": 0, "steps": 10, "total_overtakes": 0}]
    designer._episode_count = designer.warmup_episodes
    designer._evolve()

    assert len(designer.archive.entries) == 0
    assert not _is_placeholder_code(designer._current_code)


def test_evolution_frozen_when_crash_rate_high(monkeypatch):
    workdir = tempfile.mkdtemp()
    reward_path = os.path.join(workdir, "reward_program.py")
    code = passing_reward_code()
    with open(reward_path, "w", encoding="utf-8") as f:
        f.write(code)

    designer = RewardDesigner(
        archive_path=os.path.join(workdir, "reward_archive.json"),
        reward_path=reward_path,
        evolve_max_crash_rate=0.70,
        verbose=False,
    )
    generate_called = {"n": 0}
    monkeypatch.setattr(
        designer,
        "_call_generate_with_repair",
        lambda *_a, **_k: generate_called.__setitem__("n", generate_called["n"] + 1) or None,
    )

    crashed_ep = {"mean_speed": 28.0, "collisions": 1, "steps": 40, "total_overtakes": 0}
    designer._episode_stats = [crashed_ep]
    designer._episode_count = designer.warmup_episodes

    assert designer._evolve() is False
    assert len(designer.archive.entries) == 0
    assert generate_called["n"] == 0
    assert designer._episode_stats == []


def test_evolution_proceeds_when_crash_rate_below_threshold(monkeypatch):
    workdir = tempfile.mkdtemp()
    reward_path = os.path.join(workdir, "reward_program.py")
    code = passing_reward_code()
    with open(reward_path, "w", encoding="utf-8") as f:
        f.write(code)

    designer = RewardDesigner(
        archive_path=os.path.join(workdir, "reward_archive.json"),
        reward_path=reward_path,
        evolve_max_crash_rate=0.70,
        verbose=False,
    )
    monkeypatch.setattr(designer, "_call_generate_with_repair", lambda *_a, **_k: None)
    monkeypatch.setattr(designer, "_call_critique", lambda *_a, **_k: "")

    safe_ep = {"mean_speed": 26.0, "collisions": 0, "steps": 100, "total_overtakes": 2}
    designer._episode_stats = [safe_ep]
    designer._episode_count = designer.warmup_episodes

    assert designer._evolve() is False
    assert len(designer.archive.entries) == 1
    assert designer.archive.entries[0]["metrics"]["crash_rate"] == 0.0


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


def test_aggregate_metrics_pools_step_ttc_values():
    stats = [
        {"ttc_vals": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], "collisions": 0},
        {"ttc_vals": [20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0], "collisions": 0},
    ]
    metrics = RewardDesigner._aggregate_metrics(stats)
    assert metrics["min_ttc"] == 1.0
    assert metrics["p10_ttc"] < 6.0
