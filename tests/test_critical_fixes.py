import json
import os
import tempfile

from reward_designer import RewardDesigner, _is_placeholder_code, write_default_reward_program


def test_is_placeholder_code():
    assert _is_placeholder_code("")
    assert _is_placeholder_code('def compute_reward(state):\n    return 0.0  # placeholder\n')
    assert not _is_placeholder_code('def compute_reward(state):\n    return 1.0\n')


def test_disk_reward_not_clobbered_by_stale_archive():
    workdir = tempfile.mkdtemp()
    archive_path = os.path.join(workdir, "reward_archive.json")
    reward_path = os.path.join(workdir, "reward_program.py")

    new_code = 'def compute_reward(state):\n    return 42.0\n'
    with open(reward_path, "w", encoding="utf-8") as f:
        f.write(new_code)

    old_code = 'def compute_reward(state):\n    return 1.0\n'
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

    designer = RewardDesigner(
        archive_path=archive_path,
        reward_path=reward_path,
        verbose=False,
    )

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
