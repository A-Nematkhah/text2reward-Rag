"""Tests for evaluate.py medium-severity fixes."""

import json
import os
import tempfile

import pytest

from evaluate import _pool_ttc_metrics, _write_validated_archive_reward
from reward_archive import RewardArchive
from reward_designer import DEFAULT_BOOTSTRAP_REWARD_BODY
from reward_sandbox import validate_reward_code


def _passing_reward_code() -> str:
    return DEFAULT_BOOTSTRAP_REWARD_BODY.strip()


def test_pool_ttc_metrics_uses_step_values():
    results = [
        {"ttc_vals": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]},
        {"ttc_vals": [20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0]},
    ]
    p10, min_ttc = _pool_ttc_metrics(results)
    assert min_ttc == 1.0
    assert p10 < 6.0


def test_write_validated_archive_reward_rejects_invalid_code():
    workdir = tempfile.mkdtemp()
    archive_path = os.path.join(workdir, "archive.json")
    bad_code = 'import os\n\ndef compute_reward(state):\n    return 1.0\n'
    archive = RewardArchive(archive_path)
    archive.add_entry(
        reward_code=bad_code,
        metrics={"mean_speed": 20.0, "crash_rate": 0.0},
        critique="",
    )
    entry = archive.get_by_generation(0)
    assert entry is not None
    assert not validate_reward_code(bad_code)[0]

    with pytest.raises(ValueError, match="AST validation"):
        _write_validated_archive_reward(entry, 0)


def test_write_validated_archive_reward_accepts_safe_code():
    code = _passing_reward_code()
    workdir = tempfile.mkdtemp()
    archive_path = os.path.join(workdir, "archive.json")
    archive = RewardArchive(archive_path)
    archive.add_entry(
        reward_code=code,
        metrics={"mean_speed": 20.0, "crash_rate": 0.0},
        critique="",
    )
    entry = archive.get_by_generation(0)
    path = _write_validated_archive_reward(entry, 0)
    try:
        with open(path, encoding="utf-8") as f:
            written = f.read()
        assert "compute_reward" in written
    finally:
        os.remove(path)
