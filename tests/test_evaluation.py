"""Evaluation pipeline: TTC pooling and validated archive reward restore."""

import os
import tempfile

import pytest
from txt2reward.archive.archive import RewardArchive
from txt2reward.core.metrics import pool_ttc_p10_min
from txt2reward.llm.validation import write_validated_reward_tempfile
from txt2reward.sandbox.sandbox import validate_reward_code

from tests.helpers import passing_reward_code


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
