"""Task 5 — archive retrieval quality: dedup, pathological filter, legacy rescoring."""

from reward_archive import (
    RewardArchive,
    compute_fitness_v7,
    dedupe_entries_by_code,
    effective_fitness,
    is_crash_farming,
    is_pathological_for_retrieval,
)


def _entry(gen: int, code: str, metrics: dict, fitness: float, *, fitness_version: int = 8):
    return {
        "generation": gen,
        "reward_code": code,
        "metrics": metrics,
        "fitness": fitness,
        "fitness_version": fitness_version,
        "critique": "",
        "critique_meta": {"failure_modes": [], "strengths": [], "summary": ""},
    }


def test_effective_fitness_rescores_legacy_v7_flatline():
    metrics = {
        "mean_speed": 29.0,
        "crash_rate": 1.0,
        "mean_overtakes": 1.0,
        "mean_long_jerk": 8.0,
        "mean_ttc": 2.0,
        "p10_ttc": 1.5,
        "min_ttc": 0.8,
        "total_lane_changes": 10,
        "total_overtakes": 5,
        "n_episodes": 10,
    }
    entry = _entry(0, "def compute_reward(state):\n    return -30.0\n", metrics, 0.01, fitness_version=7)
    assert entry["fitness"] == 0.01
    assert compute_fitness_v7(metrics) == 0.01
    rescored = effective_fitness(entry)
    assert rescored > 0.01
    assert rescored < 0.15


def test_top_k_excludes_crash_farming_when_safer_alternative_exists(tmp_path):
    path = tmp_path / "archive.json"
    archive = RewardArchive(str(path))
    safe_code = "def compute_reward(state):\n    return 1.0\n"
    farm_code = "def compute_reward(state):\n    return 2.0\n"
    archive.entries = [
        _entry(
            0,
            safe_code,
            {"crash_rate": 0.15, "mean_speed": 26.0, "mean_overtakes": 2.0},
            0.55,
        ),
        _entry(
            1,
            farm_code,
            {"crash_rate": 1.0, "mean_speed": 29.0, "mean_overtakes": 1.0},
            0.58,
        ),
    ]
    top = archive.get_top_k(2)
    assert len(top) == 1
    assert top[0]["generation"] == 0
    assert is_crash_farming(archive.entries[1]["metrics"])


def test_top_k_spreads_crash_bands_when_possible(tmp_path):
    path = tmp_path / "archive.json"
    archive = RewardArchive(str(path))
    archive.entries = [
        _entry(0, "def compute_reward(state):\n    return 1.0\n", {"crash_rate": 0.05, "mean_speed": 27.0, "mean_overtakes": 2.0}, 0.95),
        _entry(1, "def compute_reward(state):\n    return 2.0\n", {"crash_rate": 0.25, "mean_speed": 26.0, "mean_overtakes": 1.5}, 0.70),
        _entry(2, "def compute_reward(state):\n    return 3.0\n", {"crash_rate": 0.55, "mean_speed": 24.0, "mean_overtakes": 1.0}, 0.40),
    ]
    top = archive.get_top_k(3)
    bands = {round(e["metrics"]["crash_rate"], 2) for e in top}
    assert len(top) == 3
    assert len(bands) >= 2


def test_get_failed_rewards_includes_crash_farming(tmp_path):
    path = tmp_path / "archive.json"
    archive = RewardArchive(str(path))
    archive.entries = [
        _entry(
            0,
            "def compute_reward(state):\n    return 1.0\n",
            {"crash_rate": 1.0, "mean_speed": 29.0, "mean_overtakes": 1.0},
            0.12,
        ),
        _entry(
            1,
            "def compute_reward(state):\n    return 2.0\n",
            {"crash_rate": 0.05, "mean_speed": 27.0, "mean_overtakes": 2.0},
            0.90,
        ),
    ]
    failed = archive.get_failed_rewards(k=3, max_fitness=0.08)
    gens = {e["generation"] for e in failed}
    assert 0 in gens


def test_get_failed_rewards_dedupes_by_code(tmp_path):
    path = tmp_path / "archive.json"
    archive = RewardArchive(str(path))
    code = "def compute_reward(state):\n    return 1.0\n"
    archive.entries = [
        _entry(0, code, {"crash_rate": 0.5, "mean_speed": 20.0, "mean_overtakes": 0.0}, 0.05),
        _entry(1, code, {"crash_rate": 0.6, "mean_speed": 19.0, "mean_overtakes": 0.0}, 0.04),
    ]
    failed = archive.get_failed_rewards(k=3, max_fitness=0.50)
    assert len(failed) == 1


def test_dedupe_entries_by_code():
    entries = [
        _entry(0, "def compute_reward(state):\n    return 1.0\n", {}, 0.1),
        _entry(1, "def compute_reward(state):\n    return 1.0\n", {}, 0.2),
    ]
    assert len(dedupe_entries_by_code(entries)) == 1


def test_is_pathological_flags_crash_farm_and_stationary():
    assert is_pathological_for_retrieval(
        _entry(0, "x", {"crash_rate": 1.0, "mean_speed": 29.0}, 0.1)
    )
    assert is_pathological_for_retrieval(
        _entry(0, "x", {"crash_rate": 0.0, "mean_speed": 2.0}, 0.1)
    )
    assert not is_pathological_for_retrieval(
        _entry(0, "x", {"crash_rate": 0.1, "mean_speed": 26.0, "mean_overtakes": 2.0}, 0.8)
    )
