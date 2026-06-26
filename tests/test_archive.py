"""Reward archive persistence, enrichment, and retrieval quality."""

from txt2reward.archive.archive import (
    RewardArchive,
    compute_fitness,
    compute_fitness_v7,
    dedupe_entries_by_code,
    effective_fitness,
    is_crash_farming,
    is_pathological_for_retrieval,
)

from tests.helpers import archive_entry, base_metrics

# ── Persistence & enrichment ──────────────────────────────────────────────────


def test_archive_insert_and_persist(tmp_path):
    path = tmp_path / "archive.json"
    archive = RewardArchive(str(path))
    archive.add_entry(
        "def compute_reward(state):\n    return 1.0\n",
        {"crash_rate": 0.1, "mean_speed": 26.0, "mean_overtakes": 1.5},
    )
    assert path.exists()
    reloaded = RewardArchive(str(path))
    assert len(reloaded.entries) == 1
    assert reloaded.entries[0]["fitness_version"] == 8


def test_archive_entry_stores_enriched_metrics(tmp_path):
    path = tmp_path / "archive.json"
    archive = RewardArchive(str(path))
    entry = archive.add_entry(
        "def compute_reward(state):\n    return 1.0\n",
        {
            "crash_rate": 0.08,
            "mean_speed": 27.0,
            "mean_overtakes": 2.0,
            "total_lane_changes": 10,
            "total_overtakes": 8,
            "n_episodes": 10,
        },
    )
    m = entry["metrics"]
    assert "curriculum_phase" in m
    assert "safe_overtake_ratio" in m
    assert entry["fitness"] == compute_fitness(m)


def test_archive_summary_uses_effective_fitness(tmp_path):
    path = tmp_path / "archive.json"
    archive = RewardArchive(str(path))
    archive.add_entry(
        "def compute_reward(state):\n    return 1.0\n",
        {"crash_rate": 0.05, "mean_speed": 27.0, "mean_overtakes": 2.0},
    )
    summary = archive.summary()
    assert "best fitness=" in summary
    assert str(round(effective_fitness(archive.entries[0]), 4)) in summary


# ── Retrieval, dedup, rescoring ───────────────────────────────────────────────


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
    entry = archive_entry(0, "def compute_reward(state):\n    return -30.0\n", metrics, 0.01, fitness_version=7)
    assert entry["fitness"] == 0.01
    assert compute_fitness_v7(metrics) == 0.01
    rescored = effective_fitness(entry)
    assert rescored > 0.01
    assert rescored < 0.15


def test_top_k_dedupes_identical_code(tmp_path):
    archive = RewardArchive(str(tmp_path / "archive.json"))
    code = "def compute_reward(state):\n    return 1.0\n"
    for i in range(4):
        archive.entries.append(archive_entry(i, code, base_metrics(crash_rate=0.1 + i * 0.05), 0.5 + i * 0.05))
    assert len(archive.get_top_k(3)) == 1


def test_top_k_excludes_crash_farming_when_safer_alternative_exists(tmp_path):
    archive = RewardArchive(str(tmp_path / "archive.json"))
    archive.entries = [
        archive_entry(0, "def compute_reward(state):\n    return 1.0\n", base_metrics(crash_rate=0.15), 0.55),
        archive_entry(
            1,
            "def compute_reward(state):\n    return 2.0\n",
            base_metrics(crash_rate=1.0, mean_speed=29.0, mean_overtakes=1.0),
            0.58,
        ),
    ]
    top = archive.get_top_k(2)
    assert len(top) == 1
    assert top[0]["generation"] == 0
    assert is_crash_farming(archive.entries[1]["metrics"])


def test_top_k_spreads_crash_bands_when_possible(tmp_path):
    archive = RewardArchive(str(tmp_path / "archive.json"))
    archive.entries = [
        archive_entry(
            0,
            "def compute_reward(state):\n    return 1.0\n",
            {"crash_rate": 0.05, "mean_speed": 27.0, "mean_overtakes": 2.0},
            0.95,
        ),
        archive_entry(
            1,
            "def compute_reward(state):\n    return 2.0\n",
            {"crash_rate": 0.25, "mean_speed": 26.0, "mean_overtakes": 1.5},
            0.70,
        ),
        archive_entry(
            2,
            "def compute_reward(state):\n    return 3.0\n",
            {"crash_rate": 0.55, "mean_speed": 24.0, "mean_overtakes": 1.0},
            0.40,
        ),
    ]
    top = archive.get_top_k(3)
    bands = {round(e["metrics"]["crash_rate"], 2) for e in top}
    assert len(top) == 3
    assert len(bands) >= 2


def test_get_failed_rewards_includes_crash_farming(tmp_path):
    archive = RewardArchive(str(tmp_path / "archive.json"))
    archive.entries = [
        archive_entry(
            0,
            "def compute_reward(state):\n    return 1.0\n",
            base_metrics(crash_rate=1.0, mean_speed=29.0, mean_overtakes=1.0),
            0.12,
        ),
        archive_entry(
            1,
            "def compute_reward(state):\n    return 2.0\n",
            {"crash_rate": 0.05, "mean_speed": 27.0, "mean_overtakes": 2.0},
            0.90,
        ),
    ]
    failed = archive.get_failed_rewards(k=3, max_fitness=0.08)
    assert 0 in {e["generation"] for e in failed}


def test_get_failed_rewards_dedupes_by_code(tmp_path):
    archive = RewardArchive(str(tmp_path / "archive.json"))
    code = "def compute_reward(state):\n    return 1.0\n"
    archive.entries = [
        archive_entry(0, code, {"crash_rate": 0.5, "mean_speed": 20.0, "mean_overtakes": 0.0}, 0.05),
        archive_entry(1, code, {"crash_rate": 0.6, "mean_speed": 19.0, "mean_overtakes": 0.0}, 0.04),
    ]
    assert len(archive.get_failed_rewards(k=3, max_fitness=0.50)) == 1


def test_dedupe_entries_by_code():
    entries = [
        archive_entry(0, "def compute_reward(state):\n    return 1.0\n", {}, 0.1),
        archive_entry(1, "def compute_reward(state):\n    return 1.0\n", {}, 0.2),
    ]
    assert len(dedupe_entries_by_code(entries)) == 1


def test_is_pathological_flags_crash_farm_and_stationary():
    assert is_pathological_for_retrieval(archive_entry(0, "x", {"crash_rate": 1.0, "mean_speed": 29.0}, 0.1))
    assert is_pathological_for_retrieval(archive_entry(0, "x", {"crash_rate": 0.0, "mean_speed": 2.0}, 0.1))
    assert not is_pathological_for_retrieval(
        archive_entry(0, "x", {"crash_rate": 0.1, "mean_speed": 26.0, "mean_overtakes": 2.0}, 0.8)
    )


def test_effective_fitness_uses_current_metrics_not_stale_score(tmp_path):
    entry = archive_entry(
        0,
        "def compute_reward(state):\n    return 1.0\n",
        base_metrics(crash_rate=1.0, mean_speed=29.0),
        0.99,
    )
    assert effective_fitness(entry) < 0.2
