from reward_archive import RewardArchive, compute_fitness, effective_fitness


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
