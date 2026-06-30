"""End-to-end integration: metrics → archive → evolution."""

from __future__ import annotations

import os
import tempfile

import pytest
from txt2reward.archive.archive import RewardArchive, compute_fitness
from txt2reward.llm.designer import RewardDesigner


class TestEndToEndPipeline:
    def test_aggregate_metrics_to_archive_to_top_k(self):
        episode_stats = [
            {
                "mean_speed": 26.0,
                "collisions": 0,
                "steps": 120,
                "total_overtakes": 2,
                "total_lane_changes": 3,
                "mean_ttc": 5.0,
                "p10_ttc": 4.0,
                "min_ttc": 3.0,
                "mean_long_jerk": 1.0,
                "mean_accel": 0.8,
            }
            for _ in range(40)
        ]
        metrics = RewardDesigner._aggregate_metrics(episode_stats)
        assert metrics["curriculum_phase"] in {"survive", "speed", "overtake", "refine"}
        assert metrics["crash_rate"] == 0.0

        workdir = tempfile.mkdtemp()
        archive = RewardArchive(os.path.join(workdir, "archive.json"))
        entry = archive.add_entry("def compute_reward(state):\n    return 1.0\n", metrics)
        assert entry["metrics"]["curriculum_phase"] == metrics["curriculum_phase"]
        assert entry["fitness"] == pytest.approx(compute_fitness(metrics), rel=1e-4)

        top = archive.get_top_k(1)
        assert len(top) == 1
        assert top[0]["generation"] == 0

    def test_evolve_archives_before_generating(self, monkeypatch):
        workdir = tempfile.mkdtemp()
        archive_path = os.path.join(workdir, "reward_archive.json")
        reward_path = os.path.join(workdir, "reward_program.py")
        with open(reward_path, "w", encoding="utf-8") as f:
            f.write("def compute_reward(state):\n    return 1.0\n")

        designer = RewardDesigner(archive_path=archive_path, reward_path=reward_path, verbose=False)
        monkeypatch.setattr(designer, "_call_generate_with_repair", lambda *a, **k: None)
        monkeypatch.setattr(designer, "_call_critique", lambda *a, **k: "")

        designer._episode_stats = [
            {
                "mean_speed": 25.0,
                "collisions": 0,
                "steps": 80,
                "total_overtakes": 1,
                "total_lane_changes": 2,
                "mean_ttc": 3.0,
                "p10_ttc": 2.0,
                "min_ttc": 1.0,
                "mean_long_jerk": 2.0,
                "mean_accel": 1.0,
            }
            for _ in range(40)
        ]
        designer._episode_count = designer.warmup_episodes
        designer._evolve()

        assert len(designer.archive.entries) == 1
        assert designer.get_last_evolution_metrics() is not None
        assert "curriculum_phase" in designer.get_last_evolution_metrics()
