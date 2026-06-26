"""Task 15 — behavioral regression guards for the v2 package layout.

Locks stable outputs for fitness, curriculum, config defaults, and critical
import/wiring paths so refactors cannot silently change PPO/evolution behavior.
"""

from __future__ import annotations

import importlib
import json
import tempfile

import pytest
from txt2reward.archive.archive import RewardArchive, compute_fitness, compute_fitness_v8, enrich_fitness_metrics
from txt2reward.config.fitness import FITNESS_VERSION_DEFAULT
from txt2reward.config.training import (
    DEFAULT_EVOLVE_EVERY,
    DEFAULT_WARMUP_EPISODES,
    PPO_BATCH_SIZE,
    PPO_N_EPOCHS,
    PPO_N_STEPS,
)
from txt2reward.config.validation import BANK_MAX_VIOLATION_RATE, TRAJECTORY_REF_FITNESS_VERSION
from txt2reward.llm.designer import RewardDesigner
from txt2reward.llm.validation import _full_validation_pipeline
from txt2reward.sandbox.sandbox import validate_reward_code

from tests.helpers import base_metrics, passing_reward_code

# ── Config defaults (PPO + evolution schedule) ────────────────────────────────


def test_ppo_and_evolution_defaults_unchanged():
    assert DEFAULT_WARMUP_EPISODES == 40
    assert DEFAULT_EVOLVE_EVERY == 20
    assert PPO_N_STEPS == 512
    assert PPO_BATCH_SIZE == 64
    assert PPO_N_EPOCHS == 5
    assert FITNESS_VERSION_DEFAULT == 8
    assert TRAJECTORY_REF_FITNESS_VERSION == 7
    assert BANK_MAX_VIOLATION_RATE == pytest.approx(0.13)


# ── Fitness + curriculum (archive logic) ──────────────────────────────────────


def test_fitness_golden_value_on_canonical_metrics():
    m = enrich_fitness_metrics(base_metrics())
    assert m["curriculum_phase"] == "refine"
    assert compute_fitness(m) == pytest.approx(0.9806, rel=1e-4)
    assert compute_fitness_v8(m) == pytest.approx(0.9806, rel=1e-4)


def test_archive_roundtrip_preserves_fitness_and_schema(tmp_path):
    path = tmp_path / "archive.json"
    archive = RewardArchive(str(path))
    metrics = enrich_fitness_metrics(base_metrics())
    code = passing_reward_code()
    entry = archive.add_entry(code, metrics, critique="ok")
    assert entry["generation"] == 0
    assert entry["fitness_version"] == 8
    assert entry["fitness"] == pytest.approx(compute_fitness(metrics), rel=1e-6)

    reloaded = RewardArchive(str(path))
    blob = json.loads(path.read_text(encoding="utf-8"))
    assert "entries" in blob
    assert len(reloaded.entries) == 1
    assert reloaded.entries[0]["reward_code"] == code


# ── Validation + evaluation pipeline ─────────────────────────────────────────


def test_bootstrap_validation_pipeline_still_passes():
    ok, err, _ = _full_validation_pipeline(passing_reward_code())
    assert ok, err
    assert validate_reward_code(passing_reward_code())[0]


def test_legacy_components_module_importable():
    mod = importlib.import_module("txt2reward.reward.components")
    assert callable(mod.compute_shaped_reward)
    assert callable(mod.load_weights)


# ── Evolution orchestration wiring ────────────────────────────────────────────


def test_designer_aggregate_metrics_matches_archive_enrichment():
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
    enriched = enrich_fitness_metrics(metrics)
    assert metrics["curriculum_phase"] == enriched["curriculum_phase"]
    assert compute_fitness(metrics) == pytest.approx(compute_fitness(enriched), rel=1e-6)


def test_evolve_archives_before_llm_generation(monkeypatch):
    workdir = tempfile.mkdtemp()
    archive_path = f"{workdir}/reward_archive.json"
    reward_path = f"{workdir}/reward_program.py"
    with open(reward_path, "w", encoding="utf-8") as f:
        f.write("def compute_reward(state):\n    return 1.0\n")

    designer = RewardDesigner(archive_path=archive_path, reward_path=reward_path, verbose=False)
    monkeypatch.setattr(designer, "_call_generate_with_repair", lambda *a, **k: None)
    monkeypatch.setattr(designer, "_call_critique", lambda *a, **k: "")
    designer._episode_stats = [
        {
            "mean_speed": 25.0,
            "collisions": 1,
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
    assert designer.archive.entries[0]["generation"] == 0


# ── Package import smoke (refactor wiring) ────────────────────────────────────


@pytest.mark.parametrize(
    "module",
    [
        "txt2reward.archive.archive",
        "txt2reward.llm.designer",
        "txt2reward.llm.validation",
        "txt2reward.sandbox.sandbox",
        "txt2reward.training.train",
        "txt2reward.training.callbacks",
        "txt2reward.evaluation.evaluate",
        "txt2reward.reward.wrapper",
        "txt2reward.trajectory.bank",
    ],
)
def test_critical_modules_import(module: str):
    importlib.import_module(module)
