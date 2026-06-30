"""Task 6 — metrics-driven evolution curriculum."""

import os
import tempfile

import pytest
from txt2reward.archive.archive import (
    CURRICULUM_GUIDANCE,
    CURRICULUM_PHASES,
    RewardArchive,
    _curriculum_quality_weights,
    curriculum_guidance,
    enrich_fitness_metrics,
    infer_curriculum_phase,
    infer_curriculum_transition,
)
from txt2reward.llm.designer import RewardDesigner

from tests.helpers import archive_entry, base_metrics


def test_curriculum_phase_independent_of_generation():
    low_crash = {"crash_rate": 0.05, "mean_speed": 27.0, "mean_overtakes": 2.0}
    high_crash = {"crash_rate": 0.6, "mean_speed": 28.0, "mean_overtakes": 2.0}
    assert infer_curriculum_phase(low_crash) == "refine"
    assert infer_curriculum_phase(high_crash) == "survive"
    # Same metrics at gen 0 vs gen 50 must agree
    assert infer_curriculum_phase(low_crash) == infer_curriculum_phase(low_crash)


def test_curriculum_transition_detects_phase_change():
    prev = {"crash_rate": 0.5, "mean_speed": 28.0, "mean_overtakes": 1.0}
    cur = {"crash_rate": 0.08, "mean_speed": 26.0, "mean_overtakes": 0.5}
    text = infer_curriculum_transition(prev, cur)
    assert "survive → speed" in text or "survive → overtake" in text


def test_fitness_v8_weights_shift_by_curriculum_phase():
    overtake_w = _curriculum_quality_weights("overtake")
    speed_w = _curriculum_quality_weights("speed")
    assert overtake_w["overtake"] > speed_w["overtake"]


def test_enrich_fitness_metrics_sets_curriculum_phase():
    m = enrich_fitness_metrics({"crash_rate": 0.4, "mean_speed": 25.0, "mean_overtakes": 1.0})
    assert m["curriculum_phase"] == "survive"


def test_format_for_llm_includes_curriculum_section(tmp_path):
    path = tmp_path / "archive.json"
    archive = RewardArchive(str(path))
    archive.add_entry(
        "def compute_reward(state):\n    return 1.0\n",
        {"crash_rate": 0.1, "mean_speed": 26.0, "mean_overtakes": 1.5},
    )
    text = archive.format_for_llm(k=1, curriculum_phase="overtake")
    assert "CURRENT CURRICULUM PHASE: overtake" in text
    assert curriculum_guidance("overtake")[:20] in text


def test_evolve_passes_metrics_curriculum_to_generate(monkeypatch):
    workdir = tempfile.mkdtemp()
    archive_path = os.path.join(workdir, "reward_archive.json")
    reward_path = os.path.join(workdir, "reward_program.py")
    code = "def compute_reward(state):\n    return 1.0\n"
    with open(reward_path, "w", encoding="utf-8") as f:
        f.write(code)

    captured: dict = {}

    def _fake_generate(archive_context, curriculum_phase="survive", **kwargs):
        captured["curriculum_phase"] = curriculum_phase
        captured["archive_context"] = archive_context
        return None

    designer = RewardDesigner(archive_path=archive_path, reward_path=reward_path, verbose=False)
    monkeypatch.setattr(designer, "_call_generate_with_repair", _fake_generate)
    monkeypatch.setattr(designer, "_call_critique", lambda *a, **k: "")

    designer._episode_stats = [
        {
            "mean_speed": 26.0,
            "collisions": 0,
            "steps": 100,
            "total_overtakes": 0,
            "total_lane_changes": 0,
            "mean_ttc": 5.0,
            "p10_ttc": 4.0,
            "min_ttc": 3.0,
            "mean_long_jerk": 1.0,
            "mean_accel": 1.0,
        }
        for _ in range(40)
    ]
    designer._episode_count = designer.warmup_episodes
    designer._evolve()

    assert captured["curriculum_phase"] == "overtake"
    assert "CURRENT CURRICULUM PHASE: overtake" in captured["archive_context"]


@pytest.mark.parametrize("phase", CURRICULUM_PHASES)
def test_all_phases_have_guidance(phase):
    assert phase in CURRICULUM_GUIDANCE
    assert len(curriculum_guidance(phase)) > 20


def test_get_top_k_restricts_to_curriculum_phase_when_enough_entries(tmp_path):
    archive = RewardArchive(str(tmp_path / "archive.json"))
    survive_codes = [
        "def compute_reward(state):\n    return state['speed_ms']\n",
        "def compute_reward(state):\n    return state.get('speed_ms', 0.0)\n",
        "def compute_reward(state):\n    return state['speed_ms'] + state.get('lane_offset', 0.0)\n",
    ]
    for i, code in enumerate(survive_codes):
        archive.entries.append(
            archive_entry(
                i,
                code,
                {**base_metrics(crash_rate=0.45, mean_speed=20.0), "curriculum_phase": "survive"},
                0.5 + i * 0.05,
            )
        )
    archive.entries.append(
        archive_entry(
            3,
            "def compute_reward(state):\n    return state.get('mean_speed', 0.0)\n",
            {**base_metrics(crash_rate=0.05, mean_speed=28.0), "curriculum_phase": "refine"},
            0.9,
        )
    )
    top = archive.get_top_k(k=3, curriculum_phase="refine")
    assert len(top) == 3


def test_get_top_k_activates_phase_restriction_when_enough_same_phase(tmp_path):
    archive = RewardArchive(str(tmp_path / "archive.json"))
    archive.entries.append(
        archive_entry(
            0,
            "def compute_reward(state):\n    return state['speed_ms'] * 9.0\n",
            {**base_metrics(crash_rate=0.05, mean_speed=28.0), "curriculum_phase": "survive"},
            0.95,
        )
    )
    codes = [
        "def compute_reward(state):\n    return state['speed_ms']\n",
        "def compute_reward(state):\n    return state.get('speed_ms', 0.0)\n",
        "def compute_reward(state):\n    return state['speed_ms'] + state.get('lane_offset', 0.0)\n",
    ]
    for i, code in enumerate(codes, start=1):
        archive.entries.append(
            archive_entry(
                i,
                code,
                {**base_metrics(crash_rate=0.05, mean_speed=27.0), "curriculum_phase": "refine"},
                0.70 + i * 0.05,
            )
        )
    top = archive.get_top_k(k=3, curriculum_phase="refine")
    assert len(top) == 3
    assert all(e.get("metrics", {}).get("curriculum_phase") == "refine" for e in top)
    assert 0 not in {e["generation"] for e in top}


def test_format_for_llm_shows_failure_mode_recurrence(tmp_path):
    archive = RewardArchive(str(tmp_path / "archive.json"))
    archive.entries.append(
        archive_entry(
            0,
            "def compute_reward(state):\n    return state['speed_ms']\n",
            base_metrics(crash_rate=0.05, mean_speed=27.0, mean_overtakes=2.0),
            0.98,
        )
    )
    archive.entries[0]["critique_meta"] = {"failure_modes": [], "strengths": [], "summary": ""}
    for i, code in enumerate(
        [
            "def compute_reward(state):\n    return state.get('a', 0.0)\n",
            "def compute_reward(state):\n    return state.get('b', 0.0)\n",
        ],
        start=1,
    ):
        entry = archive_entry(
            i,
            code,
            base_metrics(crash_rate=1.0, mean_speed=18.0, mean_overtakes=0.0),
            0.002,
        )
        entry["critique_meta"] = {"failure_modes": ["passive_driving"], "strengths": [], "summary": ""}
        archive.entries.append(entry)
    passive_codes = [
        "def compute_reward(state):\n    return state.get('speed_ms', 0.0) * 0.1\n",
        "def compute_reward(state):\n    return state['lane_offset']\n",
        "def compute_reward(state):\n    return state['speed_ms'] + state.get('lane_offset', 0.0)\n",
        "def compute_reward(state):\n    return state.get('on_road', 1.0)\n",
        "def compute_reward(state):\n    return state['speed_ms'] - state.get('lane_offset', 0.0)\n",
        "def compute_reward(state):\n    return state.get('crashed', 0.0)\n",
        "def compute_reward(state):\n    return state['speed_ms'] * state.get('on_road', 1.0)\n",
        "def compute_reward(state):\n    return state.get('mean_speed', 0.0)\n",
    ]
    for i, code in enumerate(passive_codes, start=3):
        entry = archive_entry(
            i,
            code,
            base_metrics(crash_rate=0.08, mean_speed=18.0, mean_overtakes=0.0),
            0.12,
        )
        entry["critique_meta"] = {"failure_modes": ["passive_driving"], "strengths": [], "summary": ""}
        archive.entries.append(entry)
    text = archive.format_for_llm(k=1, current_failure_modes=["passive_driving"])
    assert "Recurrence: 'passive_driving' appeared in" in text
