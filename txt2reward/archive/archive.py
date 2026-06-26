"""Backward-compatible facade for the archive package.

Prefer focused imports:
  txt2reward.archive.store      — RewardArchive persistence
  txt2reward.archive.fitness    — fitness scoring
  txt2reward.archive.retrieval  — RAG retrieval helpers
  txt2reward.core.metrics       — episode metric aggregation
"""

from __future__ import annotations

from txt2reward.archive.critique import (
    FAILURE_MODE_TAGS,
    STRENGTH_MODE_TAGS,
    parse_structured_critique,
)
from txt2reward.archive.curriculum import (
    CURRICULUM_GUIDANCE,
    CURRICULUM_PHASES,
    curriculum_guidance,
    infer_curriculum_phase,
    infer_curriculum_transition,
)
from txt2reward.archive.fitness import (
    _curriculum_quality_weights,
    _passive_driving_gate,
    _safety_penalty_v7,
    _survival_score_v8,
    compute_fitness,
    compute_fitness_v6,
    compute_fitness_v7,
    compute_fitness_v8,
    is_passive_driving,
)
from txt2reward.archive.retrieval import (
    dedupe_entries_by_code,
    effective_fitness,
    is_crash_farming,
    is_pathological_for_retrieval,
    is_stationary_farming,
    prefetch_effective_fitness,
    reward_code_hash,
)
from txt2reward.archive.store import RewardArchive
from txt2reward.config.fitness import FITNESS_VERSION_DEFAULT
from txt2reward.core.metrics import (
    aggregate_episode_stats,
    enrich_fitness_metrics,
    near_miss_rate,
    safe_overtake_ratio,
)

__all__ = [
    "CURRICULUM_GUIDANCE",
    "CURRICULUM_PHASES",
    "FAILURE_MODE_TAGS",
    "FITNESS_VERSION_DEFAULT",
    "STRENGTH_MODE_TAGS",
    "RewardArchive",
    "_curriculum_quality_weights",
    "_passive_driving_gate",
    "_safety_penalty_v7",
    "_survival_score_v8",
    "aggregate_episode_stats",
    "compute_fitness",
    "compute_fitness_v6",
    "compute_fitness_v7",
    "compute_fitness_v8",
    "curriculum_guidance",
    "dedupe_entries_by_code",
    "effective_fitness",
    "enrich_fitness_metrics",
    "infer_curriculum_phase",
    "infer_curriculum_transition",
    "is_crash_farming",
    "is_passive_driving",
    "is_pathological_for_retrieval",
    "is_stationary_farming",
    "near_miss_rate",
    "parse_structured_critique",
    "prefetch_effective_fitness",
    "reward_code_hash",
    "safe_overtake_ratio",
]
