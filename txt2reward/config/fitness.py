"""Fitness scoring version and archive retrieval thresholds.

``FITNESS_VERSION_DEFAULT`` selects the active scorer; retrieval constants
gate RAG examples in ``RewardArchive.get_top_k`` / ``get_failed_rewards``.
Constants only — no side effects on import.
"""

from __future__ import annotations

# Active fitness function for new archive entries and ranking.
FITNESS_VERSION_DEFAULT = 8

# get_top_k: minimum effective fitness to appear in positive examples.
ARCHIVE_MIN_TOP_FITNESS = 0.03

# get_failed_rewards: maximum fitness for negative examples.
ARCHIVE_FAILED_MAX_FITNESS = 0.08

# is_crash_farming heuristic (fast + near-universal crashes).
CRASH_FARMING_CRASH_MIN = 0.90
CRASH_FARMING_SPEED_MIN = 26.0  # m/s

# Single source of truth for "is this agent safe enough that passive/cruising
# behaviour should be penalised" across v6 gate, v7 penalty, and v8 hard clamp.
PASSIVE_DRIVING_CRASH_CEILING = 0.30
