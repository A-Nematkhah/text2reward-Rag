"""Smoke-test gates, sandbox timeouts, and trajectory-bank calibration.

Stage A uses ``SMOKE_TEST_TIMEOUT_SEC``; Stage B pairwise checks use
``BANK_*`` thresholds against ``TRAJECTORY_REF_FITNESS_VERSION`` ground truth.
Constants only — no side effects on import.
"""

from __future__ import annotations

import os

# Per-step shaped reward execution in LLMRewardWrapper.
REWARD_STEP_TIMEOUT_SEC = 0.05

# Default execute_reward() wall-clock limit (sandbox).
SANDBOX_EXECUTE_TIMEOUT_SEC = 0.1

# Stage A smoke-test scenarios (validation.py).
SMOKE_TEST_TIMEOUT_SEC = 0.5

# Gate 1b: collided-state reward must be <= this value (prompts should match).
SMOKE_COLLISION_SEVERITY_MAX = -40.0

# Stage B trajectory-bank pairwise consistency gate (default / refine).
# Raised from 12% → 13% after calibration (see scripts/calibrate_smoke_gate.py).
BANK_MAX_VIOLATION_RATE = 0.13
# Looser Stage B thresholds during early curriculum (survive/speed).
BANK_MAX_VIOLATION_RATE_BY_PHASE: dict[str, float] = {
    "survive": 0.18,
    "speed": 0.15,
    "overtake": 0.13,
    "refine": 0.13,
}
BANK_MIN_FITNESS_GAP = 0.06

# Pinned reference fitness version for stable Stage B ground truth.
TRAJECTORY_REF_FITNESS_VERSION = 7

# Fixed seed for reproducible synthetic trajectories.
TRAJECTORY_BANK_SEED = 20260620

# Stage B bank size for evolution smoke gate: "lite" (~16 trajectories) or "full" (~40).
# Env var TRAJECTORY_BANK_MODE overrides; lite reduces redundant pairwise false rejects.
_TRAJECTORY_BANK_MODE_RAW = os.environ.get("TRAJECTORY_BANK_MODE", "lite").strip().lower()
TRAJECTORY_BANK_MODE = _TRAJECTORY_BANK_MODE_RAW if _TRAJECTORY_BANK_MODE_RAW in ("lite", "full") else "lite"

# Sandbox DoS limits (AST validation).
MAX_REWARD_SOURCE_CHARS = 16_384
MAX_REWARD_AST_NODES = 2_500
MAX_REWARD_STRING_LITERAL_CHARS = 256


def bank_max_violation_rate_for_phase(phase: str | None) -> float:
    """Stage B soft-violation ceiling; looser during survive/speed curriculum."""
    if phase and phase in BANK_MAX_VIOLATION_RATE_BY_PHASE:
        return BANK_MAX_VIOLATION_RATE_BY_PHASE[phase]
    return BANK_MAX_VIOLATION_RATE
