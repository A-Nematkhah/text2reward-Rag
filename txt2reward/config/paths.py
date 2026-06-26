"""Default filesystem paths for runtime artifacts.

Paths are relative to the process working directory unless noted.
Constants only — no side effects on import.

Exports:
    REWARD_PROGRAM_PATH, ARCHIVE_FILE, LOG_FILE, WEIGHTS_FILE, API_KEYS_FILE
"""

from __future__ import annotations

from pathlib import Path

# Hot-reloaded reward program (repo root by default).
REWARD_PROGRAM_PATH = "reward_program.py"

# Evolution archive and training telemetry.
ARCHIVE_FILE = "reward_archive.json"
LOG_FILE = "training_log.json"

# Legacy weight-based reward (unused by Text-to-Reward loop).
WEIGHTS_FILE = "reward_weights.json"

# Groq API key pool at repo root (optional; env var GROQ_API_KEY is also supported).
API_KEYS_FILE = Path(__file__).resolve().parents[2] / "api_keys.json"
