"""LLM API and generation hyperparameters.

Groq model name, sampling parameters, and API key rotation limits.
Constants only — no side effects on import.
"""

from __future__ import annotations

# Groq chat model for reward generation and critique.
LLM_MODEL = "llama-3.3-70b-versatile"

# Reward generation (_call_generate_with_repair).
GENERATION_TEMPERATURE = 0.5
GENERATION_MAX_TOKENS = 800
GENERATION_MAX_RETRIES = 3

# Post-episode critique.
CRITIQUE_TEMPERATURE = 0.3
CRITIQUE_MAX_TOKENS = 500
CRITIQUE_MAX_RETRIES = 2

# API key rotation (txt2reward.llm.key_manager).
KEY_ROTATION_WAIT_SEC = 60
KEY_ROTATION_MAX_ROUNDS = 3
