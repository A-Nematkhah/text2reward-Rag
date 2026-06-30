"""LLM API and generation hyperparameters.

Supports two providers, selectable via ``LLM_PROVIDER`` (or the
``LLM_PROVIDER`` environment variable):

  - "groq"        : Groq API (default, original behaviour)
  - "openrouter"  : OpenRouter (OpenAI-compatible) API — useful for free models

Constants only — no side effects on import.
"""

from __future__ import annotations

import os

# ── Provider selection ────────────────────────────────────────────────────────
# "groq" or "openrouter". Env var overrides the hardcoded default so you can
# switch without editing code, e.g.:
#   export LLM_PROVIDER=openrouter
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").strip().lower()

# Groq chat model for reward generation and critique.
GROQ_MODEL = "llama-3.3-70b-versatile"

# OpenRouter chat model. Default picked from the free tier
# (https://openrouter.ai/openrouter/free) — change if you prefer another
# free (or paid) model id. Free models on OpenRouter are rate-limited and
# may be less reliable at following the strict reward-code format than
# Groq's llama-3.3-70b-versatile, so expect more repair retries.
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")

# OpenRouter API base URL (OpenAI-compatible).
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Backward-compatible alias: LLM_MODEL always reflects the active provider's
# model, so existing imports (`from txt2reward.config.llm import LLM_MODEL`)
# keep working unchanged.
LLM_MODEL = OPENROUTER_MODEL if LLM_PROVIDER == "openrouter" else GROQ_MODEL

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
