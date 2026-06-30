"""
key_manager.py — Smart API Key Manager for Groq / OpenRouter
--------------------------------------------------------------
Put multiple free keys in ``api_keys.json`` at the repository root, under
either ``groq_keys`` or ``openrouter_keys`` (or both — see
``api_keys.json.example``). Whenever a rate limit occurs, it automatically
switches to the next key for the active provider.

Provider selection is controlled by ``txt2reward.config.llm.LLM_PROVIDER``
(or the ``LLM_PROVIDER`` env var: "groq" | "openrouter").

Usage:
    from txt2reward.llm.key_manager import get_llm_client, call_with_rotation

    client = get_llm_client()   # current active key, current provider
    resp = call_with_rotation(
        model="llama-3.3-70b-versatile",
        messages=[...],
        temperature=0.5,
        max_tokens=800,
    )
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import groq
from groq import Groq

try:
    import openai
    from openai import OpenAI
except ImportError:  # OpenRouter path is optional until needed
    openai = None
    OpenAI = None  # type: ignore[misc]

from txt2reward.config.llm import (
    GENERATION_MAX_TOKENS,
    GENERATION_TEMPERATURE,
    KEY_ROTATION_MAX_ROUNDS,
    KEY_ROTATION_WAIT_SEC,
    LLM_PROVIDER,
    OPENROUTER_BASE_URL,
)
from txt2reward.config.paths import API_KEYS_FILE
from txt2reward.core.log import get_logger

log = get_logger("key_manager")

# ── Settings ──────────────────────────────────────────────────────────────────
_KEYS_FILE = API_KEYS_FILE

# Wait time (seconds) before retrying after all keys are exhausted
_WAIT_ALL_EXHAUSTED = KEY_ROTATION_WAIT_SEC

# Maximum retry rounds after all keys hit rate limits
_MAX_FULL_ROTATIONS = KEY_ROTATION_MAX_ROUNDS

# Env var names per provider — keep both supported regardless of which
# provider is active, so switching providers doesn't require touching env.
_ENV_VAR_BY_PROVIDER = {
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
_JSON_FIELD_BY_PROVIDER = {
    "groq": "groq_keys",
    "openrouter": "openrouter_keys",
}


# ── Load Keys ────────────────────────────────────────────────────────────────
def _load_keys(provider: str) -> list[str]:
    """Loads keys for ``provider`` from env var or api_keys.json."""
    keys: list[str] = []

    env_var = _ENV_VAR_BY_PROVIDER.get(provider, "GROQ_API_KEY")
    env_key = os.environ.get(env_var, "").strip()
    if env_key:
        keys.append(env_key)

    if _KEYS_FILE.exists():
        try:
            data = json.loads(_KEYS_FILE.read_text(encoding="utf-8"))
            field = _JSON_FIELD_BY_PROVIDER.get(provider, "groq_keys")
            file_keys = data.get(field, [])
            for k in file_keys:
                k = k.strip()
                if k and k not in keys and not k.lower().startswith(("gsk_your", "sk-or-your", "your_")):
                    keys.append(k)
        except Exception as exc:
            log.warning(f"[key_manager] Warning: Failed to read {_KEYS_FILE}: {exc}")

    if not keys:
        if provider == "openrouter":
            raise EnvironmentError(
                "\n[ERROR] No OpenRouter API Key found!\n"
                "Use one of the following methods:\n"
                "  1) Add keys to api_keys.json under 'openrouter_keys'\n"
                "  2) export OPENROUTER_API_KEY=sk-or-xxxxxxxx\n"
                "Get a free key from: https://openrouter.ai/\n"
            )
        raise EnvironmentError(
            "\n[ERROR] No Groq API Key found!\n"
            "Use one of the following methods:\n"
            "  1) Add keys to api_keys.json under 'groq_keys'\n"
            "  2) export GROQ_API_KEY=gsk_xxxxxxxx\n"
            "Get a free key from: https://console.groq.com\n"
        )

    return keys


# ── Key Manager (Singleton, provider-aware) ───────────────────────────────────
class _KeyManager:
    """Rotating API Key manager with Rate Limit detection, per provider."""

    def __init__(self, provider: str | None = None) -> None:
        self.provider = (provider or LLM_PROVIDER or "groq").strip().lower()
        if self.provider not in ("groq", "openrouter"):
            log.warning(f"[key_manager] Unknown LLM_PROVIDER '{self.provider}' — defaulting to 'groq'")
            self.provider = "groq"
        self._keys: list[str] = []
        self._index: int = 0  # current active key
        self._clients: dict[str, Any] = {}
        self._cooldown_until: dict[str, float] = {}  # key → cooldown expiration time
        self._initialized = False

    def _ensure_init(self) -> None:
        if not self._initialized:
            self._keys = _load_keys(self.provider)
            self._index = 0
            self._initialized = True
            log.info(f"[key_manager] provider={self.provider} | {len(self._keys)} keys loaded.")

    def _get_client(self, key: str) -> Any:
        if key not in self._clients:
            if self.provider == "openrouter":
                if OpenAI is None:
                    raise EnvironmentError(
                        "OpenRouter provider selected but the 'openai' package is not installed. "
                        "Run: pip install openai"
                    )
                self._clients[key] = OpenAI(
                    api_key=key,
                    base_url=OPENROUTER_BASE_URL,
                    default_headers={
                        # Optional but recommended by OpenRouter for routing/analytics.
                        "HTTP-Referer": "https://github.com/A-Nematkhah/text2reward-Rag",
                        "X-Title": "txt2reward-v2",
                    },
                )
            else:
                self._clients[key] = Groq(api_key=key)
        return self._clients[key]

    def _current_key(self) -> str:
        self._ensure_init()
        return self._keys[self._index]

    def _rotate(self) -> bool:
        """Switches to the next key without cooldown; returns False if all keys are in cooldown."""
        now = time.time()
        n = len(self._keys)
        for _ in range(n):
            self._index = (self._index + 1) % n
            key = self._keys[self._index]
            if self._cooldown_until.get(key, 0) <= now:
                log.info(f"[key_manager] Switched to key #{self._index + 1}")
                return True
        return False  # all keys are in cooldown

    def get_active_client(self) -> Any:
        self._ensure_init()
        key = self._current_key()
        return self._get_client(key)

    # ── Provider-specific error classification ────────────────────────────────

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        if self.provider == "groq":
            return isinstance(exc, groq.RateLimitError)
        if openai is not None and isinstance(exc, openai.RateLimitError):
            return True
        # OpenRouter sometimes surfaces rate limits as generic 429 API errors.
        status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
        return status == 429

    def _is_auth_error(self, exc: Exception) -> bool:
        if self.provider == "groq":
            return isinstance(exc, groq.AuthenticationError)
        if openai is not None and isinstance(exc, openai.AuthenticationError):
            return True
        status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
        return status == 401

    def _retry_after(self, exc: Exception) -> float | None:
        try:
            headers = getattr(getattr(exc, "response", None), "headers", None)
            if not headers:
                return None
            val = headers.get("retry-after") or headers.get("x-ratelimit-reset-requests")
            if val:
                return float(val)
        except Exception:
            pass
        return None

    def call(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.5,
        max_tokens: int = 800,
        **kwargs: Any,
    ) -> Any:
        """
        Makes an API call with automatic key rotation on Rate Limit.

        If all keys hit Rate Limit, waits _WAIT_ALL_EXHAUSTED seconds
        and retries (up to _MAX_FULL_ROTATIONS rounds). Works identically
        for Groq and OpenRouter — both clients expose the same
        ``chat.completions.create(...)`` interface.
        """
        self._ensure_init()

        for rotation in range(_MAX_FULL_ROTATIONS):
            n = len(self._keys)

            # One full pass through all keys
            for _ in range(n):
                key = self._current_key()

                # If this key is still in cooldown, try the next one
                cooldown_left = self._cooldown_until.get(key, 0) - time.time()
                if cooldown_left > 0:
                    if not self._rotate():
                        break  # all keys are in cooldown
                    continue

                try:
                    client = self._get_client(key)
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                    return resp  # ✅ success

                except Exception as exc:
                    if self._is_rate_limit_error(exc):
                        wait = self._retry_after(exc) or 60
                        self._cooldown_until[key] = time.time() + wait
                        log.warning(
                            f"[key_manager] Key #{self._index + 1} ({self.provider}) hit rate limit "
                            f"(wait {wait:.0f}s). Switching..."
                        )
                        if not self._rotate():
                            break  # all keys in cooldown
                        continue

                    if self._is_auth_error(exc):
                        log.warning(f"[key_manager] Key #{self._index + 1} ({self.provider}) is invalid. Switching...")
                        self._cooldown_until[key] = float("inf")  # never use this key again
                        if not self._rotate():
                            raise RuntimeError("No valid keys remaining!") from exc
                        continue

                    raise  # propagate other errors

            # Reaching here means all keys hit rate limits
            if rotation < _MAX_FULL_ROTATIONS - 1:
                min_wait = _min_cooldown_wait(self._cooldown_until, self._keys)
                log.warning(
                    f"[key_manager] All {n} keys ({self.provider}) are rate-limited. "
                    f"Waiting {min_wait:.0f}s ... "
                    f"(round {rotation + 2}/{_MAX_FULL_ROTATIONS})"
                )
                _interruptible_sleep(min_wait)

                now = time.time()
                for i, k in enumerate(self._keys):
                    if self._cooldown_until.get(k, 0) <= now:
                        self._index = i
                        break

        raise RuntimeError(f"[key_manager] No available key after {_MAX_FULL_ROTATIONS} rotation rounds.")


def _interruptible_sleep(seconds: float) -> None:
    """Sleep in 1s chunks so KeyboardInterrupt is not blocked for minutes."""
    end = time.time() + max(0.0, seconds)
    while True:
        remaining = end - time.time()
        if remaining <= 0:
            return
        time.sleep(min(1.0, remaining))


def _min_cooldown_wait(cooldown_until: dict[str, float], keys: list[str]) -> float:
    """Returns the minimum wait time until a key becomes available."""
    now = time.time()
    waits = [max(0.0, cooldown_until.get(k, 0) - now) for k in keys if cooldown_until.get(k, 0) > now]
    return min(waits) if waits else 1.0


# ── Public Singleton (provider-aware, lazily (re)created if provider switches) ─
_manager: _KeyManager | None = None


def _get_manager() -> _KeyManager:
    global _manager
    active_provider = (LLM_PROVIDER or "groq").strip().lower()
    if _manager is None or _manager.provider != active_provider:
        _manager = _KeyManager(provider=active_provider)
    return _manager


def get_llm_client() -> Any:
    """Returns the active-provider client for the currently active key."""
    return _get_manager().get_active_client()


# Backward-compatible alias (old call sites / external scripts).
def get_groq_client() -> Any:
    """Deprecated alias — returns the active client (Groq only if LLM_PROVIDER=groq)."""
    return get_llm_client()


def call_with_rotation(
    model: str,
    messages: list[dict],
    temperature: float = GENERATION_TEMPERATURE,
    max_tokens: int = GENERATION_MAX_TOKENS,
    **kwargs: Any,
) -> Any:
    """
    API call with automatic key rotation, against whichever provider is
    configured via LLM_PROVIDER. Direct replacement for
    client.chat.completions.create().
    """
    return _get_manager().call(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
