"""
key_manager.py — Smart API Key Manager for Groq
--------------------------------------------------
Put multiple free keys in api_keys.json.
Whenever a rate limit occurs, it automatically switches to the next key.

Usage:
    from key_manager import get_groq_client, call_with_rotation

    client = get_groq_client()   # current active key
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
from pathlib import Path
from typing import Any

import groq
from groq import Groq

# ── Settings ──────────────────────────────────────────────────────────────────
_KEYS_FILE = Path(__file__).parent / "api_keys.json"

# Wait time (seconds) before retrying after all keys are exhausted
_WAIT_ALL_EXHAUSTED = 60

# Maximum retry rounds after all keys hit rate limits
_MAX_FULL_ROTATIONS = 3


# ── Load Keys ────────────────────────────────────────────────────────────────
def _load_keys() -> list[str]:
    """Loads keys from api_keys.json or environment variables."""
    keys: list[str] = []

    # Check env var first (if you only have one key)
    env_key = os.environ.get("GROQ_API_KEY", "").strip()
    if env_key:
        keys.append(env_key)

    # Then load from json file
    if _KEYS_FILE.exists():
        try:
            data = json.loads(_KEYS_FILE.read_text(encoding="utf-8"))
            file_keys = data.get("groq_keys", [])
            for k in file_keys:
                k = k.strip()
                if k and k not in keys and not k.startswith("gsk_YOUR"):
                    keys.append(k)
        except Exception as exc:
            print(f"[key_manager] Warning: Failed to read {_KEYS_FILE}: {exc}")

    if not keys:
        raise EnvironmentError(
            "\n[ERROR] No Groq API Key found!\n"
            "Use one of the following methods:\n"
            "  1) Add keys to api_keys.json\n"
            "  2) export GROQ_API_KEY=gsk_xxxxxxxx\n"
            "Get a free key from: https://console.groq.com\n"
        )

    return keys


# ── Key Manager (Singleton) ───────────────────────────────────────────────────
class _KeyManager:
    """Rotating API Key manager with Rate Limit detection."""

    def __init__(self) -> None:
        self._keys: list[str] = []
        self._index: int = 0  # current active key
        self._clients: dict[str, Groq] = {}
        self._cooldown_until: dict[str, float] = {}  # key → cooldown expiration time
        self._initialized = False

    def _ensure_init(self) -> None:
        if not self._initialized:
            self._keys = _load_keys()
            self._index = 0
            self._initialized = True
            print(f"[key_manager] {len(self._keys)} keys loaded.")

    def _get_client(self, key: str) -> Groq:
        if key not in self._clients:
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
                print(f"[key_manager] Switched to key #{self._index + 1}")
                return True
        return False  # all keys are in cooldown

    def get_active_client(self) -> Groq:
        self._ensure_init()
        key = self._current_key()
        return self._get_client(key)

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
        and retries (up to _MAX_FULL_ROTATIONS rounds).
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

                except groq.RateLimitError as exc:
                    # Extract wait time from response headers (if available)
                    wait = _parse_retry_after(exc) or 60
                    self._cooldown_until[key] = time.time() + wait
                    print(
                        f"[key_manager] Key #{self._index + 1} hit rate limit "
                        f"(wait {wait:.0f}s). Switching..."
                    )
                    if not self._rotate():
                        break  # all keys in cooldown

                except groq.AuthenticationError:
                    print(f"[key_manager] Key #{self._index + 1} is invalid. Switching...")
                    self._cooldown_until[key] = float("inf")  # never use this key again
                    if not self._rotate():
                        raise RuntimeError("No valid keys remaining!")

                except Exception:
                    raise  # propagate other errors

            # Reaching here means all keys hit rate limits
            if rotation < _MAX_FULL_ROTATIONS - 1:
                # Find when the next key becomes available
                min_wait = _min_cooldown_wait(self._cooldown_until, self._keys)
                print(
                    f"[key_manager] All {n} keys are rate-limited. "
                    f"Waiting {min_wait:.0f}s ... "
                    f"(round {rotation + 2}/{_MAX_FULL_ROTATIONS})"
                )
                _interruptible_sleep(min_wait)

                # After sleep, select the first available key
                now = time.time()
                for i, k in enumerate(self._keys):
                    if self._cooldown_until.get(k, 0) <= now:
                        self._index = i
                        break

        raise RuntimeError(
            f"[key_manager] No available key after {_MAX_FULL_ROTATIONS} rotation rounds."
        )


def _parse_retry_after(exc: groq.RateLimitError) -> float | None:
    """Extracts retry time from response headers (if available)."""
    try:
        headers = exc.response.headers  # type: ignore[union-attr]
        val = headers.get("retry-after") or headers.get("x-ratelimit-reset-requests")
        if val:
            return float(val)
    except Exception:
        pass
    return None


def _interruptible_sleep(seconds: float) -> None:
    """Sleep in 1s chunks so KeyboardInterrupt is not blocked for minutes."""
    end = time.time() + max(0.0, seconds)
    while True:
        remaining = end - time.time()
        if remaining <= 0:
            return
        time.sleep(min(1.0, remaining))


    """Returns the minimum wait time until a key becomes available."""
    now = time.time()
    waits = [
        max(0.0, cooldown_until.get(k, 0) - now)
        for k in keys
        if cooldown_until.get(k, 0) > now
    ]
    return min(waits) if waits else 1.0


# ── Public Singleton ──────────────────────────────────────────────────────────
_manager = _KeyManager()


def get_groq_client() -> Groq:
    """Returns the Groq client for the currently active key."""
    return _manager.get_active_client()


def call_with_rotation(
    model: str,
    messages: list[dict],
    temperature: float = 0.5,
    max_tokens: int = 800,
    **kwargs: Any,
) -> Any:
    """
    API call with automatic key rotation.
    Direct replacement for client.chat.completions.create().
    """
    return _manager.call(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
