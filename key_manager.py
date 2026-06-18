"""
key_manager.py — مدیریت هوشمند API Key برای Groq
--------------------------------------------------
چند کلید رایگان در api_keys.json بذار.
هرموقع rate limit خورد، خودش سوئیچ می‌کنه به کلید بعدی.

استفاده:
    from key_manager import get_groq_client, call_with_rotation

    client = get_groq_client()   # کلیه فعال فعلی
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

# ── تنظیمات ──────────────────────────────────────────────────────────────────
_KEYS_FILE = Path(__file__).parent / "api_keys.json"

# زمان انتظار (ثانیه) قبل از retry بعد از exhaustion تمام کلیدها
_WAIT_ALL_EXHAUSTED = 60

# حداکثر دفعه retry بعد از اینکه همه کلیدها rate-limit خوردن
_MAX_FULL_ROTATIONS = 3


# ── بارگذاری کلیدها ──────────────────────────────────────────────────────────
def _load_keys() -> list[str]:
    """کلیدها را از api_keys.json یا environment variable بارگذاری می‌کند."""
    keys: list[str] = []

    # اول env var رو چک کن (اگه فقط یه کلید داری)
    env_key = os.environ.get("GROQ_API_KEY", "").strip()
    if env_key:
        keys.append(env_key)

    # بعد فایل json رو بارگذاری کن
    if _KEYS_FILE.exists():
        try:
            data = json.loads(_KEYS_FILE.read_text(encoding="utf-8"))
            file_keys = data.get("groq_keys", [])
            for k in file_keys:
                k = k.strip()
                if k and k not in keys and not k.startswith("gsk_YOUR"):
                    keys.append(k)
        except Exception as exc:
            print(f"[key_manager] هشدار: خواندن {_KEYS_FILE} ناموفق بود: {exc}")

    if not keys:
        raise EnvironmentError(
            "\n[ERROR] هیچ Groq API Key ای پیدا نشد!\n"
            "یکی از این روش‌ها را انجام دهید:\n"
            "  ۱) کلیدها را در api_keys.json وارد کنید\n"
            "  ۲) export GROQ_API_KEY=gsk_xxxxxxxx\n"
            "کلید رایگان از: https://console.groq.com\n"
        )

    return keys


# ── Key Manager (Singleton) ───────────────────────────────────────────────────
class _KeyManager:
    """مدیریت چرخشی API Keyها با تشخیص Rate Limit."""

    def __init__(self) -> None:
        self._keys: list[str] = []
        self._index: int = 0          # کلید فعال فعلی
        self._clients: dict[str, Groq] = {}
        self._cooldown_until: dict[str, float] = {}  # key → زمان رفع محدودیت
        self._initialized = False

    def _ensure_init(self) -> None:
        if not self._initialized:
            self._keys = _load_keys()
            self._index = 0
            self._initialized = True
            print(f"[key_manager] {len(self._keys)} کلید بارگذاری شد.")

    def _get_client(self, key: str) -> Groq:
        if key not in self._clients:
            self._clients[key] = Groq(api_key=key)
        return self._clients[key]

    def _current_key(self) -> str:
        self._ensure_init()
        return self._keys[self._index]

    def _rotate(self) -> bool:
        """به کلید بعدی که cooldown ندارد رفته؛ اگر همه cooldown باشند False برمی‌گرداند."""
        now = time.time()
        n = len(self._keys)
        for _ in range(n):
            self._index = (self._index + 1) % n
            key = self._keys[self._index]
            if self._cooldown_until.get(key, 0) <= now:
                print(f"[key_manager] سوئیچ به کلید #{self._index + 1}")
                return True
        return False   # همه کلیدها در cooldown هستند

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
        یک API call با چرخش خودکار کلید هنگام Rate Limit.

        اگر همه کلیدها Rate Limit خوردند، _WAIT_ALL_EXHAUSTED ثانیه صبر می‌کند
        و دوباره تلاش می‌کند (تا _MAX_FULL_ROTATIONS دور).
        """
        self._ensure_init()

        for rotation in range(_MAX_FULL_ROTATIONS):
            n = len(self._keys)
            # یک دور کامل روی همه کلیدها
            for _ in range(n):
                key = self._current_key()

                # اگه این کلید هنوز cooldown داره، بعدی رو امتحان کن
                cooldown_left = self._cooldown_until.get(key, 0) - time.time()
                if cooldown_left > 0:
                    if not self._rotate():
                        break   # همه cooldown → از حلقه بیرون بزن
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
                    return resp   # ✅ موفق

                except groq.RateLimitError as exc:
                    # زمان انتظار را از هدر بگیر (اگه موجود بود)
                    wait = _parse_retry_after(exc) or 60
                    self._cooldown_until[key] = time.time() + wait
                    print(
                        f"[key_manager] کلید #{self._index + 1} rate-limit خورد "
                        f"(انتظار {wait:.0f}s). سوئیچ..."
                    )
                    if not self._rotate():
                        break   # همه cooldown

                except groq.AuthenticationError:
                    print(f"[key_manager] کلید #{self._index + 1} نامعتبر است. سوئیچ...")
                    self._cooldown_until[key] = float("inf")  # این کلید دیگه نباید استفاده بشه
                    if not self._rotate():
                        raise RuntimeError("هیچ کلید معتبری باقی نمانده!")

                except Exception:
                    raise   # بقیه خطاها رو بالا بفرست

            # اگه اینجا رسیدیم یعنی همه کلیدها rate-limit خوردند
            if rotation < _MAX_FULL_ROTATIONS - 1:
                # پیدا کن چه موقع زودترین کلید آزاد میشه
                min_wait = _min_cooldown_wait(self._cooldown_until, self._keys)
                print(
                    f"[key_manager] همه {n} کلید rate-limit. "
                    f"انتظار {min_wait:.0f}s ... (دور {rotation + 2}/{_MAX_FULL_ROTATIONS})"
                )
                time.sleep(min_wait)
                # بعد از sleep ، اولین کلید آزاد رو پیدا کن
                now = time.time()
                for i, k in enumerate(self._keys):
                    if self._cooldown_until.get(k, 0) <= now:
                        self._index = i
                        break

        raise RuntimeError(
            f"[key_manager] بعد از {_MAX_FULL_ROTATIONS} دور چرخش، "
            "هیچ کلیدی در دسترس نبود."
        )


def _parse_retry_after(exc: groq.RateLimitError) -> float | None:
    """زمان retry را از هدر پاسخ استخراج می‌کند (اگه موجود باشد)."""
    try:
        headers = exc.response.headers  # type: ignore[union-attr]
        val = headers.get("retry-after") or headers.get("x-ratelimit-reset-requests")
        if val:
            return float(val)
    except Exception:
        pass
    return None


def _min_cooldown_wait(
    cooldown_until: dict[str, float], keys: list[str]
) -> float:
    """کمترین زمان انتظار تا آزاد شدن یک کلید را برمی‌گرداند."""
    now = time.time()
    waits = [
        max(0.0, cooldown_until.get(k, 0) - now)
        for k in keys
        if cooldown_until.get(k, 0) > now
    ]
    return min(waits) if waits else 1.0


# ── Singleton عمومی ──────────────────────────────────────────────────────────
_manager = _KeyManager()


def get_groq_client() -> Groq:
    """کلاینت Groq کلید فعال را برمی‌گرداند."""
    return _manager.get_active_client()


def call_with_rotation(
    model: str,
    messages: list[dict],
    temperature: float = 0.5,
    max_tokens: int = 800,
    **kwargs: Any,
) -> Any:
    """
    API call با چرخش خودکار کلید.
    جایگزین مستقیم client.chat.completions.create().
    """
    return _manager.call(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
