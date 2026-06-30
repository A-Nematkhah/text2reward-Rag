"""Unit tests for Groq / OpenRouter key rotation and provider selection."""

from __future__ import annotations

import importlib
import json
from unittest.mock import Mock, patch

import groq
import pytest
import txt2reward.config.llm as llm_config
import txt2reward.llm.key_manager as key_manager
from txt2reward.llm.key_manager import (
    _KeyManager,
    _load_keys,
    _min_cooldown_wait,
    call_with_rotation,
)


@pytest.fixture(autouse=True)
def _reset_singleton_and_keys(monkeypatch, tmp_path):
    """Isolate each test from real api_keys.json and the module-level singleton."""
    key_manager._manager = None
    monkeypatch.setattr(key_manager, "_KEYS_FILE", tmp_path / "api_keys.json")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    yield
    key_manager._manager = None


def _write_keys_file(path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _mock_client_factory(responses: list):
    """Return a side_effect for chat.completions.create."""
    state = {"i": 0}

    def _create(**_kwargs):
        action = responses[state["i"]]
        state["i"] += 1
        if isinstance(action, Exception):
            raise action
        return action

    client = Mock()
    client.chat.completions.create = Mock(side_effect=_create)
    return client


def _groq_rate_limit(retry_after: str = "5") -> groq.RateLimitError:
    response = Mock()
    response.headers = {"retry-after": retry_after}
    return groq.RateLimitError("rate limited", response=response, body=None)


def _groq_auth_error() -> groq.AuthenticationError:
    response = Mock()
    response.headers = {}
    return groq.AuthenticationError("invalid key", response=response, body=None)


class _OpenRouterRateLimit(Exception):
    status_code = 429


class _OpenRouterAuthError(Exception):
    status_code = 401


# ── Key loading ───────────────────────────────────────────────────────────────


def test_load_keys_prefers_env_then_json(tmp_path, monkeypatch):
    keys_file = tmp_path / "api_keys.json"
    monkeypatch.setattr(key_manager, "_KEYS_FILE", keys_file)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_from_env")
    _write_keys_file(keys_file, {"groq_keys": ["gsk_from_file", "gsk_your_placeholder"]})

    keys = _load_keys("groq")
    assert keys == ["gsk_from_env", "gsk_from_file"]


def test_load_keys_openrouter_missing_raises(monkeypatch):
    with pytest.raises(EnvironmentError, match="No OpenRouter API Key"):
        _load_keys("openrouter")


def test_unknown_provider_defaults_to_groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_only")
    mgr = _KeyManager(provider="not-a-real-provider")
    assert mgr.provider == "groq"


# ── Groq rotation ─────────────────────────────────────────────────────────────


def test_groq_rotates_to_second_key_on_rate_limit(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_key_a")
    _write_keys_file(key_manager._KEYS_FILE, {"groq_keys": ["gsk_key_b"]})

    mgr = _KeyManager(provider="groq")
    clients = {
        "gsk_key_a": _mock_client_factory([_groq_rate_limit(), "ok-a"]),
        "gsk_key_b": _mock_client_factory(["ok-b"]),
    }

    with patch.object(mgr, "_get_client", side_effect=lambda k: clients[k]):
        result = mgr.call(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": "hi"}])

    assert result == "ok-b"
    assert clients["gsk_key_a"].chat.completions.create.call_count == 1
    assert clients["gsk_key_b"].chat.completions.create.call_count == 1


def test_groq_raises_when_all_keys_are_invalid(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_bad_a")
    _write_keys_file(key_manager._KEYS_FILE, {"groq_keys": ["gsk_bad_b"]})

    mgr = _KeyManager(provider="groq")
    clients = {
        "gsk_bad_a": _mock_client_factory([_groq_auth_error()]),
        "gsk_bad_b": _mock_client_factory([_groq_auth_error()]),
    }

    with patch.object(mgr, "_get_client", side_effect=lambda k: clients[k]):
        with pytest.raises(RuntimeError, match="No valid keys remaining"):
            mgr.call(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": "hi"}])


def test_groq_propagates_non_auth_non_rate_errors(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_key")

    mgr = _KeyManager(provider="groq")
    client = _mock_client_factory([RuntimeError("network down")])

    with patch.object(mgr, "_get_client", return_value=client):
        with pytest.raises(RuntimeError, match="network down"):
            mgr.call(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": "hi"}])


# ── OpenRouter rotation ───────────────────────────────────────────────────────


def test_openrouter_rotates_on_429(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-a")
    _write_keys_file(key_manager._KEYS_FILE, {"openrouter_keys": ["sk-or-b"]})

    mgr = _KeyManager(provider="openrouter")
    clients = {
        "sk-or-a": _mock_client_factory([_OpenRouterRateLimit(), "ok-a"]),
        "sk-or-b": _mock_client_factory(["ok-b"]),
    }

    with patch.object(mgr, "_get_client", side_effect=lambda k: clients[k]):
        result = mgr.call(model="meta-llama/llama-3.3-70b-instruct:free", messages=[{"role": "user", "content": "hi"}])

    assert result == "ok-b"


def test_openrouter_openai_client_uses_base_url(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    mgr = _KeyManager(provider="openrouter")

    with patch("txt2reward.llm.key_manager.OpenAI") as openai_cls:
        openai_cls.return_value = Mock()
        mgr._ensure_init()
        mgr.get_active_client()

    openai_cls.assert_called_once()
    _, kwargs = openai_cls.call_args
    assert kwargs["api_key"] == "sk-or-test"
    assert kwargs["base_url"] == llm_config.OPENROUTER_BASE_URL


# ── Singleton / provider switching ───────────────────────────────────────────


def test_get_manager_rebuilds_when_provider_changes(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_one")
    monkeypatch.setattr(key_manager, "LLM_PROVIDER", "groq")
    key_manager._manager = None

    mgr_groq = key_manager._get_manager()
    assert mgr_groq.provider == "groq"

    monkeypatch.setattr(key_manager, "LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-one")

    mgr_or = key_manager._get_manager()
    assert mgr_or.provider == "openrouter"
    assert mgr_or is not mgr_groq


def test_call_with_rotation_delegates_to_manager(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_delegate")
    mock_mgr = Mock()
    mock_mgr.call.return_value = "delegated"

    with patch.object(key_manager, "_get_manager", return_value=mock_mgr):
        out = call_with_rotation(model="m", messages=[{"role": "user", "content": "x"}])

    assert out == "delegated"
    mock_mgr.call.assert_called_once()


# ── Config / model selection ──────────────────────────────────────────────────


def test_llm_model_switches_with_provider_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    importlib.reload(llm_config)
    try:
        assert llm_config.LLM_PROVIDER == "openrouter"
        assert llm_config.LLM_MODEL == llm_config.OPENROUTER_MODEL
        assert llm_config.LLM_MODEL != llm_config.GROQ_MODEL
    finally:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
        importlib.reload(llm_config)


def test_min_cooldown_wait_picks_earliest_expiry():
    now = 100.0
    keys = ["a", "b"]
    cooldown = {"a": now + 30.0, "b": now + 10.0}
    with patch("txt2reward.llm.key_manager.time.time", return_value=now):
        assert _min_cooldown_wait(cooldown, keys) == pytest.approx(10.0)
