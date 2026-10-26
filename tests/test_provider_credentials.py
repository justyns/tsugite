import json
import stat

import pytest

from tsugite.providers import clear_cache, get_provider
from tsugite.providers.base import (
    API_KEY_CACHE_TTL_SECONDS,
    api_key_secret_name,
    invalidate_api_key_cache,
    resolve_api_key,
    resolve_binary,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_cache()
    yield
    clear_cache()


def _auth_header(provider) -> str | None:
    headers = provider._build_headers()
    return headers.get("Authorization") or headers.get("x-api-key")


def test_secret_name_is_derived_from_the_env_var():
    assert api_key_secret_name("OPENAI_API_KEY") == "openai-api-key"


@pytest.mark.parametrize(
    ("provider", "env_var", "expected"),
    [
        ("openai", "OPENAI_API_KEY", "Bearer from-secret"),
        ("openrouter", "OPENROUTER_API_KEY", "Bearer from-secret"),
        ("anthropic", "ANTHROPIC_API_KEY", "from-secret"),
    ],
)
def test_secret_overrides_env_var(secret_backend, monkeypatch, provider, env_var, expected):
    monkeypatch.setenv(env_var, "from-env")
    secret_backend.set(api_key_secret_name(env_var), "from-secret")
    assert _auth_header(get_provider(provider)) == expected


def test_env_var_applies_when_no_secret_is_stored(secret_backend, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "from-env")
    assert _auth_header(get_provider("mistral")) == "Bearer from-env"


def test_explicit_kwarg_overrides_the_secret(secret_backend):
    from tsugite.providers.anthropic import create_provider

    secret_backend.set("anthropic-api-key", "from-secret")
    assert _auth_header(create_provider(api_key="explicit")) == "explicit"


def test_key_stored_after_the_provider_is_cached_applies(secret_backend, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = get_provider("openai")
    assert _auth_header(provider) is None

    secret_backend.set("openai-api-key", "late")
    invalidate_api_key_cache()
    assert _auth_header(get_provider("openai")) == "Bearer late"

    secret_backend.delete("openai-api-key")
    invalidate_api_key_cache()
    assert _auth_header(provider) is None


def test_backend_lookups_are_cached_until_invalidated(secret_backend, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    secret_backend.set("openai-api-key", "first")
    assert resolve_api_key("OPENAI_API_KEY") == "first"

    secret_backend.set("openai-api-key", "second")
    assert resolve_api_key("OPENAI_API_KEY") == "first"
    invalidate_api_key_cache()
    assert resolve_api_key("OPENAI_API_KEY") == "second"


def test_backend_lookups_expire_after_the_ttl(secret_backend, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    now = [1000.0]
    monkeypatch.setattr("tsugite.providers.base.time.monotonic", lambda: now[0])
    secret_backend.set("openai-api-key", "first")
    assert resolve_api_key("OPENAI_API_KEY") == "first"

    secret_backend.set("openai-api-key", "second")
    now[0] += API_KEY_CACHE_TTL_SECONDS - 1
    assert resolve_api_key("OPENAI_API_KEY") == "first"
    now[0] += 1
    assert resolve_api_key("OPENAI_API_KEY") == "second"


class _BrokenBackend:
    def get(self, name):
        raise RuntimeError("keychain locked")

    def list_names(self):
        raise RuntimeError("keychain locked")


def test_env_var_applies_when_the_backend_raises(monkeypatch, caplog):
    monkeypatch.setattr("tsugite.secrets._backend", _BrokenBackend())
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert resolve_api_key("OPENAI_API_KEY") == "from-env"
    assert resolve_api_key("OPENAI_API_KEY") == "from-env"
    assert [r.getMessage() for r in caplog.records if "keychain locked" in r.getMessage()] == [
        "Secrets backend failed reading the key for OPENAI_API_KEY, using the env var: keychain locked"
    ]


def _executable(path):
    path.write_text("#!/bin/sh\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_binary_resolves_from_path_when_unconfigured(tmp_path, monkeypatch):
    _executable(tmp_path / "claude")
    monkeypatch.setenv("PATH", str(tmp_path))
    assert resolve_binary("claude_code", "claude") == str(tmp_path / "claude")


def test_configured_binary_overrides_path(tmp_path, monkeypatch, xdg_config_file):
    _executable(tmp_path / "claude")
    custom = _executable(tmp_path / "my-claude")
    monkeypatch.setenv("PATH", str(tmp_path))
    xdg_config_file.write_text(json.dumps({"plugins": {"claude_code": {"binary": str(custom)}}}))
    assert resolve_binary("claude_code", "claude") == str(custom)


def test_configured_binary_that_does_not_exist_is_not_found(tmp_path, monkeypatch, xdg_config_file):
    _executable(tmp_path / "claude")
    monkeypatch.setenv("PATH", str(tmp_path))
    xdg_config_file.write_text(json.dumps({"plugins": {"claude_code": {"binary": str(tmp_path / "missing")}}}))
    assert resolve_binary("claude_code", "claude") is None


def test_unconfigured_binary_missing_from_path_is_not_found(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert resolve_binary("claude_code", "claude") is None
