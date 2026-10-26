import json
import stat
from unittest.mock import AsyncMock

import pytest
from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPServer
from tsugite_daemon.auth import TokenStore
from tsugite_daemon.config import HTTPConfig
from tsugite_daemon.webhook_store import WebhookStore

from tsugite.providers import _cache, clear_cache, model_registry
from tsugite.providers.base import CompletionResponse, ModelInfo


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def client(tmp_path):
    token_store = TokenStore(tmp_path / "tokens.json")
    _st, raw = token_store.create_admin_token(name="providers-token")
    server = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8374),
        adapter=None,
        webhook_store=WebhookStore(tmp_path / "webhooks.json"),
        token_store=token_store,
    )
    return TestClient(server.app, headers={"Authorization": f"Bearer {raw}"})


def _by_name(client) -> dict[str, dict]:
    resp = client.get("/api/providers")
    assert resp.status_code == 200
    return {p["name"]: p for p in resp.json()["providers"]}


def test_requires_auth(client):
    del client.headers["Authorization"]
    assert client.get("/api/providers").status_code == 401


def test_lists_builtin_and_plugin_providers(client):
    providers = _by_name(client)
    assert {"openai", "ollama", "claude_code"} <= providers.keys()
    assert providers["openai"]["binary"] is None
    assert providers["claude_code"]["key"] is None


def test_api_key_status_reports_source_but_never_the_value(client, secret_backend, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-value")
    secret_backend.set("openai-api-key", "stored-value")

    resp = client.get("/api/providers")
    assert "stored-value" not in resp.text
    assert "env-value" not in resp.text
    providers = {p["name"]: p for p in resp.json()["providers"]}
    assert providers["openai"]["key"] == {
        "env": "OPENAI_API_KEY",
        "secret": "openai-api-key",
        "set": True,
        "source": "secret",
    }
    assert providers["anthropic"]["key"]["source"] == "env"
    assert providers["mistral"]["key"]["set"] is False
    assert providers["mistral"]["key"]["source"] is None


class _BrokenBackend:
    def get(self, name):
        raise RuntimeError("keychain locked")

    def list_names(self):
        raise RuntimeError("keychain locked")


def test_a_raising_secrets_backend_fails_only_the_rows_with_keys(client, monkeypatch):
    monkeypatch.setattr("tsugite.secrets._backend", _BrokenBackend())
    providers = _by_name(client)
    assert providers["openai"]["error"] == "keychain locked"
    assert "error" not in providers["claude_code"]


def test_a_raising_provider_fails_only_its_row(client, monkeypatch):
    from tsugite_daemon.adapters.http import providers as providers_module

    real = providers_module.get_provider

    def get_provider(name):
        if name == "mistral":
            raise RuntimeError("bad config")
        return real(name)

    monkeypatch.setattr(providers_module, "get_provider", get_provider)
    providers = _by_name(client)
    assert providers["mistral"] == {"name": "mistral", "binary": None, "key": None, "error": "bad config"}
    assert providers["openai"]["key"]["env"] == "OPENAI_API_KEY"


def test_secret_set_and_delete_through_the_api_apply_to_cached_keys(client, secret_backend, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert _by_name(client)["openai"]["key"]["set"] is False

    assert client.post("/api/secrets/openai-api-key", json={"value": "stored"}).status_code == 200
    assert _by_name(client)["openai"]["key"]["set"] is True

    assert client.delete("/api/secrets/openai-api-key").status_code == 200
    assert _by_name(client)["openai"]["key"]["set"] is False


def _executable(path):
    path.write_text("#!/bin/sh\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_binary_status_and_update(client, tmp_path, monkeypatch, xdg_config_file):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert _by_name(client)["claude_code"]["binary"] == {"default": "claude", "configured": None, "resolved": None}

    custom = _executable(tmp_path / "claude-bin")
    resp = client.put("/api/providers/claude_code", json={"binary": str(custom)})
    assert resp.status_code == 200
    assert resp.json()["binary"] == {"default": "claude", "configured": str(custom), "resolved": str(custom)}
    assert json.loads(xdg_config_file.read_text())["plugins"]["claude_code"]["binary"] == str(custom)

    resp = client.put("/api/providers/claude_code", json={"binary": ""})
    assert resp.json()["binary"]["configured"] is None
    assert "binary" not in json.loads(xdg_config_file.read_text())["plugins"]["claude_code"]


def test_binary_update_rejects_unknown_and_non_cli_providers(client):
    assert client.put("/api/providers/nope", json={"binary": "/x"}).status_code == 404
    assert client.put("/api/providers/openai", json={"binary": "/x"}).status_code == 400


@pytest.mark.parametrize("body", [["/x"], {"binary": 5}])
def test_binary_update_rejects_a_malformed_body(client, body, xdg_config_file):
    assert client.put("/api/providers/claude_code", json=body).status_code == 400


class _FakeProvider:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls: list[str] = []

    async def list_models(self) -> list[str]:
        return ["gpt-4.1", "gpt-4.1-nano"]

    async def acompletion(self, messages, model, stream=False, **kwargs):
        self.calls.append(model)
        if self.error:
            raise self.error
        return CompletionResponse(content="OK")


def test_test_runs_a_request_with_the_cheapest_listed_model(client):
    fake = _cache["openai"] = _FakeProvider()
    resp = client.post("/api/providers/openai/test", json={})
    assert resp.json() == {"ok": True, "model": "gpt-4.1-nano", "reply": "OK"}
    assert fake.calls == ["gpt-4.1-nano"]


def test_test_falls_back_to_registered_models_when_the_listing_is_empty(client, monkeypatch):
    fake = _cache["openai"] = _FakeProvider()
    fake.list_models = AsyncMock(return_value=[])
    monkeypatch.setattr(model_registry, "_REGISTRY", {"openai/pricey": ModelInfo(input_cost_per_million=9.0)})
    monkeypatch.setitem(model_registry._REGISTRY, "openai/cheap", ModelInfo(input_cost_per_million=0.1))
    assert client.post("/api/providers/openai/test", json={}).json()["model"] == "cheap"


def test_test_returns_the_error_text(client):
    _cache["openai"] = _FakeProvider(error=RuntimeError("401 invalid api key"))
    resp = client.post("/api/providers/openai/test", json={})
    assert resp.status_code == 200
    assert resp.json() == {"ok": False, "model": "gpt-4.1-nano", "error": "401 invalid api key"}


def test_test_returns_the_error_when_the_model_listing_fails(client):
    fake = _cache["openai"] = _FakeProvider()
    fake.list_models = AsyncMock(side_effect=ConnectionError("connection refused"))
    resp = client.post("/api/providers/openai/test", json={})
    assert resp.status_code == 200
    assert resp.json() == {"ok": False, "model": None, "error": "connection refused"}


def test_test_returns_the_error_when_the_provider_fails_to_build(client, monkeypatch):
    from tsugite_daemon.adapters.http import providers as providers_module

    def get_provider(name):
        raise RuntimeError("bad config")

    monkeypatch.setattr(providers_module, "get_provider", get_provider)
    resp = client.post("/api/providers/openai/test", json={})
    assert resp.status_code == 200
    assert resp.json() == {"ok": False, "model": None, "error": "bad config"}


def test_test_unknown_provider_is_404(client):
    assert client.post("/api/providers/nope/test", json={}).status_code == 404
