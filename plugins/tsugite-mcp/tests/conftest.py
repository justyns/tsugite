"""Shared fixtures for the MCP plugin's tests."""

import httpx
import pytest
import pytest_asyncio
from fake_server import FakeServer


class MemoryBackend:
    def __init__(self):
        self.values = {}

    def get(self, name):
        return self.values.get(name)

    def list_names(self):
        return sorted(self.values)

    def set(self, name, value):
        self.values[name] = value

    def delete(self, name):
        return self.values.pop(name, None) is not None


@pytest.fixture(autouse=True)
def isolate_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))


@pytest.fixture
def secrets_backend():
    from tsugite.secrets import set_backend
    from tsugite.secrets.registry import get_registry

    backend = MemoryBackend()
    set_backend(backend)
    yield backend
    set_backend(None)
    get_registry().clear()


@pytest.fixture
def fake(request):
    """The fake server. Parametrize it indirectly with FakeServer options."""
    return FakeServer(**getattr(request, "param", {}))


@pytest_asyncio.fixture
async def adapter(tmp_path, secrets_backend, fake):
    from tsugite_mcp.adapter import McpAdapter

    adapter = McpAdapter(tmp_path / "servers.json", client_factory=fake.client_factory)
    await adapter.start()
    yield adapter
    await adapter.stop()


@pytest_asyncio.fixture
async def daemon(adapter, tmp_path):
    """An in-process client for the daemon HTTP app with the plugin's routes mounted, and the auth headers."""
    from tsugite_daemon.adapters.http import HTTPServer
    from tsugite_daemon.auth import TokenStore
    from tsugite_daemon.config import HTTPConfig
    from tsugite_daemon.plugin_wiring import attach_plugin_http
    from tsugite_daemon.webhook_store import WebhookStore

    token_store = TokenStore(tmp_path / "tokens.json")
    server = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8374),
        adapter=None,
        webhook_store=WebhookStore(tmp_path / "webhooks.json"),
        token_store=token_store,
    )
    attach_plugin_http(server, "mcp", adapter, [])
    _record, raw = token_store.create_admin_token(name="test-request-token")
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://daemon")
    client.auth_headers = {"Authorization": f"Bearer {raw}"}
    async with client:
        yield client
