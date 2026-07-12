"""Adapter plugins can mount HTTP routes under /api/plugins/{name}.

Two surfaces:
  - get_http_routes()        -> auth-wrapped (daemon bearer token required)
  - get_public_http_routes() -> no auth (plugin does its own access control)

Wiring is the gateway module-level attach_plugin_http(http_server, name, adapter),
which is duck-typed + error-isolated so one misbehaving plugin can't abort startup.
"""

import logging

import pytest
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPServer
from tsugite_daemon.config import HTTPConfig
from tsugite_daemon.gateway import attach_plugin_executors, attach_plugin_http
from tsugite_daemon.webhook_store import WebhookStore


def _route(path: str, marker: str) -> Route:
    async def endpoint(request):
        return JSONResponse({"marker": marker, "path": request.url.path})

    return Route(path, endpoint, methods=["GET"])


class FakePluginAdapter:
    """Duck-typed stand-in for a loaded adapter plugin."""

    def __init__(self, authed=None, public=None, executors=None):
        self._authed = authed or []
        self._public = public or []
        self._executors = executors or {}
        self.event_bus = None
        self.http_check_auth = None

    def get_http_routes(self):
        return self._authed

    def get_public_http_routes(self):
        return self._public

    def get_job_executors(self):
        return self._executors


class FakeOrchestrator:
    """Records register_executor(name, executor) calls."""

    def __init__(self):
        self.registered: list[tuple[str, object]] = []

    def register_executor(self, name, executor):
        self.registered.append((name, executor))


@pytest.fixture
def token_store(tmp_path):
    from tsugite_daemon.auth import TokenStore

    return TokenStore(tmp_path / "tokens.json")


@pytest.fixture
def test_token(token_store):
    _st, raw = token_store.create_admin_token(name="test-request-token")
    return raw


@pytest.fixture
def server(tmp_path, token_store):
    return HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8374),
        adapters={},
        webhook_store=WebhookStore(tmp_path / "webhooks.json"),
        agent_configs={},
        token_store=token_store,
    )


# ── mount_plugin_routes: auth wrapping + prefix ──


def test_authed_route_401_without_token_200_with(server, test_token):
    server.mount_plugin_routes("alpha", [_route("/ping", "alpha-authed")], [])
    client = TestClient(server.app)
    assert client.get("/api/plugins/alpha/ping").status_code == 401
    resp = client.get("/api/plugins/alpha/ping", headers={"Authorization": f"Bearer {test_token}"})
    assert resp.status_code == 200
    assert resp.json()["marker"] == "alpha-authed"


def test_public_route_200_without_token(server):
    server.mount_plugin_routes("alpha", [], [_route("/hook", "alpha-public")])
    client = TestClient(server.app)
    resp = client.get("/api/plugins/alpha/hook")
    assert resp.status_code == 200
    assert resp.json()["marker"] == "alpha-public"


def test_prefix_isolation_between_two_plugins(server):
    server.mount_plugin_routes("alpha", [], [_route("/a", "alpha")])
    server.mount_plugin_routes("beta", [], [_route("/b", "beta")])
    client = TestClient(server.app)
    assert client.get("/api/plugins/alpha/a").json()["marker"] == "alpha"
    assert client.get("/api/plugins/beta/b").json()["marker"] == "beta"
    # A plugin's route must not be reachable under another plugin's prefix.
    assert client.get("/api/plugins/alpha/b").status_code == 404
    assert client.get("/api/plugins/beta/a").status_code == 404


def test_existing_daemon_routes_unaffected(server, test_token):
    server.mount_plugin_routes("alpha", [_route("/ping", "alpha")], [])
    client = TestClient(server.app)
    # Public health still open; authed core route still guarded + reachable with a token.
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/agents").status_code == 401
    assert client.get("/api/agents", headers={"Authorization": f"Bearer {test_token}"}).status_code == 200


def test_public_and_authed_coexist_under_same_plugin(server, test_token):
    server.mount_plugin_routes("alpha", [_route("/admin", "authed")], [_route("/hook", "public")])
    client = TestClient(server.app)
    assert client.get("/api/plugins/alpha/hook").status_code == 200
    assert client.get("/api/plugins/alpha/admin").status_code == 401
    assert client.get("/api/plugins/alpha/admin", headers={"Authorization": f"Bearer {test_token}"}).status_code == 200


# ── attach_plugin_http: gateway wiring, isolation, disabled/routeless ──


def test_attach_sets_event_bus_and_check_auth_and_mounts(server):
    adapter = FakePluginAdapter(public=[_route("/hook", "x")])
    attach_plugin_http(server, "alpha", adapter)
    assert adapter.event_bus is server.event_bus, "the plugin adapter must get the shared SSE bus"
    assert adapter.http_check_auth == server.check_auth, (
        "the adapter must get the daemon auth callable for its own routes"
    )
    client = TestClient(server.app)
    assert client.get("/api/plugins/alpha/hook").status_code == 200


def test_attach_routeless_adapter_adds_no_mount(server):
    before = len(server.app.router.routes)
    attach_plugin_http(server, "alpha", FakePluginAdapter())
    assert len(server.app.router.routes) == before, "an adapter with no routes must not add a Mount"


def test_attach_adapter_without_route_methods_skipped(server):
    class Bare:
        event_bus = None

    before = len(server.app.router.routes)
    attach_plugin_http(server, "alpha", Bare())  # must not raise
    assert len(server.app.router.routes) == before


def test_attach_error_isolation_does_not_raise_or_mount(server, caplog):
    class Boom:
        event_bus = None

        def get_http_routes(self):
            raise RuntimeError("plugin route collection blew up")

        def get_public_http_routes(self):
            return []

    before = len(server.app.router.routes)
    with caplog.at_level(logging.WARNING):
        attach_plugin_http(server, "boom", Boom())  # must not propagate
    assert len(server.app.router.routes) == before
    # A subsequent well-behaved plugin still mounts (per-plugin isolation).
    attach_plugin_http(server, "good", FakePluginAdapter(public=[_route("/ok", "good")]))
    assert TestClient(server.app).get("/api/plugins/good/ok").status_code == 200


def test_attach_http_disabled_warns_and_skips(caplog):
    adapter = FakePluginAdapter(public=[_route("/hook", "x")])
    with caplog.at_level(logging.WARNING):
        attach_plugin_http(None, "alpha", adapter)  # HTTP disabled -> http_server is None
    assert any("alpha" in r.getMessage() for r in caplog.records), "must warn naming the plugin"


# ── attach_plugin_executors: connects the WS3a orchestrator seam ──


def test_attach_executors_registers_each():
    orch = FakeOrchestrator()
    cc = object()
    adapter = FakePluginAdapter(executors={"cc": cc})
    attach_plugin_executors(orch, "cc_driver", adapter)
    assert orch.registered == [("cc", cc)], "each get_job_executors() entry must be registered on the orchestrator"


def test_attach_executors_backrefs_orchestrator_onto_adapter():
    """Executors need the orchestrator to call complete_worker/fail_worker; the
    wiring hands it back via set_jobs_orchestrator when the adapter exposes it."""

    class AdapterWithBackref:
        def __init__(self):
            self.got = None

        def get_job_executors(self):
            return {"cc": object()}

        def set_jobs_orchestrator(self, orch):
            self.got = orch

    orch = FakeOrchestrator()
    adapter = AdapterWithBackref()
    attach_plugin_executors(orch, "cc_driver", adapter)
    assert adapter.got is orch, "the adapter must receive the orchestrator backref"


def test_attach_executors_none_orchestrator_no_crash():
    adapter = FakePluginAdapter(executors={"cc": object()})
    attach_plugin_executors(None, "cc_driver", adapter)  # jobs disabled -> must not raise


def test_attach_executors_adapter_without_method_skipped():
    orch = FakeOrchestrator()

    class Bare:
        pass

    attach_plugin_executors(orch, "x", Bare())  # must not raise
    assert orch.registered == []


def test_attach_executors_error_isolation(caplog):
    orch = FakeOrchestrator()

    class Boom:
        def get_job_executors(self):
            raise RuntimeError("executor discovery blew up")

    with caplog.at_level(logging.WARNING):
        attach_plugin_executors(orch, "boom", Boom())  # must not propagate
    assert orch.registered == []


# ── BaseAdapter defaults ──


def test_base_adapter_default_route_methods_return_empty():
    from tsugite_daemon.adapters.base import BaseAdapter

    assert BaseAdapter.get_http_routes(object()) == []
    assert BaseAdapter.get_public_http_routes(object()) == []
