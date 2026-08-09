"""Adapter plugins can contribute UI surfaces the web UI renders in an iframe.

get_ui_surfaces() declares them; the gateway namespaces each one to
`plugin/<name>/<kind>` and resolves its `entry` under the plugin's own
`/api/plugins/<name>/` mount, so a surface can only ever point at the plugin that
declared it. GET /api/plugins serves the merged list.

Same duck-typed, error-isolated wiring as the route methods next door in
test_plugin_http_routes.py: one misbehaving plugin can't abort startup.
"""

import logging

import pytest
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPServer
from tsugite_daemon.config import HTTPConfig
from tsugite_daemon.gateway import attach_plugin_http, normalize_ui_surfaces
from tsugite_daemon.webhook_store import WebhookStore

DOC_SURFACE = {
    "kind": "doc",
    "label": "Document",
    "icon": "files",
    "entry": "ui/editor.html",
    "nav": True,
    "params": ["path"],
}


class FakePluginAdapter:
    """Duck-typed stand-in for a loaded adapter plugin."""

    def __init__(self, surfaces=None, public=None):
        self._surfaces = surfaces or []
        self._public = public or []
        self.event_bus = None

    def get_ui_surfaces(self):
        return self._surfaces

    def get_public_http_routes(self):
        return self._public


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


# ── normalize_ui_surfaces: namespacing, entry resolution, defaults ──


def test_kind_and_entry_are_namespaced_to_the_declaring_plugin():
    assert normalize_ui_surfaces("onlyoffice", [DOC_SURFACE]) == [
        {
            "plugin": "onlyoffice",
            "kind": "plugin/onlyoffice/doc",
            "label": "Document",
            "icon": "files",
            "entry": "/api/plugins/onlyoffice/ui/editor.html",
            "nav": True,
            "params": ["path"],
        }
    ]


def test_leading_slash_on_entry_does_not_double_up():
    (surface,) = normalize_ui_surfaces("alpha", [{"kind": "x", "entry": "/ui/index.html"}])
    assert surface["entry"] == "/api/plugins/alpha/ui/index.html"


def test_optional_fields_default_without_being_declared():
    (surface,) = normalize_ui_surfaces("alpha", [{"kind": "board", "entry": "ui/index.html"}])
    assert surface["label"] == "board", "label falls back to the declared kind, not the namespaced one"
    assert surface["icon"] == "", "only the web UI knows its icon set, so it owns that fallback"
    assert surface["nav"] is False
    assert surface["params"] == []


@pytest.mark.parametrize(
    "bad",
    [
        {"entry": "ui/index.html"},
        {"kind": "x"},
        {"kind": "", "entry": "ui/index.html"},
        "not-a-dict",
    ],
)
def test_incomplete_surface_is_dropped_with_a_warning(bad, caplog):
    """A surface the UI could not open must not reach it as a broken tab."""
    with caplog.at_level(logging.WARNING):
        surfaces = normalize_ui_surfaces("alpha", [bad, DOC_SURFACE])
    assert [s["kind"] for s in surfaces] == ["plugin/alpha/doc"], "the well-formed sibling must still survive"
    assert "alpha" in caplog.text


# ── attach_plugin_http: collection, isolation, http-disabled ──


def test_attach_registers_surfaces_on_the_server(server):
    attach_plugin_http(server, "onlyoffice", FakePluginAdapter(surfaces=[DOC_SURFACE]))
    assert [s["kind"] for s in server.plugin_ui_surfaces] == ["plugin/onlyoffice/doc"]


def test_surfaces_alone_are_enough_to_register(server):
    """A plugin can contribute a surface without contributing any HTTP route of
    its own (its assets can come from a Mount, or another plugin's routes)."""
    before = len(server.app.router.routes)
    attach_plugin_http(server, "alpha", FakePluginAdapter(surfaces=[DOC_SURFACE]))
    assert len(server.plugin_ui_surfaces) == 1
    assert len(server.app.router.routes) == before, "no routes declared means no Mount"


def test_two_plugins_surfaces_merge_without_collision(server):
    attach_plugin_http(server, "alpha", FakePluginAdapter(surfaces=[{"kind": "doc", "entry": "ui/a.html"}]))
    attach_plugin_http(server, "beta", FakePluginAdapter(surfaces=[{"kind": "doc", "entry": "ui/b.html"}]))
    assert [s["kind"] for s in server.plugin_ui_surfaces] == ["plugin/alpha/doc", "plugin/beta/doc"]


def test_adapter_without_the_method_is_skipped(server):
    class Bare:
        event_bus = None

    attach_plugin_http(server, "alpha", Bare())  # must not raise
    assert server.plugin_ui_surfaces == []


def test_raising_get_ui_surfaces_is_logged_and_skipped(server, caplog):
    class Boom:
        event_bus = None

        def get_ui_surfaces(self):
            raise RuntimeError("surface declaration blew up")

    with caplog.at_level(logging.WARNING):
        attach_plugin_http(server, "boom", Boom())  # must not propagate
    assert server.plugin_ui_surfaces == []
    assert "boom" in caplog.text
    # A subsequent well-behaved plugin still registers (per-plugin isolation).
    attach_plugin_http(server, "good", FakePluginAdapter(surfaces=[DOC_SURFACE]))
    assert [s["kind"] for s in server.plugin_ui_surfaces] == ["plugin/good/doc"]


def test_raising_surfaces_does_not_cost_the_plugin_its_routes(server):
    """The two collections are independent: a broken get_ui_surfaces() must not
    take the plugin's working routes down with it."""

    class HalfBroken(FakePluginAdapter):
        def get_ui_surfaces(self):
            raise RuntimeError("nope")

    async def endpoint(request):
        return JSONResponse({"ok": True})

    attach_plugin_http(server, "alpha", HalfBroken(public=[Route("/hook", endpoint, methods=["GET"])]))
    assert TestClient(server.app).get("/api/plugins/alpha/hook").status_code == 200


def test_http_disabled_skips_surfaces_with_a_warning(caplog):
    adapter = FakePluginAdapter(surfaces=[DOC_SURFACE])
    with caplog.at_level(logging.WARNING):
        attach_plugin_http(None, "alpha", adapter)  # HTTP disabled -> http_server is None
    assert any("alpha" in r.getMessage() for r in caplog.records), "must warn naming the plugin"


# ── GET /api/plugins: the payload the web UI reads ──


def test_plugins_endpoint_serves_registered_surfaces(server, test_token):
    attach_plugin_http(server, "onlyoffice", FakePluginAdapter(surfaces=[DOC_SURFACE]))
    body = TestClient(server.app).get("/api/plugins", headers={"Authorization": f"Bearer {test_token}"}).json()
    assert [s["kind"] for s in body["ui_surfaces"]] == ["plugin/onlyoffice/doc"]
    assert body["ui_surfaces"][0]["entry"] == "/api/plugins/onlyoffice/ui/editor.html"


def test_plugins_endpoint_reports_no_surfaces_when_none_registered(server, test_token):
    body = TestClient(server.app).get("/api/plugins", headers={"Authorization": f"Bearer {test_token}"}).json()
    assert body["ui_surfaces"] == []


# ── BaseAdapter default ──


def test_base_adapter_default_returns_empty():
    from tsugite_daemon.adapters.base import BaseAdapter

    assert BaseAdapter.get_ui_surfaces(object()) == []
