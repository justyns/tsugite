"""Adapter plugins can contribute UI surfaces the web UI renders in an iframe.

get_ui_surfaces() declares them; the gateway namespaces each one to
`plugin/<name>/<kind>`, resolves its `entry` under the plugin's own
`/api/plugins/<name>/` mount, and serves the declared `assets` directory there.
GET /api/plugins serves the merged list.

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
from tsugite_daemon.gateway import _collect_plugin_ui, attach_plugin_http
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


# ── _collect_plugin_ui: namespacing, entry resolution, defaults, assets ──


def surfaces_of(plugin_name, declared):
    return _collect_plugin_ui(plugin_name, declared)[0]


def test_kind_and_entry_are_namespaced_to_the_declaring_plugin():
    assert surfaces_of("onlyoffice", [DOC_SURFACE]) == [
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
    (surface,) = surfaces_of("alpha", [{"kind": "x", "entry": "/ui/index.html"}])
    assert surface["entry"] == "/api/plugins/alpha/ui/index.html"


def test_optional_fields_default_without_being_declared():
    (surface,) = surfaces_of("alpha", [{"kind": "board", "entry": "ui/index.html"}])
    assert surface["label"] == "board", "label falls back to the declared kind, not the namespaced one"
    assert surface["icon"] == ""
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
        surfaces = surfaces_of("alpha", [bad, DOC_SURFACE])
    assert [s["kind"] for s in surfaces] == ["plugin/alpha/doc"], "the well-formed sibling must still survive"
    assert "alpha" in caplog.text


def test_assets_dir_is_taken_from_the_declaring_surface(tmp_path):
    assert _collect_plugin_ui("alpha", [{"kind": "x", "entry": "ui/a.html", "assets": tmp_path}])[1] == tmp_path


def test_a_path_that_is_not_a_directory_is_not_mounted(tmp_path, caplog):
    declared = [{"kind": "x", "entry": "ui/a.html", "assets": tmp_path / "gone"}]
    with caplog.at_level(logging.WARNING):
        assert _collect_plugin_ui("alpha", declared)[1] is None
    assert "alpha" in caplog.text


def test_no_assets_declared_means_nothing_to_mount():
    assert _collect_plugin_ui("alpha", [DOC_SURFACE])[1] is None


def test_surfaces_sharing_one_dir_is_not_a_conflict(tmp_path, caplog):
    declared = [
        {"kind": "a", "entry": "ui/a.html", "assets": tmp_path},
        {"kind": "b", "entry": "ui/b.html", "assets": str(tmp_path)},
    ]
    with caplog.at_level(logging.WARNING):
        assert _collect_plugin_ui("alpha", declared)[1] == tmp_path
    assert caplog.text == ""


def test_a_second_different_dir_is_named_not_silently_dropped(tmp_path, caplog):
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    declared = [
        {"kind": "a", "entry": "ui/a.html", "assets": tmp_path / "one"},
        {"kind": "b", "entry": "ui/b.html", "assets": tmp_path / "two"},
    ]
    with caplog.at_level(logging.WARNING):
        assert _collect_plugin_ui("alpha", declared)[1] == tmp_path / "one"
    assert "alpha" in caplog.text


def test_assets_on_a_dropped_descriptor_is_dropped_with_it(tmp_path, caplog):
    """One validity rule: a descriptor the UI can't open must not still get a mount."""
    with caplog.at_level(logging.WARNING):
        surfaces, assets = _collect_plugin_ui("alpha", [{"assets": tmp_path}])
    assert (surfaces, assets) == ([], None)


def test_assets_never_reaches_the_payload(tmp_path):
    """It is a server-side path; shipping it would leak the daemon's filesystem."""
    (surface,) = surfaces_of("alpha", [{"kind": "x", "entry": "ui/a.html", "assets": tmp_path}])
    assert "assets" not in surface


# ── attach_plugin_http: collection, isolation, http-disabled ──


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


def test_declared_assets_are_served_public_under_the_plugin_prefix(server, tmp_path):
    """The entry URL the payload advertises must serve without a token."""
    (tmp_path / "panel.html").write_text("<!doctype html><title>panel</title>")
    attach_plugin_http(
        server,
        "alpha",
        FakePluginAdapter(surfaces=[{"kind": "panel", "entry": "ui/panel.html", "assets": tmp_path}]),
    )
    (surface,) = server.plugin_ui_surfaces
    resp = TestClient(server.app).get(surface["entry"])
    assert resp.status_code == 200, "the entry URL the payload advertises must actually serve"
    assert "<title>panel</title>" in resp.text


def test_a_missing_assets_dir_is_reported_at_startup_and_not_mounted(server, caplog):
    with caplog.at_level(logging.WARNING):
        attach_plugin_http(
            server,
            "alpha",
            FakePluginAdapter(surfaces=[{"kind": "panel", "entry": "ui/panel.html", "assets": "/no/such/dir"}]),
        )
    assert "alpha" in caplog.text
    assert TestClient(server.app).get("/api/plugins/alpha/ui/panel.html").status_code == 404


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


# ── BaseAdapter default ──


def test_base_adapter_default_returns_empty():
    from tsugite_daemon.adapters.base import BaseAdapter

    assert BaseAdapter.get_ui_surfaces(object()) == []
