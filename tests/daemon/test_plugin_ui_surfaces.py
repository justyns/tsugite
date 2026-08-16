"""Plugins can contribute UI surfaces the web UI renders in an iframe.

A plugin registers them at import; the gateway namespaces each one to
`plugin/<name>/<kind>`, resolves its `entry` under the plugin's own
`/api/plugins/<name>/` mount, and serves the declared `assets` directory there.
GET /api/plugins serves the merged list.

Surfaces reach the gateway as an argument, so a plugin with no adapter still gets
its page, and a page function can replace the assets directory.
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

    def __init__(self, public=None):
        self._public = public or []
        self.event_bus = None

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
            "events": [],
            "mode": "full",
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
    assert surface["events"] == [], "a surface that asks for no daemon events is forwarded none"
    assert surface["mode"] == "full", "an undeclared mode keeps the whole-region rail behaviour"


def test_a_surface_declares_which_daemon_events_it_wants():
    """The host holds the one event stream and forwards a surface only the types it named,
    so a plugin page never sees the rest of the daemon's traffic."""
    (surface,) = surfaces_of("alpha", [{**DOC_SURFACE, "events": ["onlyoffice_document_update"]}])
    assert surface["events"] == ["onlyoffice_document_update"]


def test_a_surface_can_declare_that_it_docks_beside_the_workspace():
    (surface,) = surfaces_of("alpha", [{**DOC_SURFACE, "mode": "workspace"}])
    assert surface["mode"] == "workspace"


def test_an_unknown_mode_is_named_and_falls_back_to_full(caplog):
    with caplog.at_level(logging.WARNING):
        (surface,) = surfaces_of("alpha", [{**DOC_SURFACE, "mode": "sidebar"}])
    assert surface["mode"] == "full"
    assert "alpha" in caplog.text


@pytest.mark.parametrize(
    "bad",
    [
        {"entry": "ui/index.html"},
        {"kind": "x"},
        {"kind": "", "entry": "ui/index.html"},
        {"page": lambda: "<h1>no kind</h1>"},
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
    attach_plugin_http(server, "alpha", FakePluginAdapter(), [DOC_SURFACE])
    assert len(server.plugin_ui_surfaces) == 1
    assert len(server.app.router.routes) == before, "no routes declared means no Mount"


def test_a_plugin_with_no_adapter_can_mount_a_surface(server):
    """A surface comes from the registry, so a single-file plugin that never
    produced an adapter still gets its page."""
    attach_plugin_http(server, "dashboard", None, [DOC_SURFACE])
    assert [s["kind"] for s in server.plugin_ui_surfaces] == ["plugin/dashboard/doc"]


def test_two_plugins_surfaces_merge_without_collision(server):
    attach_plugin_http(server, "alpha", FakePluginAdapter(), [{"kind": "doc", "entry": "ui/a.html"}])
    attach_plugin_http(server, "beta", FakePluginAdapter(), [{"kind": "doc", "entry": "ui/b.html"}])
    assert [s["kind"] for s in server.plugin_ui_surfaces] == ["plugin/alpha/doc", "plugin/beta/doc"]


def test_declared_assets_are_served_public_under_the_plugin_prefix(server, tmp_path):
    """The entry URL the payload advertises must serve without a token."""
    (tmp_path / "panel.html").write_text("<!doctype html><title>panel</title>")
    attach_plugin_http(
        server, "alpha", FakePluginAdapter(), [{"kind": "panel", "entry": "ui/panel.html", "assets": tmp_path}]
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
            FakePluginAdapter(),
            [{"kind": "panel", "entry": "ui/panel.html", "assets": "/no/such/dir"}],
        )
    assert "alpha" in caplog.text
    assert TestClient(server.app).get("/api/plugins/alpha/ui/panel.html").status_code == 404


def test_adapter_routes_and_a_registered_surface_share_one_mount(server, tmp_path):
    """Starlette returns on the first prefix match, so a second Mount under
    /api/plugins/<name> would be dead code. Routes and surface must merge before
    mounting, or one of them silently 404s."""
    (tmp_path / "panel.html").write_text("<!doctype html><title>panel</title>")

    async def endpoint(request):
        return JSONResponse({"ok": True})

    attach_plugin_http(
        server,
        "alpha",
        FakePluginAdapter(public=[Route("/hook", endpoint, methods=["GET"])]),
        [{"kind": "panel", "entry": "ui/panel.html", "assets": tmp_path}],
    )
    client = TestClient(server.app)
    assert client.get("/api/plugins/alpha/hook").status_code == 200
    assert client.get("/api/plugins/alpha/ui/panel.html").status_code == 200


def test_http_disabled_skips_surfaces_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        attach_plugin_http(None, "alpha", FakePluginAdapter(), [DOC_SURFACE])
    assert any("alpha" in r.getMessage() for r in caplog.records), "must warn naming the plugin"


# ── page functions: a surface whose entry is generated, not a static file ──


def test_a_page_function_is_served_as_the_entry(server):
    """A page function removes the assets directory, so a one-file plugin stays one file."""
    attach_plugin_http(server, "alpha", None, [{"kind": "dash", "page": lambda: "<h1>homelab</h1>"}])

    (surface,) = server.plugin_ui_surfaces
    assert surface["entry"] == "/api/plugins/alpha/page/dash"
    resp = TestClient(server.app).get(surface["entry"])
    assert resp.status_code == 200, "the page must serve without a token, as an iframe navigation carries none"
    assert resp.text == "<h1>homelab</h1>"
    assert "no-store" in resp.headers["cache-control"], "a generated page must not be heuristically cached"


def test_an_async_page_function_is_served(server):
    async def page():
        return "<h1>async</h1>"

    attach_plugin_http(server, "alpha", None, [{"kind": "dash", "page": page}])

    assert TestClient(server.app).get("/api/plugins/alpha/page/dash").text == "<h1>async</h1>"


def test_a_raising_page_is_isolated_and_named(server, caplog):
    def page():
        raise RuntimeError("page blew up")

    attach_plugin_http(server, "alpha", None, [{"kind": "dash", "page": page}])

    with caplog.at_level(logging.WARNING):
        resp = TestClient(server.app).get("/api/plugins/alpha/page/dash")
    assert resp.status_code == 500
    assert "alpha" in caplog.text


# ── GET /api/plugins: the payload the web UI reads ──


def test_plugins_endpoint_serves_registered_surfaces(server, test_token):
    attach_plugin_http(server, "onlyoffice", FakePluginAdapter(), [DOC_SURFACE])
    body = TestClient(server.app).get("/api/plugins", headers={"Authorization": f"Bearer {test_token}"}).json()
    assert [s["kind"] for s in body["ui_surfaces"]] == ["plugin/onlyoffice/doc"]
    assert body["ui_surfaces"][0]["entry"] == "/api/plugins/onlyoffice/ui/editor.html"
