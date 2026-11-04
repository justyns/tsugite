"""GET /api/link-preview answers from the plugin link resolver registry."""

import asyncio
import logging
import threading

import pytest
from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPServer
from tsugite_daemon.config import HTTPConfig
from tsugite_daemon.webhook_store import WebhookStore

from tsugite.link_previews import attributing_to, register_link_resolver
from tsugite.secrets.registry import get_registry

PR_URL = "https://forge.example.test/octo-org/demo/pulls/7"


@pytest.fixture
def token_store(tmp_path):
    from tsugite_daemon.auth import TokenStore

    return TokenStore(tmp_path / "tokens.json")


@pytest.fixture
def server(tmp_path, token_store):
    return HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8374),
        adapter=None,
        webhook_store=WebhookStore(tmp_path / "webhooks.json"),
        token_store=token_store,
    )


@pytest.fixture
def client(server, token_store):
    _record, raw = token_store.create_admin_token(name="test-request-token")
    return TestClient(server.app, headers={"Authorization": f"Bearer {raw}"})


def _register(resolve, path=r"/pulls/\d+$", plugin="forge"):
    with attributing_to(plugin):
        register_link_resolver(name="pulls", hosts=["forge.example.test"], path=path, resolve=resolve)


def _preview(client, url=PR_URL):
    return client.get("/api/link-preview", params={"url": url})


def test_serves_the_normalized_preview(client):
    _register(
        lambda url: {
            "label": "demo#7",
            "title": "Add retry",
            "url": url,
            "status": {"text": "CI failure", "tone": "err"},
            "badges": [{"text": "open", "tone": "ok"}, {"text": "CI failure", "tone": "nope"}],
            "fields": [{"label": "author", "value": "octocat"}, {"label": "number", "value": 7}],
        }
    )

    resp = _preview(client)

    assert resp.status_code == 200
    assert resp.json() == {
        "label": "demo#7",
        "title": "Add retry",
        "url": PR_URL,
        "status": {"text": "CI failure", "tone": "err"},
        "badges": [{"text": "open", "tone": "ok"}, {"text": "CI failure", "tone": "mute"}],
        "fields": [{"label": "author", "value": "octocat"}, {"label": "number", "value": "7"}],
        "icon": None,
    }


def test_clips_long_text_and_long_lists(client):
    _register(
        lambda url: {
            "label": "x" * 500,
            "badges": [{"text": str(n), "tone": "info"} for n in range(50)],
            "fields": [{"label": "f", "value": "v\n" * 500}] * 50,
        }
    )

    body = _preview(client).json()

    assert len(body["label"]) <= 80
    assert body["title"] == body["label"]
    assert body["status"] is None
    assert len(body["badges"]) == len(body["fields"]) == 8
    assert "\n" not in body["fields"][0]["value"]
    assert len(body["fields"][0]["value"]) <= 200


def test_keeps_only_an_http_url(client):
    _register(lambda url: {"label": "demo#7", "url": "javascript:alert(1)"})

    assert _preview(client).json()["url"] == PR_URL


def test_keeps_an_icon_under_the_resolvers_own_plugin_assets(client):
    _register(lambda url: {"label": "demo#7", "icon": "/api/plugins/forge/ui/icons/forgejo.svg"})

    assert _preview(client).json()["icon"] == "/api/plugins/forge/ui/icons/forgejo.svg"


@pytest.mark.parametrize(
    "icon",
    [
        "/api/plugins/other/ui/icons/forgejo.svg",
        "/api/plugins/forge/ui/../../link-preview",
        "/api/plugins/forge/ui/icons/../icon.svg",
        "/api/plugins/forge/ui/",
        "https://forge.example.test/api/plugins/forge/ui/icon.svg",
        "//forge.example.test/api/plugins/forge/ui/icon.svg",
        "javascript:alert(1)",
        "/api/plugins/forge/ui/icon.svg?v=1",
        "/api/plugins/forge/ui/icon.svg#x",
        "/api/plugins/forge/ui/icon.svg'),url(x",
        '/api/plugins/forge/ui/icon".svg',
        "/api/plugins/forge/ui/icon%2esvg",
        7,
    ],
)
def test_drops_any_other_icon(client, icon):
    _register(lambda url: {"label": "demo#7", "icon": icon})

    assert _preview(client).json()["icon"] is None


def test_masks_registered_secrets(client):
    get_registry().register("forge-token", "tok-SECRETVALUE-123")
    try:
        _register(lambda url: {"label": "demo#7", "title": "leak tok-SECRETVALUE-123"})
        assert "SECRETVALUE" not in _preview(client).text
    finally:
        get_registry().clear()


def test_awaits_an_async_resolver(client):
    async def resolve(url):
        return {"label": "demo#7"}

    _register(resolve)

    assert _preview(client).json()["label"] == "demo#7"


@pytest.mark.parametrize("url", [None, "ftp://forge.example.test/x", "javascript:alert(1)"])
def test_rejects_anything_but_an_http_url(client, url):
    resp = client.get("/api/link-preview", params={} if url is None else {"url": url})
    assert resp.status_code == 400


def test_a_link_no_resolver_matches_is_not_found(client):
    _register(lambda url: {"label": "demo#7"})

    assert _preview(client, "https://forge.example.test/octo-org/demo/issues/7").status_code == 404


@pytest.mark.parametrize(
    "resolve",
    [
        lambda url: None,
        lambda url: 1 / 0,
        lambda url: {"title": "no label"},
        lambda url: {"label": None},
        lambda url: {"label": " "},
        lambda url: "not a dict",
    ],
    ids=["none", "raises", "no-label", "null-label", "blank-label", "not-a-dict"],
)
def test_a_resolver_without_a_preview_is_not_found_and_logged(client, caplog, resolve):
    _register(resolve)

    with caplog.at_level(logging.DEBUG, logger="tsugite_daemon"):
        assert _preview(client).status_code == 404

    assert "forge:pulls" in caplog.text


def _slow_resolvers(gate):
    async def slow_async(url):
        while not gate.is_set():
            await asyncio.sleep(0.01)
        return {"label": "demo#7"}

    def slow_sync(url):
        gate.wait(1)
        return {"label": "demo#7"}

    return {"async": slow_async, "sync": slow_sync}


@pytest.mark.parametrize("kind", ["async", "sync"])
def test_a_slow_resolver_times_out_and_is_asked_again(client, monkeypatch, caplog, kind):
    from tsugite_daemon.adapters.http import link_preview

    monkeypatch.setattr(link_preview, "RESOLVE_TIMEOUT", 0.05)
    gate = threading.Event()
    _register(_slow_resolvers(gate)[kind])

    with caplog.at_level(logging.INFO, logger="tsugite_daemon"):
        resp = _preview(client)
    assert resp.status_code == 504
    assert resp.json() == {"error": "preview timed out"}
    assert "forge:pulls" in caplog.text

    gate.set()
    assert _preview(client).status_code == 200


def test_needs_the_full_token(server, client):
    _register(lambda url: {"label": "demo#7"})
    server.plugin_ui_surfaces.append({"plugin": "forge"})
    surface = client.post("/api/plugin-surface-token", json={"plugin": "forge"}).json()["token"]
    bare = TestClient(server.app)

    assert bare.get("/api/link-preview", params={"url": PR_URL}).status_code == 401
    resp = bare.get("/api/link-preview", params={"url": PR_URL}, headers={"Authorization": f"Bearer {surface}"})
    assert resp.status_code == 403


def test_plugins_endpoint_lists_resolver_patterns(client):
    _register(lambda url: None)

    assert client.get("/api/plugins").json()["link_resolvers"] == [
        {"plugin": "forge", "name": "pulls", "hosts": ["forge.example.test"], "path": r"/pulls/\d+$"}
    ]
