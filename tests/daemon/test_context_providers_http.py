"""HTTP surface for context providers.

Three routes let the composer enumerate menu providers, fetch a provider's
submenu, and run a server-side capture. They resolve the request's session to the
``ctx`` dict providers receive (user_id / agent / workspace_dir) exactly like the
neighboring chat/command endpoints, and share the daemon token guard.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPServer
from tsugite_daemon.auth import TokenStore
from tsugite_daemon.config import HTTPConfig
from tsugite_daemon.session_store import SessionStore
from tsugite_daemon.webhook_store import WebhookStore

from tsugite import context as ctx_module
from tsugite.attachments.base import Attachment
from tsugite.context import ContextChoice, ContextProvider, register_context_provider


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    monkeypatch.setattr(ctx_module, "ensure_loaded", lambda: None)
    ctx_module.reset_context_providers()
    yield
    ctx_module.reset_context_providers()


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


@pytest.fixture
def token(tmp_path):
    store = TokenStore(tmp_path / "tokens.json")
    _st, raw = store.create_admin_token(name="ctx-test")
    return store, raw


@pytest.fixture
def session_store(tmp_path):
    return SessionStore(tmp_path / "sessions.json")


@pytest.fixture
def client(tmp_path, workspace, token, session_store):
    token_store, _raw = token
    adapter = SimpleNamespace(
        agent_name="smokeagent",
        session_store=session_store,
        agent_config=SimpleNamespace(workspace_dir=workspace),
    )
    server = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8599),
        adapters={"smokeagent": adapter},
        webhook_store=WebhookStore(tmp_path / "webhooks.json"),
        agent_configs={},
        token_store=token_store,
    )
    return TestClient(server.app)


def _auth(token):
    return {"Authorization": f"Bearer {token[1]}"}


def test_list_requires_auth(client):
    assert client.get("/api/context-providers").status_code == 401


def test_capture_requires_auth(client):
    assert client.post("/api/context-providers/x/capture", json={"session_id": None, "arg": None}).status_code == 401


def test_choices_requires_auth(client):
    assert client.get("/api/context-providers/x/choices").status_code == 401


def test_search_requires_auth(client):
    assert client.get("/api/context-providers/x/search").status_code == 401


def test_list_filters_to_menu_providers(client, token):
    register_context_provider(
        ContextProvider(
            key="terminal",
            label="Terminal output",
            icon="term",
            capture=lambda arg, c: [],
            choices=lambda c: [],
        )
    )
    register_context_provider(ContextProvider(key="webpage", label="Web page", icon="link", detect=lambda m, c: []))
    resp = client.get("/api/context-providers", headers=_auth(token))
    assert resp.status_code == 200
    assert resp.json() == {
        "providers": [
            {
                "key": "terminal",
                "label": "Terminal output",
                "icon": "term",
                "has_choices": True,
                "picker": False,
                "in_menu": True,
                "autocomplete_prefix": None,
            }
        ]
    }


def test_list_has_choices_false_for_capture_only(client, token):
    register_context_provider(ContextProvider(key="snippet", label="Snippet", capture=lambda arg, c: []))
    body = client.get("/api/context-providers", headers=_auth(token)).json()
    assert body["providers"] == [
        {
            "key": "snippet",
            "label": "Snippet",
            "icon": "sparkle",
            "has_choices": False,
            "picker": False,
            "in_menu": True,
            "autocomplete_prefix": None,
        }
    ]


def test_list_serializes_picker_flag(client, token):
    register_context_provider(
        ContextProvider(key="file", label="Workspace file", icon="file", picker=True, capture=lambda arg, c: [])
    )
    body = client.get("/api/context-providers", headers=_auth(token)).json()
    assert body["providers"] == [
        {
            "key": "file",
            "label": "Workspace file",
            "icon": "file",
            "has_choices": False,
            "picker": True,
            "in_menu": True,
            "autocomplete_prefix": None,
        }
    ]


def test_list_includes_autocomplete_source_out_of_menu(client, token):
    """A prefix+search provider (menu=False) is listed so the frontend learns its
    prefix, but rides with in_menu=False so the add-context menu still hides it."""
    register_context_provider(
        ContextProvider(
            key="jira",
            label="Jira",
            icon="link",
            capture=lambda arg, c: [],
            search=lambda c, q: [],
            autocomplete_prefix="jira",
            menu=False,
        )
    )
    body = client.get("/api/context-providers", headers=_auth(token)).json()
    assert body["providers"] == [
        {
            "key": "jira",
            "label": "Jira",
            "icon": "link",
            "has_choices": False,
            "picker": False,
            "in_menu": False,
            "autocomplete_prefix": "jira",
        }
    ]


def test_capture_happy_path(client, token):
    register_context_provider(
        ContextProvider(
            key="snippet",
            label="Snippet",
            capture=lambda arg, c: [Attachment.context("snippet", "Snippet", "hello world")],
        )
    )
    resp = client.post(
        "/api/context-providers/snippet/capture",
        json={"session_id": None, "arg": None},
        headers=_auth(token),
    )
    assert resp.status_code == 200
    assert resp.json() == {"items": [{"key": "snippet", "label": "Snippet", "value": "hello world"}]}


def test_capture_provider_error_returns_400(client, token):
    def boom(arg, c):
        raise RuntimeError("provider exploded")

    register_context_provider(ContextProvider(key="boom", label="Boom", capture=boom))
    resp = client.post(
        "/api/context-providers/boom/capture",
        json={"session_id": None, "arg": None},
        headers=_auth(token),
    )
    assert resp.status_code == 400
    assert resp.json() == {"error": "provider exploded"}


def test_capture_unknown_key_returns_empty_items(client, token):
    resp = client.post(
        "/api/context-providers/nope/capture",
        json={"session_id": None, "arg": None},
        headers=_auth(token),
    )
    assert resp.status_code == 200
    assert resp.json() == {"items": []}


def test_choices_returns_options(client, token):
    register_context_provider(
        ContextProvider(
            key="terminal",
            label="Terminal output",
            capture=lambda arg, c: [],
            choices=lambda c: [ContextChoice("t1", "bash"), ContextChoice("t2", "python")],
        )
    )
    resp = client.get("/api/context-providers/terminal/choices", headers=_auth(token))
    assert resp.status_code == 200
    assert resp.json() == {"choices": [{"value": "t1", "label": "bash"}, {"value": "t2", "label": "python"}]}


def test_choices_empty_when_provider_has_none(client, token):
    register_context_provider(ContextProvider(key="snippet", label="Snippet", capture=lambda arg, c: []))
    resp = client.get("/api/context-providers/snippet/choices", headers=_auth(token))
    assert resp.status_code == 200
    assert resp.json() == {"choices": []}


def test_capture_resolves_session_ctx(client, token, session_store, workspace):
    """A capture must receive the session's user_id/agent/workspace_dir and the
    arg, resolved from the posted session_id."""
    seen: dict = {}

    def capture(arg, c):
        seen["arg"] = arg
        seen["ctx"] = c
        return []

    register_context_provider(ContextProvider(key="probe", label="Probe", capture=capture))
    session = session_store.get_or_create_interactive("web-user", "smokeagent")
    resp = client.post(
        "/api/context-providers/probe/capture",
        json={"session_id": session.id, "arg": "chosen-value"},
        headers=_auth(token),
    )
    assert resp.status_code == 200
    assert seen["arg"] == "chosen-value"
    assert seen["ctx"]["session_id"] == session.id
    assert seen["ctx"]["user_id"] == "web-user"
    assert seen["ctx"]["agent"] == "smokeagent"
    assert seen["ctx"]["workspace_dir"] == workspace


def test_choices_resolves_session_ctx(client, token, session_store, workspace):
    seen: dict = {}

    def choices(c):
        seen["ctx"] = c
        return []

    register_context_provider(ContextProvider(key="probe", label="Probe", capture=lambda arg, c: [], choices=choices))
    session = session_store.get_or_create_interactive("web-user", "smokeagent")
    resp = client.get(
        f"/api/context-providers/probe/choices?session_id={session.id}",
        headers=_auth(token),
    )
    assert resp.status_code == 200
    assert seen["ctx"]["session_id"] == session.id
    assert seen["ctx"]["agent"] == "smokeagent"
    assert seen["ctx"]["workspace_dir"] == workspace


def test_search_returns_query_matches(client, token):
    def search(context, query):
        tickets = {"auth flow": "PROJ-1", "billing": "PROJ-2"}
        return [ContextChoice(v, k) for k, v in tickets.items() if query.lower() in k]

    register_context_provider(ContextProvider(key="jira", label="Jira", autocomplete_prefix="jira", search=search))
    resp = client.get("/api/context-providers/jira/search?q=auth", headers=_auth(token))
    assert resp.status_code == 200
    assert resp.json() == {"results": [{"value": "PROJ-1", "label": "auth flow"}]}


def test_search_unknown_or_no_search_returns_empty(client, token):
    register_context_provider(ContextProvider(key="snippet", label="Snippet", capture=lambda arg, c: []))
    assert client.get("/api/context-providers/missing/search?q=x", headers=_auth(token)).json() == {"results": []}
    assert client.get("/api/context-providers/snippet/search?q=x", headers=_auth(token)).json() == {"results": []}


def test_search_resolves_session_ctx_and_query(client, token, session_store, workspace):
    seen: dict = {}

    def search(context, query):
        seen["ctx"] = context
        seen["query"] = query
        return []

    register_context_provider(ContextProvider(key="jira", label="Jira", autocomplete_prefix="jira", search=search))
    session = session_store.get_or_create_interactive("web-user", "smokeagent")
    resp = client.get(
        f"/api/context-providers/jira/search?session_id={session.id}&q=auth",
        headers=_auth(token),
    )
    assert resp.status_code == 200
    assert seen["query"] == "auth"
    assert seen["ctx"]["session_id"] == session.id
    assert seen["ctx"]["agent"] == "smokeagent"
    assert seen["ctx"]["workspace_dir"] == workspace
