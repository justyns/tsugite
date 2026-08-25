"""Session source distinguishes where a session was created (web / discord /
cli) instead of collapsing everything to 'interactive', so the UI can badge
each session by origin.

Covers: the new enum values, threading `source` through every session-creation
path, backward-compatible load of legacy 'interactive' records, the ?source=
filter, and untouched serialization on GET /api/chat/sessions.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite_daemon.auth import TokenStore
from tsugite_daemon.config import HTTPConfig, RuntimeDefaults
from tsugite_daemon.session_store import (
    Session,
    SessionSource,
    SessionStore,
    create_interactive_session,
)

# ── enum ──


def test_source_enum_has_web_discord_cli():
    assert SessionSource.WEB.value == "web"
    assert SessionSource.DISCORD.value == "discord"


# ── store-level source threading ──


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "session_store.json")


def test_get_or_create_interactive_defaults_to_interactive(store):
    s = store.get_or_create_interactive("u1")
    assert s.source == SessionSource.INTERACTIVE.value


def test_get_or_create_interactive_stamps_source(store):
    s = store.get_or_create_interactive("u1", source=SessionSource.WEB.value)
    assert s.source == "web"


def test_create_default_session_defaults_to_interactive(store):
    assert store.create_default_session("u1").source == SessionSource.INTERACTIVE.value


def test_create_default_session_stamps_source(store):
    assert store.create_default_session("u1", source=SessionSource.DISCORD.value).source == "discord"


def test_get_or_create_named_session_defaults_to_interactive(store):
    assert store.get_or_create_named_session("u1", "discord").source == SessionSource.INTERACTIVE.value


def test_get_or_create_named_session_stamps_source(store):
    s = store.get_or_create_named_session("u1", "discord", source=SessionSource.DISCORD.value)
    assert s.source == "discord"


def test_get_or_create_channel_session_defaults_to_interactive(store):
    assert store.get_or_create_channel_session("chan-1", "u1").source == SessionSource.INTERACTIVE.value


def test_get_or_create_channel_session_stamps_source(store):
    s = store.get_or_create_channel_session("chan-1", "u1", source=SessionSource.DISCORD.value)
    assert s.source == "discord"


def test_create_interactive_session_defaults_to_interactive(store):
    sid = create_interactive_session(store, "agent-x", "u1")
    assert store.get_session(sid).source == SessionSource.INTERACTIVE.value


def test_create_interactive_session_stamps_source(store):
    sid = create_interactive_session(store, "agent-x", "web-anon", source=SessionSource.WEB.value)
    assert store.get_session(sid).source == "web"


# ── backward compatibility: legacy records must still load ──


def test_persisted_interactive_session_loads(tmp_path):
    path = tmp_path / "session_store.json"
    SessionStore(path).create_session(Session(id="legacy-1", source="interactive", user_id="u1"))
    reloaded = SessionStore(path)
    assert reloaded.get_session("legacy-1").source == "interactive"


def test_persisted_new_source_value_loads(tmp_path):
    path = tmp_path / "session_store.json"
    SessionStore(path).create_session(Session(id="web-1", source=SessionSource.WEB.value, user_id="u1"))
    reloaded = SessionStore(path)
    assert reloaded.get_session("web-1").source == "web"


# ── ?source= filter ──


def test_list_sessions_filters_by_new_source(store):
    store.create_session(Session(id="w1", source=SessionSource.WEB.value, user_id="u1"))
    store.create_session(Session(id="d1", source=SessionSource.DISCORD.value, user_id="u1"))
    assert [s.id for s in store.list_sessions(source="web")] == ["w1"]
    assert [s.id for s in store.list_sessions(source="discord")] == ["d1"]


# ── web creation paths (real HTTP call sites) ──


@pytest.fixture
def web_client(tmp_path):
    from tsugite.workspace import WorkspaceNotFoundError

    ws = tmp_path / "ws"
    ws.mkdir()
    store = SessionStore(tmp_path / "session_store.json")
    config = RuntimeDefaults(workspace_dir=ws, agent_file="default")
    with patch("tsugite.workspace.Workspace") as mock_ws:
        mock_ws.load.side_effect = WorkspaceNotFoundError("nope")
        adapter = HTTPAgentAdapter(runtime=config, session_store=store)
    token_store = TokenStore(tmp_path / "tokens.json")
    _t, raw = token_store.create_admin_token(name="t")
    server = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8374),
        adapter=adapter,
        webhook_store=None,
        token_store=token_store,
    )
    from tsugite_daemon.session_runner import SessionRunner

    server.session_runner = SessionRunner(store=store, adapter=adapter)
    return TestClient(server.app), raw, adapter


def test_web_chat_stamps_web_source(web_client):
    client, token, adapter = web_client

    async def quick_handle(*args, **kwargs):
        return "ok"

    with patch.object(adapter, "handle_message", side_effect=quick_handle):
        with client.stream(
            "POST",
            "/api/chat",
            json={"message": "hi", "user_id": "web-anon"},
            headers={"Authorization": f"Bearer {token}"},
        ) as resp:
            assert resp.status_code == 200
            for _ in resp.iter_bytes():
                pass

    sessions = adapter.session_store.list_sessions()
    assert sessions, "chat should have created a session"
    assert all(s.source == "web" for s in sessions)


def test_web_sessions_new_stamps_web_source(web_client):
    client, token, adapter = web_client
    resp = client.post(
        "/api/chat/sessions/new",
        json={"user_id": "web-anon"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 201
    sid = resp.json()["id"]
    assert adapter.session_store.get_session(sid).source == "web"


def test_sessions_endpoint_serializes_new_source_untouched(web_client):
    client, token, adapter = web_client
    adapter.session_store.create_session(Session(id="w1", source=SessionSource.WEB.value, user_id="web-anon"))
    adapter.session_store.create_session(Session(id="d1", source=SessionSource.DISCORD.value, user_id="12345"))
    resp = client.get("/api/chat/sessions", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    rows = {r["id"]: r for r in resp.json()["sessions"]}
    assert rows["w1"]["source"] == "web"
    assert rows["d1"]["source"] == "discord"


def test_api_sessions_endpoint_serializes_new_source_untouched(web_client):
    """The other listing surface, GET /api/sessions (sessions.py), must also
    pass the raw source string through with no enum coercion."""
    client, token, adapter = web_client
    adapter.session_store.create_session(Session(id="w2", source=SessionSource.WEB.value, user_id="web-anon"))
    resp = client.get("/api/sessions", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    rows = {r["id"]: r for r in resp.json()["sessions"]}
    assert rows["w2"]["source"] == "web"


# ── discord creation paths (real adapter call sites) ──


@pytest.fixture
def discord_adapter(tmp_path):
    from tsugite_daemon.config import DiscordBotConfig
    from tsugite_discord import DiscordAdapter

    from tsugite.workspace import WorkspaceNotFoundError

    ws = tmp_path / "ws"
    ws.mkdir()
    store = SessionStore(tmp_path / "session_store.json")
    config = RuntimeDefaults(workspace_dir=ws, agent_file="default")
    bot_config = DiscordBotConfig(name="b", agent="test-agent", token_secret="dummy")
    with patch("tsugite.workspace.Workspace") as mock_ws:
        mock_ws.load.side_effect = WorkspaceNotFoundError("nope")
        return DiscordAdapter(
            bot_config=bot_config,
            runtime=config,
            session_store=store,
        )


def _channel_context(**overrides):
    from tsugite_daemon.adapters.base import ChannelContext

    base = dict(source="discord", channel_id="c", user_id="12345", reply_to="discord:c")
    base.update(overrides)
    return ChannelContext(**base)


def test_discord_dm_named_route_is_discord_source(discord_adapter):
    """Default DM route: bot_config.session_name defaults to 'discord', so this
    lands on get_or_create_named_session."""
    msg = SimpleNamespace(author=SimpleNamespace(id=12345), channel=SimpleNamespace(id=999), guild=None)
    session = discord_adapter._resolve_target_session(
        msg, _channel_context(channel_id="999"), is_thread=False, is_dm=True, thread_id=None
    )
    assert session.source == "discord"


def test_discord_dm_default_interactive_is_discord_source(discord_adapter):
    """DM route with session_name disabled falls through to get_or_create_interactive."""
    discord_adapter.bot_config.session_name = ""
    msg = SimpleNamespace(author=SimpleNamespace(id=12345), channel=SimpleNamespace(id=999), guild=None)
    session = discord_adapter._resolve_target_session(
        msg, _channel_context(channel_id="999"), is_thread=False, is_dm=True, thread_id=None
    )
    assert session.source == "discord"


def test_discord_channel_session_is_discord_source(discord_adapter):
    msg = SimpleNamespace(
        author=SimpleNamespace(id=12345),
        channel=SimpleNamespace(id=777, name="general"),
        guild=SimpleNamespace(id=555),
    )
    cc = _channel_context(channel_id="777", metadata={"guild_id": "555"})
    session = discord_adapter._resolve_target_session(msg, cc, is_thread=False, is_dm=False, thread_id=None)
    assert session.source == "discord"


def test_discord_thread_session_and_parent_are_discord_source(discord_adapter):
    channel = SimpleNamespace(id=888, name="my-thread", parent_id=777)
    msg = SimpleNamespace(author=SimpleNamespace(id=12345), channel=channel, guild=SimpleNamespace(id=555))
    cc = _channel_context(channel_id="888", thread_id="888", metadata={"guild_id": "555"})
    session = discord_adapter._resolve_target_session(msg, cc, is_thread=True, is_dm=False, thread_id="888")
    assert session.source == "discord"
    assert discord_adapter.session_store.get_session(session.parent_id).source == "discord"


# ── cli: no daemon-session creation path exists ──


def test_cli_chat_does_not_create_daemon_sessions():
    """`tsu chat` runs the agent loop against the JSONL history system; it never
    constructs daemon Session objects or touches a SessionStore. There is no CLI
    call site to stamp source='cli'. This test pins that finding
    so a future reader does not hunt for a nonexistent CLI creation path.
    """
    import inspect

    from tsugite.cli import chat as chat_module

    src = inspect.getsource(chat_module)
    assert "SessionStore" not in src
    assert "session_store" not in src
    assert "get_or_create_interactive" not in src
    assert "create_interactive_session" not in src
