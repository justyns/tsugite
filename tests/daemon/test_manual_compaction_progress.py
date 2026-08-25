"""Manual compaction triggers must emit compaction_progress phase events.

Both user-initiated compaction paths must thread a progress callback into
`_compact_session` so subscribers see the same
compaction_started / compaction_progress / compaction_finished stream the
automatic (token-threshold / prompt-too-long) paths already emit:

- the web UI Compact button -> POST /api/chat/compact
- the /compact slash command -> POST /api/commands/compact

Without the callback, users clicking Compact or typing /compact only ever see
the bare "preparing to compact..." fallback while the phase-aware renderer
(summarizing N turns... / combining...) sits unused.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite_daemon.config import HTTPConfig, RuntimeDefaults
from tsugite_daemon.webhook_store import WebhookStore


class _RecordingBus:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def emit(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, dict(payload)))


# Phase payloads a real summarize_session emits, replayed by the fake compactor
# so the assertions exercise the manual site's wiring, not memory.py internals
# (those are covered end-to-end by tests/test_compaction_progress.py).
_FAKE_PHASES = [
    {"phase": "starting", "replaced_count": 3, "retained_count": 2},
    {"phase": "chunking"},
    {"phase": "summarizing", "chunk_index": 1, "chunk_total": 2},
    {"phase": "summarizing", "chunk_index": 2, "chunk_total": 2},
    {"phase": "combining"},
]


def _fake_compact_session_factory():
    async def fake_compact(session_id, instructions=None, reason=None, progress_callback=None):
        if progress_callback is not None:
            for payload in _FAKE_PHASES:
                progress_callback(payload)
        return SimpleNamespace(id=f"{session_id}-successor")

    return fake_compact


@pytest.fixture
def tmp_workspace(tmp_path):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    return workspace_dir


@pytest.fixture
def agent_config(tmp_workspace):
    return RuntimeDefaults(workspace_dir=tmp_workspace, agent_file="default")


@pytest.fixture
def adapter(agent_config, tmp_path):
    from tsugite_daemon.session_store import SessionStore

    from tsugite.workspace import WorkspaceNotFoundError

    session_store = SessionStore(tmp_path / "session_store.json", default_context_limit=128_000)

    with patch("tsugite.workspace.Workspace") as mock_ws_cls:
        mock_ws_cls.load.side_effect = WorkspaceNotFoundError("not found")
        return HTTPAgentAdapter(
            runtime=agent_config,
            session_store=session_store,
        )


@pytest.fixture
def bus(adapter):
    """Attach a recording bus the way the Gateway wires the real SSEBroadcaster
    onto each adapter (gateway.py: adapter.event_bus = http_server.event_bus)."""
    recording = _RecordingBus()
    adapter.event_bus = recording
    return recording


@pytest.fixture
def token_store(tmp_path):
    from tsugite_daemon.auth import TokenStore

    return TokenStore(tmp_path / "tokens.json")


@pytest.fixture
def admin_token(token_store):
    _, raw = token_store.create_admin_token(name="test-token")
    return raw


@pytest.fixture
def client(adapter, agent_config, tmp_path, token_store):
    server = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8374),
        adapter=adapter,
        webhook_store=WebhookStore(tmp_path / "webhooks.json"),
        token_store=token_store,
    )
    return TestClient(server.app)


def _seed_default_session(adapter, user_id: str) -> str:
    session = adapter.session_store.get_or_create_interactive(user_id)
    adapter.session_store.update_token_count(session.id, 100)
    return session.id


def _auth(admin_token: str) -> dict:
    return {"Authorization": f"Bearer {admin_token}", "Content-Type": "application/json"}


def _progress_payloads(bus: _RecordingBus) -> list[dict]:
    return [p for t, p in bus.events if t == "compaction_progress"]


def test_compact_button_emits_compaction_progress(client, adapter, bus, admin_token):
    """POST /api/chat/compact (the web UI Compact button) must
    broadcast compaction_progress phase events, not just started/finished."""
    user_id = "compact-button-user"
    session_id = _seed_default_session(adapter, user_id)

    with patch.object(adapter, "_compact_session", new=AsyncMock(side_effect=_fake_compact_session_factory())):
        resp = client.post(
            "/api/chat/compact",
            content=json.dumps({"user_id": user_id}),
            headers=_auth(admin_token),
        )

    assert resp.status_code == 200, resp.text

    types = [t for t, _ in bus.events]
    assert "compaction_progress" in types, (
        "manual Compact button dropped every phase payload: the call site passes "
        "no progress_callback to _compact_session"
    )

    phases = [p["phase"] for p in _progress_payloads(bus)]
    assert phases == [p["phase"] for p in _FAKE_PHASES]

    started_idx = types.index("compaction_started")
    first_progress_idx = types.index("compaction_progress")
    finished_idx = types.index("compaction_finished")
    assert started_idx < first_progress_idx < finished_idx

    for payload in _progress_payloads(bus):
        assert payload["session_id"] == session_id


def test_compact_command_emits_compaction_progress(client, adapter, bus, admin_token):
    """POST /api/commands/compact (the /compact slash command)
    must broadcast the same compaction_progress phase events."""
    user_id = "compact-command-user"
    session_id = _seed_default_session(adapter, user_id)

    with patch.object(adapter, "_compact_session", new=AsyncMock(side_effect=_fake_compact_session_factory())):
        resp = client.post(
            "/api/commands/compact",
            content=json.dumps({"user_id": user_id, "message": "focus on the schema"}),
            headers=_auth(admin_token),
        )

    assert resp.status_code == 200, resp.text

    types = [t for t, _ in bus.events]
    assert "compaction_progress" in types, (
        "/compact command dropped every phase payload: the call site passes no progress_callback to _compact_session"
    )

    phases = [p["phase"] for p in _progress_payloads(bus)]
    assert phases == [p["phase"] for p in _FAKE_PHASES]

    started_idx = types.index("compaction_started")
    first_progress_idx = types.index("compaction_progress")
    finished_idx = types.index("compaction_finished")
    assert started_idx < first_progress_idx < finished_idx

    for payload in _progress_payloads(bus):
        assert payload["session_id"] == session_id
