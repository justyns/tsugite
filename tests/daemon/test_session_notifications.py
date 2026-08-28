"""Session-to-session notification: a finishing session messages every session
that asked to hear about it, and the sessions still waiting show up as
`waiting_on` on the sidebar payload.

The parent link (`parent_id`) is the one-target special case of the same
mechanism; `notify_sessions` is the explicit list.
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPAgentAdapter, HTTPServer
from tsugite_daemon.auth import TokenStore
from tsugite_daemon.config import HTTPConfig, RuntimeDefaults
from tsugite_daemon.session_runner import (
    MAX_CHAIN_DEPTH,
    SessionRunner,
    get_current_chain_depth,
    set_current_chain_depth,
)
from tsugite_daemon.session_store import Session, SessionSource, SessionStatus, SessionStore


class _RecordingAdapter:
    """Adapter that records every turn it is asked to run."""

    def __init__(self, result="all done"):
        self.result = result
        self.calls: list[SimpleNamespace] = []

    def resolve_model(self):
        return "test-model"

    async def handle_message(self, user_id, message, channel_context, custom_logger=None):
        self.calls.append(SimpleNamespace(user_id=user_id, message=message, source=channel_context.source))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    @property
    def notifications(self) -> list[SimpleNamespace]:
        """Turns started by a completion notification, newest last."""
        return [c for c in self.calls if c.source.startswith("session_")]

    def notified_ids(self) -> list[str]:
        return [c.user_id.removeprefix("session:") for c in self.notifications]


@pytest.fixture
def store(tmp_path, history_dir):
    return SessionStore(tmp_path / "session_store.json")


@pytest.fixture
def adapter():
    return _RecordingAdapter()


@pytest.fixture
def runner(store, adapter):
    return SessionRunner(store, adapter)


def _listener(store: SessionStore, sid: str, **kwargs) -> str:
    """Create an idle session that can receive a notification."""
    store.create_session(Session(id=sid, source=SessionSource.INTERACTIVE.value, user_id="alice", **kwargs))
    return sid


def _worker(sid: str = "child", **kwargs) -> Session:
    return Session(
        id=sid,
        source=SessionSource.BACKGROUND.value,
        prompt="do the thing",
        title="Do the thing",
        **kwargs,
    )


async def _settle(runner: SessionRunner, sid: str, timeout: float = 3.0) -> None:
    """Wait until the session's run task and its completion fan-out are done."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sid not in runner._active_tasks:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"session '{sid}' did not finish within {timeout}s")


# ── Fan-out ──


@pytest.mark.asyncio
async def test_completion_reaches_every_listed_session(store, adapter, runner):
    _listener(store, "watcher-a")
    _listener(store, "watcher-b")

    runner.start_session(_worker(notify_sessions=["watcher-a", "watcher-b"]))
    await _settle(runner, "child")

    assert adapter.notified_ids() == ["watcher-a", "watcher-b"]


@pytest.mark.asyncio
async def test_notification_carries_id_status_title_and_result(store, adapter, runner):
    _listener(store, "watcher-a")

    runner.start_session(_worker(notify_sessions=["watcher-a"]))
    await _settle(runner, "child")

    assert adapter.notified_ids() == ["watcher-a"]
    message = adapter.notifications[0].message
    assert 'id="child"' in message
    assert 'status="completed"' in message
    assert 'title="Do the thing"' in message
    assert "all done" in message


@pytest.mark.asyncio
async def test_failure_notification_carries_the_error(store, adapter, runner):
    adapter.result = RuntimeError("disk full")
    _listener(store, "watcher-a")

    runner.start_session(_worker(notify_sessions=["watcher-a"]))
    await _settle(runner, "child")

    assert adapter.notified_ids() == ["watcher-a"]
    message = adapter.notifications[0].message
    assert 'status="failed"' in message
    assert "disk full" in message


@pytest.mark.asyncio
async def test_cancellation_notifies_listeners(store, adapter, runner):
    hang = asyncio.Event()

    async def slow(user_id, message, channel_context, custom_logger=None):
        adapter.calls.append(SimpleNamespace(user_id=user_id, message=message, source=channel_context.source))
        if channel_context.source.startswith("session_"):
            return "ack"
        await hang.wait()
        return "never"

    adapter.handle_message = slow
    _listener(store, "watcher-a")

    runner.start_session(_worker(notify_sessions=["watcher-a"]))
    await asyncio.sleep(0.05)
    runner.cancel_session("child")
    await _settle(runner, "child")

    assert adapter.notified_ids() == ["watcher-a"]
    assert 'status="cancelled"' in adapter.notifications[0].message


@pytest.mark.asyncio
async def test_parent_is_notified_exactly_once_when_also_listed(store, adapter, runner):
    _listener(store, "parent")

    runner.start_session(_worker(parent_id="parent", notify_sessions=["parent"]))
    await _settle(runner, "child")

    assert adapter.notified_ids() == ["parent"]


# ── Skips: a target that cannot be woken is never an error ──


@pytest.mark.asyncio
async def test_missing_target_is_skipped_and_the_rest_still_notified(store, adapter, runner):
    _listener(store, "watcher-b")

    runner.start_session(_worker(notify_sessions=["ghost", "watcher-b"]))
    await _settle(runner, "child")

    assert adapter.notified_ids() == ["watcher-b"]
    assert store.get_session("child").status == SessionStatus.COMPLETED.value


@pytest.mark.asyncio
async def test_finished_target_is_skipped(store, adapter, runner):
    _listener(store, "watcher-a", status=SessionStatus.COMPLETED.value)
    _listener(store, "watcher-b")

    runner.start_session(_worker(notify_sessions=["watcher-a", "watcher-b"]))
    await _settle(runner, "child")

    assert adapter.notified_ids() == ["watcher-b"]


@pytest.mark.asyncio
async def test_self_reference_is_skipped(store, adapter, runner):
    runner.start_session(_worker(notify_sessions=["child"]))
    await _settle(runner, "child")

    assert adapter.notifications == []
    assert store.get_session("child").status == SessionStatus.COMPLETED.value


@pytest.mark.asyncio
async def test_a_failing_target_does_not_starve_the_others(store, adapter, runner):
    _listener(store, "watcher-a")
    _listener(store, "watcher-b")
    real_handle = adapter.handle_message

    async def flaky(user_id, message, channel_context, custom_logger=None):
        if user_id == "session:watcher-a":
            raise RuntimeError("target adapter blew up")
        return await real_handle(user_id, message, channel_context, custom_logger)

    adapter.handle_message = flaky

    runner.start_session(_worker(notify_sessions=["watcher-a", "watcher-b"]))
    await _settle(runner, "child")

    assert adapter.notified_ids() == ["watcher-b"]


@pytest.mark.asyncio
async def test_a_compacted_target_is_notified_through_its_successor(store, adapter, runner):
    _listener(store, "watcher-a")
    successor = store.compact_session("watcher-a")

    runner.start_session(_worker(notify_sessions=["watcher-a"]))
    await _settle(runner, "child")

    assert adapter.notified_ids() == [successor.id]


# ── Chain depth ──


@pytest.mark.asyncio
async def test_notification_stops_at_the_chain_depth_ceiling(store, adapter, runner):
    _listener(store, "watcher-a")
    set_current_chain_depth(MAX_CHAIN_DEPTH)

    runner.start_session(_worker(notify_sessions=["watcher-a"]))
    await _settle(runner, "child")

    assert adapter.notifications == []


@pytest.mark.asyncio
async def test_a_notified_turn_runs_one_level_deeper(store, adapter, runner):
    _listener(store, "watcher-a")
    seen = []
    real_handle = adapter.handle_message

    async def probe(user_id, message, channel_context, custom_logger=None):
        seen.append(get_current_chain_depth())
        return await real_handle(user_id, message, channel_context, custom_logger)

    adapter.handle_message = probe
    set_current_chain_depth(2)

    runner.start_session(_worker(notify_sessions=["watcher-a"]))
    await _settle(runner, "child")

    assert seen == [2, 3]


# ── notify_sessions across session rotation ──


def test_notify_sessions_survives_compaction(store):
    _listener(store, "worker", notify_sessions=["watcher-a"])

    successor = store.compact_session("worker")

    assert successor.notify_sessions == ["watcher-a"]
    assert store.get_session("worker").notify_sessions == []


def test_a_branch_does_not_inherit_notify_sessions(store):
    _listener(store, "worker", notify_sessions=["watcher-a"])

    with patch("tsugite_daemon.session_store.get_history_backend") as history:
        history.return_value.create_branch.return_value = "branch-1"
        branch = store.branch_session("worker", at_event_id=1)

    assert branch.notify_sessions == []
    assert store.get_session("worker").notify_sessions == ["watcher-a"]


# ── Registering a target after creation ──


def test_add_notify_session_appends_and_dedupes(store):
    _listener(store, "worker")
    _listener(store, "watcher-a")
    _listener(store, "watcher-b")

    store.add_notify_session("worker", "watcher-a")
    store.add_notify_session("worker", "watcher-b")
    store.add_notify_session("worker", "watcher-a")

    assert store.get_session("worker").notify_sessions == ["watcher-a", "watcher-b"]


def test_add_notify_session_survives_a_restart(store, tmp_path, history_dir):
    _listener(store, "worker")
    _listener(store, "watcher-a")
    store.add_notify_session("worker", "watcher-a")

    reopened = SessionStore(tmp_path / "session_store.json")

    assert reopened.get_session("worker").notify_sessions == ["watcher-a"]


def test_add_notify_session_rejects_an_unknown_session(store):
    with pytest.raises(ValueError, match="not found"):
        store.add_notify_session("ghost", "watcher-a")


def test_runner_add_notify_session_broadcasts(store):
    bus = MagicMock()
    runner = SessionRunner(store, {}, event_bus=bus)
    _listener(store, "worker")
    _listener(store, "watcher-a")

    runner.add_notify_session("worker", "watcher-a")

    actions = [call.args[1] for call in bus.emit.call_args_list if call.args[0] == "session_update"]
    assert actions and actions[-1]["id"] == "worker"


def test_start_session_tool_passes_notify_sessions(store, adapter):
    from tsugite.tools import sessions as sessions_tools

    runner = SessionRunner(store, adapter)
    started = {}

    with (
        patch.object(sessions_tools, "_session_runner", runner),
        patch.object(sessions_tools, "_call", lambda fn, *a, **kw: started.setdefault("session", a[0])),
    ):
        sessions_tools.start_session(prompt="go", notify_sessions=["watcher-a"])

    assert started["session"].notify_sessions == ["watcher-a"]


# ── waiting_on (derived) ──


def test_waiting_on_map_lists_unfinished_notifiers(store):
    _listener(store, "worker-1", notify_sessions=["watcher-a"])
    _listener(store, "worker-2", notify_sessions=["watcher-a", "watcher-b"])

    assert store.waiting_on_map() == {"watcher-a": ["worker-1", "worker-2"], "watcher-b": ["worker-2"]}


def test_waiting_on_map_drops_finished_notifiers(store):
    _listener(store, "worker-1", notify_sessions=["watcher-a"], status=SessionStatus.COMPLETED.value)
    _listener(store, "worker-2", notify_sessions=["watcher-a"], status=SessionStatus.RUNNING.value)

    assert store.waiting_on_map() == {"watcher-a": ["worker-2"]}


def test_waiting_on_map_ignores_a_self_reference(store):
    _listener(store, "worker-1", notify_sessions=["worker-1"])

    assert store.waiting_on_map() == {}


# ── waiting_on on the sidebar payload ──


@pytest.fixture
def http_adapter(tmp_path, history_dir):
    from tsugite.workspace import WorkspaceNotFoundError

    workspace = tmp_path / "ws"
    workspace.mkdir()
    session_store = SessionStore(tmp_path / "http_store.json")
    config = RuntimeDefaults(workspace_dir=workspace, agent_file="default")
    with patch("tsugite.workspace.Workspace") as mock_ws:
        mock_ws.load.side_effect = WorkspaceNotFoundError("nope")
        return HTTPAgentAdapter(runtime=config, session_store=session_store)


@pytest.fixture
def client_and_token(http_adapter, tmp_path):
    token_store = TokenStore(tmp_path / "tokens.json")
    _t, raw = token_store.create_admin_token(name="t")
    server = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8374),
        adapter=http_adapter,
        webhook_store=None,
        token_store=token_store,
    )
    return TestClient(server.app), raw


def _rows(client, token) -> dict:
    resp = client.get("/api/chat/sessions", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    return {r["id"]: r for r in resp.json()["sessions"]}


def test_sessions_payload_carries_waiting_on(http_adapter, client_and_token):
    client, token = client_and_token
    store = http_adapter.session_store
    store.create_session(Session(id="watcher", source=SessionSource.INTERACTIVE.value))
    store.create_session(
        Session(
            id="worker",
            source=SessionSource.BACKGROUND.value,
            status=SessionStatus.RUNNING.value,
            notify_sessions=["watcher"],
        )
    )

    rows = _rows(client, token)
    assert rows["watcher"].get("waiting_on") == ["worker"]
    assert rows["worker"].get("waiting_on") == []


def test_waiting_on_clears_when_the_worker_finishes(http_adapter, client_and_token):
    client, token = client_and_token
    store = http_adapter.session_store
    store.create_session(Session(id="watcher", source=SessionSource.INTERACTIVE.value))
    store.create_session(
        Session(
            id="worker",
            source=SessionSource.BACKGROUND.value,
            status=SessionStatus.RUNNING.value,
            notify_sessions=["watcher"],
        )
    )

    store.update_session("worker", status=SessionStatus.COMPLETED.value)

    assert _rows(client, token)["watcher"].get("waiting_on") == []


@pytest.mark.asyncio
async def test_a_target_registered_mid_run_is_notified(store, adapter, runner):
    hang = asyncio.Event()

    async def slow(user_id, message, channel_context, custom_logger=None):
        adapter.calls.append(SimpleNamespace(user_id=user_id, message=message, source=channel_context.source))
        if channel_context.source.startswith("session_"):
            return "ack"
        await hang.wait()
        return "all done"

    adapter.handle_message = slow
    _listener(store, "watcher-a")

    runner.start_session(_worker())
    await asyncio.sleep(0.05)
    runner.add_notify_session("child", "watcher-a")
    hang.set()
    await _settle(runner, "child")

    assert adapter.notified_ids() == ["watcher-a"]


@pytest.mark.asyncio
async def test_a_repeated_target_is_notified_once(store, adapter, runner):
    _listener(store, "watcher-a")

    runner.start_session(_worker(notify_sessions=["watcher-a", "watcher-a"]))
    await _settle(runner, "child")

    assert adapter.notified_ids() == ["watcher-a"]


def test_add_notify_session_rejects_an_unknown_target(store):
    _listener(store, "src")

    with pytest.raises(ValueError, match="not found"):
        store.add_notify_session("src", "no-such-session")


def test_add_notify_session_rejects_another_users_session(store):
    """A notify target starts a turn in that chat, so it stays within one person."""
    _listener(store, "src")
    store.create_session(Session(id="theirs", source=SessionSource.INTERACTIVE.value, user_id="bob"))

    with pytest.raises(ValueError, match="another user"):
        store.add_notify_session("src", "theirs")

    assert store.get_session("src").notify_sessions == []


@pytest.mark.asyncio
async def test_resumable_target_is_notified_despite_being_completed(store, adapter, runner):
    """A background session stays reachable after its first turn: `session_reply`
    never gated on FINISHED_STATUSES, so notify must not either."""
    store.create_session(
        Session(
            id="chatty",
            source=SessionSource.BACKGROUND.value,
            status=SessionStatus.COMPLETED.value,
            resumable=True,
        )
    )

    runner.start_session(_worker(notify_sessions=["chatty"]))
    await _settle(runner, "child")

    assert adapter.notified_ids() == ["chatty"]


@pytest.mark.asyncio
async def test_failed_resumable_target_is_still_skipped(store, adapter, runner):
    """Resumable only lifts the gate for a session that finished its turn cleanly."""
    store.create_session(
        Session(
            id="broken",
            source=SessionSource.BACKGROUND.value,
            status=SessionStatus.FAILED.value,
            resumable=True,
        )
    )

    runner.start_session(_worker(notify_sessions=["broken"]))
    await _settle(runner, "child")

    assert adapter.notifications == []
