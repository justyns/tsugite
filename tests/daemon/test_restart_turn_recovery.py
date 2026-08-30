"""A turn that dies with the daemon must be repaired at next boot.

Without repair, the session's history ends mid-turn (no terminal event): the
progress cache keeps a live status label forever ("Waiting on LLM..."), the
web UI treats the session as mid-turn and hides the last turn's replay
(dropTrailing), and nothing tells the user what happened.

Mechanism: adapters durably mark `Session.turn_in_flight` around each turn
(write-through), and SessionStore's boot recovery appends an explanatory
`info` + terminal `session_error` event to the history of any session whose
turn was in flight (or that was left RUNNING) when the previous daemon died.
"""

import contextlib
import sqlite3
from pathlib import Path

import pytest
from tsugite_daemon.session_store import Session, SessionSource, SessionStatus, SessionStore

from tests.history_helpers import load_history_session, seed_history_session


def _seed_mid_turn_history(history_dir: Path, session_id: str) -> None:
    storage = seed_history_session(session_id, agent="test", model="m")
    storage.record("user_input", text="do the thing")
    storage.record("turn_start", turn=1)
    storage.record("code", content="x = 1")
    storage.record("tool_call", tool="run", arguments={"command": "true"})


def _event_types(history_dir: Path, session_id: str) -> list[str]:
    storage = load_history_session(session_id)
    return [e.type for e in storage.iter_events()]


def test_boot_repair_finalizes_in_flight_interactive_turn(tmp_path, history_dir):
    store = SessionStore(tmp_path / "session_store.json")
    store.create_session(Session(id="s1", source=SessionSource.INTERACTIVE.value, user_id="u"))
    store.begin_turn("s1")
    _seed_mid_turn_history(history_dir, "s1")

    reopened = SessionStore(tmp_path / "session_store.json")

    types = _event_types(history_dir, "s1")
    assert types[-1] == "session_error", f"repair must append a terminal event; got tail {types[-3:]}"
    assert "info" in types, "repair must leave a visible explanation for the user"
    session = reopened.get_session("s1")
    assert session.turn_in_flight is False, "the in-flight marker must be cleared"
    assert session.status == SessionStatus.ACTIVE.value, "an interactive session stays usable; only the turn died"
    assert reopened.session_progress_summary("s1")["status_text"] == "", (
        "the progress label must clear so the UI stops treating the session as mid-turn"
    )


def test_boot_repair_appends_terminal_event_for_running_session(tmp_path, history_dir):
    store = SessionStore(tmp_path / "session_store.json")
    store.create_session(
        Session(
            id="s2",
            source=SessionSource.SPAWNED.value,
            status=SessionStatus.RUNNING.value,
            user_id="u",
        )
    )
    _seed_mid_turn_history(history_dir, "s2")

    reopened = SessionStore(tmp_path / "session_store.json")

    assert reopened.get_session("s2").status == SessionStatus.FAILED.value
    assert _event_types(history_dir, "s2")[-1] == "session_error"


def test_boot_repair_skips_clean_sessions(tmp_path, history_dir):
    store = SessionStore(tmp_path / "session_store.json")
    store.create_session(Session(id="s3", source=SessionSource.INTERACTIVE.value, user_id="u"))
    storage = seed_history_session("s3", agent="test", model="m")
    storage.record("user_input", text="hi")
    storage.record("session_end", status="success")
    before = _event_types(history_dir, "s3")

    SessionStore(tmp_path / "session_store.json")

    assert _event_types(history_dir, "s3") == before, "clean sessions must not get repair events"


def test_begin_and_end_turn_write_through(tmp_path, history_dir):
    store = SessionStore(tmp_path / "session_store.json")
    store.create_session(Session(id="s4", user_id="u"))
    store.begin_turn("s4")
    assert SessionStore(tmp_path / "session_store.json").get_session("s4") is not None

    # Marker is durable mid-turn... (fresh store sees it; that reopen also
    # repairs, so assert against the db row via a plain reopen BEFORE repair
    # isn't possible - instead assert end_turn clears it durably.)
    store.end_turn("s4")
    reopened = SessionStore(tmp_path / "session_store.json")
    assert reopened.get_session("s4").turn_in_flight is False


@pytest.mark.asyncio
async def test_adapter_brackets_turn_with_in_flight_marker(tmp_path, history_dir, monkeypatch):
    """handle_message must set the durable marker while the turn runs and clear
    it on both the success and error paths."""
    from tsugite_daemon.adapters.base import BaseAdapter, ChannelContext
    from tsugite_daemon.config import RuntimeDefaults

    class _StubAdapter(BaseAdapter):
        async def start(self):
            pass

        async def stop(self):
            pass

    store = SessionStore(tmp_path / "session_store.json")
    store.create_session(Session(id="s1", source=SessionSource.INTERACTIVE.value, user_id="u1"))
    adapter = _StubAdapter(RuntimeDefaults(workspace_dir=tmp_path / "ws", agent_file="default"), store)

    seen = {}

    async def fake_inner(*_args, **kwargs):
        if (st := kwargs.get("_broadcast_state")) is not None:
            st["conv_id"] = "s1"
        store.begin_turn("s1")  # inner sets it once routing resolves
        seen["during"] = store.get_session("s1").turn_in_flight
        return "ok"

    monkeypatch.setattr(adapter, "_handle_message_inner", fake_inner)
    ctx = ChannelContext(source="test", channel_id="c1", user_id="u1", reply_to="t")
    await adapter.handle_message("u1", "hi", ctx)
    assert seen["during"] is True
    assert store.get_session("s1").turn_in_flight is False, "marker must clear on the success path"

    async def fake_inner_raises(*_args, **kwargs):
        if (st := kwargs.get("_broadcast_state")) is not None:
            st["conv_id"] = "s1"
        store.begin_turn("s1")
        raise RuntimeError("boom")

    monkeypatch.setattr(adapter, "_handle_message_inner", fake_inner_raises)
    with pytest.raises(RuntimeError):
        await adapter.handle_message("u1", "hi", ctx)
    assert store.get_session("s1").turn_in_flight is False, "marker must clear on the error path"


def _adapter_with_unwritable_store(tmp_path, monkeypatch):
    """An adapter whose daemon.db refuses writes, which is what breaks end_turn
    and then the failure report that end_turn's handler attempts."""
    from tsugite_daemon.adapters.base import BaseAdapter, ChannelContext
    from tsugite_daemon.config import RuntimeDefaults

    class _StubAdapter(BaseAdapter):
        async def start(self):
            pass

        async def stop(self):
            pass

    store = SessionStore(tmp_path / "session_store.json")
    store.create_session(Session(id="s1", source=SessionSource.INTERACTIVE.value, user_id="u1"))
    adapter = _StubAdapter(RuntimeDefaults(workspace_dir=tmp_path / "ws", agent_file="default"), store)

    def _refuse_write(*_args, **_kwargs):
        raise sqlite3.OperationalError("attempt to write a readonly database")

    monkeypatch.setattr(store, "end_turn", _refuse_write)
    monkeypatch.setattr(store, "append_event", _refuse_write)

    return adapter, ChannelContext(source="test", channel_id="c1", user_id="u1", reply_to="t")


@pytest.mark.asyncio
async def test_a_failing_end_turn_report_does_not_mask_the_original_error(tmp_path, history_dir, monkeypatch):
    """The turn's own exception is the diagnosis; a teardown report that also
    fails must not overwrite it with a storage error."""
    adapter, ctx = _adapter_with_unwritable_store(tmp_path, monkeypatch)

    async def fake_inner(*_args, **kwargs):
        kwargs["_broadcast_state"]["conv_id"] = "s1"
        raise RuntimeError("the-original-failure")

    monkeypatch.setattr(adapter, "_handle_message_inner", fake_inner)

    with pytest.raises(RuntimeError, match="the-original-failure"):
        await adapter.handle_message("u1", "hi", ctx)


@pytest.mark.parametrize("break_reporting", ["raises", "unimportable"])
@pytest.mark.asyncio
async def test_the_turn_completes_broadcasting_even_when_reporting_fails(
    tmp_path, history_dir, monkeypatch, break_reporting
):
    """Without the broadcast, every connected client still shows the session as
    busy until someone reloads."""
    from tsugite_daemon import session_runner

    adapter, ctx = _adapter_with_unwritable_store(tmp_path, monkeypatch)
    if break_reporting == "unimportable":
        monkeypatch.delattr(session_runner, "report_send_failure")
    broadcast_for: list = []
    monkeypatch.setattr(adapter, "_broadcast_turn_complete", broadcast_for.append)

    async def fake_inner(*_args, **kwargs):
        kwargs["_broadcast_state"]["conv_id"] = "s1"
        return "ok"

    monkeypatch.setattr(adapter, "_handle_message_inner", fake_inner)

    with contextlib.suppress(ImportError):
        await adapter.handle_message("u1", "hi", ctx)

    assert broadcast_for == ["s1"]
