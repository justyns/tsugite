"""Regression guard for issue #300: every per-session SSE event payload carries session_id.

The daemon broadcasts a set of events that the web UI scopes to a specific
session. Each of those payloads must include `session_id` so the client can
gate on it, otherwise events for session A wake viewers of B.

This file enumerates each known per-session emit site and asserts the field is
present in the payload that reaches subscribers.
"""

import pytest
from tsugite_daemon.adapters.base import BaseAdapter
from tsugite_daemon.config import AgentConfig
from tsugite_daemon.session_runner import set_current_session_id
from tsugite_daemon.session_store import SessionStore

from tsugite.events import EventBus, SkillLoadedEvent, SkillUnloadedEvent
from tsugite.events.helpers import emit_skill_loaded_event, emit_skill_unloaded_event
from tsugite.ui.jsonl import JSONLUIHandler
from tsugite.ui_context import set_ui_context


class _StubAdapter(BaseAdapter):
    async def start(self):
        pass

    async def stop(self):
        pass


class _RecordingBus:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def emit(self, event_type, payload) -> None:
        self.events.append((event_type, dict(payload)))


@pytest.fixture
def adapter(tmp_path):
    store = SessionStore(tmp_path / "session_store.json", context_limits={"test-agent": 128_000})
    return _StubAdapter("test-agent", AgentConfig(workspace_dir=tmp_path / "workspace", agent_file="default"), store)


def test_history_update_from_broadcast_turn_complete_has_session_id(adapter):
    bus = _RecordingBus()
    adapter.event_bus = bus

    adapter._broadcast_turn_complete("sess-A")

    history_updates = [p for t, p in bus.events if t == "history_update"]
    assert history_updates, "history_update was not broadcast"
    assert history_updates[0].get("session_id") == "sess-A"


def test_history_update_skipped_when_conv_id_missing(adapter):
    """A None conv_id is the no-session-yet path; we don't fabricate a session_id."""
    bus = _RecordingBus()
    adapter.event_bus = bus

    adapter._broadcast_turn_complete(None)

    history_updates = [p for t, p in bus.events if t == "history_update"]
    assert history_updates, "history_update was not broadcast"
    assert "session_id" not in history_updates[0]


def test_skill_loaded_helper_picks_up_session_from_contextvar():
    """`emit_skill_loaded_event` resolves session_id from the daemon contextvar."""
    bus = EventBus()
    captured: list = []
    bus.subscribe(captured.append)
    set_ui_context(event_bus=bus)
    try:
        set_current_session_id("sess-XYZ")
        emit_skill_loaded_event(skill_name="foo", description="bar")
    finally:
        set_current_session_id(None)
        set_ui_context()

    skill_events = [e for e in captured if isinstance(e, SkillLoadedEvent)]
    assert len(skill_events) == 1
    assert skill_events[0].session_id == "sess-XYZ"
    assert skill_events[0].skill_name == "foo"


def test_skill_unloaded_helper_picks_up_session_from_contextvar():
    bus = EventBus()
    captured: list = []
    bus.subscribe(captured.append)
    set_ui_context(event_bus=bus)
    try:
        set_current_session_id("sess-XYZ")
        emit_skill_unloaded_event(skill_name="foo")
    finally:
        set_current_session_id(None)
        set_ui_context()

    skill_events = [e for e in captured if isinstance(e, SkillUnloadedEvent)]
    assert len(skill_events) == 1
    assert skill_events[0].session_id == "sess-XYZ"


def test_skill_loaded_sse_payload_includes_session_id():
    """Round-trip through JSONLUIHandler — the SSE payload (what tabs receive)
    must carry session_id, not just the in-process event object."""
    handler = JSONLUIHandler()
    emitted: list[tuple[str, dict]] = []
    handler._emit = lambda event_type, payload: emitted.append((event_type, payload))

    handler._handle_skill_loaded(SkillLoadedEvent(skill_name="foo", description="bar", session_id="sess-XYZ"))
    handler._handle_skill_unloaded(SkillUnloadedEvent(skill_name="foo", session_id="sess-XYZ"))

    by_type = dict(emitted)
    assert by_type["skill_loaded"]["session_id"] == "sess-XYZ"
    assert by_type["skill_unloaded"]["session_id"] == "sess-XYZ"


def test_no_session_id_field_absent_when_none():
    """When the contextvar isn't set, the SSE payload omits session_id (rather
    than emitting `session_id: null`) so non-daemon callers don't pollute the wire."""
    handler = JSONLUIHandler()
    emitted: list[tuple[str, dict]] = []
    handler._emit = lambda event_type, payload: emitted.append((event_type, payload))

    handler._handle_skill_loaded(SkillLoadedEvent(skill_name="foo", description="bar"))
    handler._handle_skill_unloaded(SkillUnloadedEvent(skill_name="foo"))

    by_type = dict(emitted)
    assert "session_id" not in by_type["skill_loaded"]
    assert "session_id" not in by_type["skill_unloaded"]
