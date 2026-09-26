"""LoggingProgressHandler persists live-only events for job and queued sessions.

An event the agent already records durably must only be broadcast here, or the
session history holds it twice.
"""

import pytest
from tsugite_daemon.session_runner import LoggingProgressHandler
from tsugite_daemon.session_store import Session, SessionSource, SessionStore

from tsugite.events.events import (
    CodeExecutionEvent,
    ModelResponseEvent,
    ObservationEvent,
    PromptSnapshotEvent,
    StepStartEvent,
)


class _Bus:
    def __init__(self):
        self.types: list[str] = []

    def emit(self, name, payload):
        self.types.append(payload["event_type"])


@pytest.fixture
def store(tmp_path, history_dir):
    store = SessionStore(tmp_path / "store.json")
    store.create_session(Session(id="s1", source=SessionSource.BACKGROUND.value, user_id="alice"))
    return store


@pytest.mark.parametrize(
    "event,expected_type",
    [
        (ModelResponseEvent(thought="done"), "model_response"),
        (PromptSnapshotEvent(token_breakdown={"total": 10}), "prompt_snapshot"),
        (CodeExecutionEvent(code="print(1)"), "code"),
        (ObservationEvent(observation="1"), "tool_result"),
    ],
)
def test_events_with_a_durable_record_are_broadcast_not_persisted(store, event, expected_type):
    bus = _Bus()
    before = len(store.read_events("s1"))

    LoggingProgressHandler(store, "s1", broadcaster=bus).handle_event(event)

    assert len(store.read_events("s1")) == before
    assert bus.types == [expected_type]


def test_live_only_events_are_still_persisted(store):
    before = len(store.read_events("s1"))

    LoggingProgressHandler(store, "s1").handle_event(StepStartEvent(step=1, max_turns=5))

    assert [e["type"] for e in store.read_events("s1")[before:]] == ["turn_start"]
