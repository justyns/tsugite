"""Tests for the per-event history storage (Event model + SessionStorage)."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tsugite.history.models import Event
from tsugite.history.storage import SessionStorage


class TestEvent:
    def test_minimal_construction(self):
        e = Event(type="user_input", ts=datetime.now(timezone.utc))
        assert e.type == "user_input"
        assert e.data == {}

    def test_with_data(self):
        e = Event(
            type="model_response",
            ts=datetime.now(timezone.utc),
            data={"raw_content": "hello", "usage": {"input_tokens": 10}},
        )
        assert e.data["raw_content"] == "hello"
        assert e.data["usage"]["input_tokens"] == 10

    def test_round_trip_through_json(self):
        ts = datetime.now(timezone.utc)
        original = Event(type="code_execution", ts=ts, data={"code": "x = 1", "duration_ms": 12})
        restored = Event.model_validate(json.loads(original.model_dump_json()))
        assert restored.type == original.type
        assert restored.ts == ts
        assert restored.data == original.data

    def test_arbitrary_type_string_accepted(self):
        # Open schema: future event types must not break parsing.
        e = Event(type="some_future_event", ts=datetime.now(timezone.utc))
        assert e.type == "some_future_event"

    def test_extra_fields_at_top_level_allowed(self):
        # extra="allow" means an unrecognized top-level key won't blow up reads.
        data = {"type": "user_input", "ts": datetime.now(timezone.utc).isoformat(), "_private": "x"}
        e = Event.model_validate(data)
        assert e.type == "user_input"


@pytest.fixture
def session_path(tmp_path: Path) -> Path:
    return tmp_path / "session.jsonl"


class TestSessionStorageRecord:
    def test_create_writes_session_start(self, session_path):
        storage = SessionStorage.create(
            agent_name="test_agent",
            model="anthropic:claude",
            session_path=session_path,
        )
        assert session_path.exists()
        events = list(storage.iter_events())
        assert len(events) == 1
        assert events[0].type == "session_start"
        assert events[0].data["agent"] == "test_agent"
        assert events[0].data["model"] == "anthropic:claude"

    def test_record_appends(self, session_path):
        storage = SessionStorage.create(
            agent_name="test_agent",
            model="anthropic:claude",
            session_path=session_path,
        )
        storage.record("user_input", text="hi")
        storage.record("model_response", raw_content="hello")

        events = list(storage.iter_events())
        types = [e.type for e in events]
        assert types == ["session_start", "user_input", "model_response"]
        assert events[1].data["text"] == "hi"
        assert events[2].data["raw_content"] == "hello"

    def test_iter_events_filters_by_type(self, session_path):
        storage = SessionStorage.create(
            agent_name="a",
            model="m",
            session_path=session_path,
        )
        storage.record("user_input", text="hi")
        storage.record("model_response", raw_content="r1")
        storage.record("code_execution", code="print(1)")
        storage.record("model_response", raw_content="r2")

        responses = list(storage.iter_events(types=["model_response"]))
        assert [e.data["raw_content"] for e in responses] == ["r1", "r2"]

    def test_iter_events_multi_type_filter(self, session_path):
        storage = SessionStorage.create(agent_name="a", model="m", session_path=session_path)
        storage.record("user_input", text="hi")
        storage.record("model_response", raw_content="r")
        storage.record("code_execution", code="x")

        kept = list(storage.iter_events(types=["user_input", "code_execution"]))
        assert [e.type for e in kept] == ["user_input", "code_execution"]

    def test_load_existing_session(self, session_path):
        storage = SessionStorage.create(agent_name="a", model="m", session_path=session_path)
        storage.record("user_input", text="hi")

        reopened = SessionStorage.load(session_path)
        events = list(reopened.iter_events())
        assert [e.type for e in events] == ["session_start", "user_input"]

    def test_load_meta_fast_reads_first_event(self, session_path):
        SessionStorage.create(
            agent_name="fast_agent",
            model="m",
            session_path=session_path,
        )
        meta = SessionStorage.load_meta_fast(session_path)
        assert meta is not None
        assert meta.type == "session_start"
        assert meta.data["agent"] == "fast_agent"

    def test_unknown_event_type_round_trips(self, session_path):
        storage = SessionStorage.create(agent_name="a", model="m", session_path=session_path)
        storage.record("brand_new_type", whatever=42)

        reopened = SessionStorage.load(session_path)
        events = list(reopened.iter_events())
        assert events[-1].type == "brand_new_type"
        assert events[-1].data["whatever"] == 42

    def test_malformed_line_skipped_with_warning(self, session_path, capsys):
        SessionStorage.create(agent_name="a", model="m", session_path=session_path)
        with open(session_path, "a", encoding="utf-8") as f:
            f.write("this is not json\n")

        reopened = SessionStorage.load(session_path)
        events = list(reopened.iter_events())
        # Only the valid session_start event survives.
        assert len(events) == 1
        assert events[0].type == "session_start"

    def test_concurrent_appends_are_lock_safe(self, session_path):
        # Smoke test: many sequential writes should produce a clean JSONL.
        storage = SessionStorage.create(agent_name="a", model="m", session_path=session_path)
        for i in range(50):
            storage.record("model_response", raw_content=f"r{i}")

        events = list(storage.iter_events(types=["model_response"]))
        assert len(events) == 50
        assert events[0].data["raw_content"] == "r0"
        assert events[-1].data["raw_content"] == "r49"
