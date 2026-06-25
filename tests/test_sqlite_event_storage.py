"""SqliteSession + SqliteHistoryBackend: the event-log battery on SQLite.

Mirrors the JSONL SessionStorage contract (record / iter / load / summary / create /
load / exists / get_meta / list_sessions) and pins the two load-bearing invariants:
- summary() read from maintained columns equals SessionSummary.from_events over the
  same event stream (fold-on-write parity);
- events round-trip as real Event objects, so reconstruction.events_to_messages works
  unchanged against a SQLite-sourced log.
"""

from datetime import datetime, timedelta, timezone

import pytest

from tsugite.history.models import Event, iso_utc
from tsugite.history.reconstruction import events_to_messages
from tsugite.history.sqlite_backend import SessionAlreadyExistsError, SqliteHistoryBackend
from tsugite.history.sqlite_conn import close_all
from tsugite.history.storage import SessionSummary

T0 = datetime(2026, 6, 22, 10, 0, 0, tzinfo=timezone.utc)


def _ev(type_, *, offset=0, **data):
    return Event(type=type_, ts=T0 + timedelta(seconds=offset), data=data)


@pytest.fixture
def backend(tmp_path):
    b = SqliteHistoryBackend(db_path=tmp_path / "history.db")
    yield b
    close_all()


def test_create_records_session_start_first(backend):
    s = backend.create("chat", "openai:gpt-4o-mini", workspace="w")
    events = s.load_events()
    assert events[0].type == "session_start"
    assert events[0].data["agent"] == "chat"
    assert events[0].data["workspace"] == "w"


def test_record_and_iter_in_order_with_ids(backend):
    s = backend.create("chat", "m")
    s.record("user_input", ts=T0, text="hi")
    s.record("model_response", ts=T0 + timedelta(seconds=1), raw_content="yo")
    types = [e.type for e in s.iter_events()]
    assert types == ["session_start", "user_input", "model_response"]
    ids = [e.id for e in s.iter_events()]
    assert ids == sorted(ids) and all(isinstance(i, int) for i in ids)
    assert [e.type for e in s.iter_events(types=["user_input"])] == ["user_input"]


def test_record_many_batches(backend):
    s = backend.create("chat", "m")
    s.record_many([_ev("user_input", text="a"), _ev("user_input", offset=1, text="b")])
    assert sum(1 for e in s.iter_events() if e.type == "user_input") == 2


def test_summary_matches_from_events(backend):
    s = backend.create("chat", "openai:gpt-4o-mini", workspace="w")
    stream = [
        _ev("user_input", offset=1, text="q1"),
        _ev("model_response", offset=2, raw_content="r1", usage={"total_tokens": 30}, cost=0.01),
        _ev("code_execution", offset=3, duration_ms=120, tools_called=["read_file"]),
        _ev("tool_invocation", offset=4, name="grep", duration_ms=5),
        _ev("user_input", offset=5, text="q2"),
        _ev("model_response", offset=6, raw_content="final", usage={"total_tokens": 12}, cost=0.02),
        _ev("session_end", offset=7, status="success"),
    ]
    s.record_many(stream)

    got = s.summary()
    # Oracle: from_events over the *full* persisted stream (session_start + the rest).
    expected = SessionSummary.from_events(s.iter_events())

    for field in ("agent", "model", "workspace", "status", "turn_count", "total_tokens", "total_duration_ms"):
        assert getattr(got, field) == getattr(expected, field), field
    assert got.total_cost == pytest.approx(expected.total_cost)
    assert got.functions_called == expected.functions_called == {"read_file", "grep"}
    assert got.last_response_text == "final"


def test_recency_columns_split_wall_clock_from_event_ts(backend):
    # Simulate copy-forward: a session whose events are all old (2020), materialized now.
    old = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    s = backend.create("chat", "m", timestamp=old)
    s.record("user_input", ts=old + timedelta(seconds=5), text="copied event")
    row = (
        backend._conn()
        .execute("SELECT updated_at, last_event_ts FROM sessions WHERE session_id=?", (s.session_id,))
        .fetchone()
    )
    # last_event_ts tracks the (old) events; updated_at is wall-clock now, far newer.
    assert row["last_event_ts"] == iso_utc(old + timedelta(seconds=5))
    assert row["updated_at"] > row["last_event_ts"]


def test_reconstruction_parity_with_jsonl(backend):
    s = backend.create("chat", "m")
    s.record("user_input", ts=T0, text="hello")
    s.record("model_response", ts=T0 + timedelta(seconds=1), raw_content="hi there")
    messages = events_to_messages(s.load_events())
    assert {"role": "assistant", "content": "hi there"} in messages
    assert any(m["role"] == "user" and "hello" in m["content"] for m in messages)


def test_explicit_existing_id_raises(backend):
    backend.create("chat", "m", session_id="fixed-1")
    with pytest.raises(SessionAlreadyExistsError):
        backend.create("chat", "m", session_id="fixed-1")


def test_generated_id_collision_regenerates(backend):
    a = backend.create("chat", "m", timestamp=T0)
    b = backend.create("chat", "m", timestamp=T0)  # identical inputs -> same base id
    assert a.session_id != b.session_id


def test_load_missing_raises(backend):
    with pytest.raises(FileNotFoundError):
        backend.load("nope")


def test_get_meta_and_exists(backend):
    s = backend.create("chat", "m")
    assert backend.exists(s.session_id)
    assert not backend.exists("nope")
    meta = backend.get_meta(s.session_id)
    assert meta.type == "session_start" and meta.data["agent"] == "chat"


def test_list_sessions_recency_order(backend):
    backend.create("chat", "m")  # earlier session
    b = backend.create("chat", "m")
    b.record("user_input", text="touch b last")
    assert backend.list_sessions()[0] == b.session_id
