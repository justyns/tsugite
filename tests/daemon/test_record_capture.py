"""Tests for the tsugite-daemon session / job context captures.

The daemon registers two capture-only providers (``session`` and ``job``) that
build a compact summary for a given record id, triggered from the web UI by an
explicit action (a button or a reference paste), never by scanning a message.
These tests wire fakes into the two seams the captures read (history backend,
jobs orchestrator) and call ``capture_session`` / ``capture_job`` directly, so
nothing needs a reinstall for the entry point.
"""

from __future__ import annotations

import importlib
from datetime import datetime, timezone

from tsugite_daemon import context as id_ctx
from tsugite_daemon.job_store import Job

from tsugite.context import get_context_provider, reset_context_providers, run_capture
from tsugite.history import SessionSummary
from tsugite.history.models import Event

SESSION_ID = "20260722_042329_odyn_85fc3c"
JOB_ID = "job-1a2b3c4d"


def _ev(type_: str, **data) -> Event:
    return Event(type=type_, ts=datetime.now(timezone.utc), data=data)


def _session_events() -> list[Event]:
    return [
        _ev("session_start", agent="odyn", model="claude_code:opus"),
        _ev("user_input", text="Investigate the flaky test in the parser"),
        _ev("model_response", raw_content="I looked at parser.py and found the bug", usage={"total_tokens": 1200}),
        _ev("user_input", text="Now fix it and add a regression test"),
        _ev("model_response", raw_content="Done, patched _tokenize and added test_flaky", usage={"total_tokens": 900}),
        _ev("session_end", status="success"),
    ]


class _FakeSession:
    def __init__(self, events: list[Event]):
        self._events = events
        self.session_id = SESSION_ID

    def iter_events(self, types=None):
        wanted = set(types) if types is not None else None
        for event in self._events:
            if wanted is None or event.type in wanted:
                yield event

    def summary(self) -> SessionSummary:
        return SessionSummary.from_events(self._events)


class _FakeBackend:
    def __init__(self, sessions: dict[str, list[Event]]):
        self._sessions = sessions

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def load(self, session_id: str) -> _FakeSession:
        return _FakeSession(self._sessions[session_id])


class _FakeOrchestrator:
    def __init__(self, jobs: list[Job]):
        self._by_id = {j.id: j for j in jobs}

    def get_job(self, job_id: str):
        return self._by_id.get(job_id)


def _wire(monkeypatch, *, backend=None, orchestrator=None):
    """Point the captures' two seams at fakes; unset ones default to empty."""
    monkeypatch.setattr(id_ctx, "get_history_backend", lambda: backend or _FakeBackend({}))
    monkeypatch.setattr(id_ctx, "get_jobs_orchestrator", lambda: orchestrator)


def test_capture_session_returns_compact_summary(monkeypatch):
    _wire(monkeypatch, backend=_FakeBackend({SESSION_ID: _session_events()}))

    items = id_ctx.capture_session(SESSION_ID, {})

    assert len(items) == 1
    item = items[0]
    assert item.key == f"session:{SESSION_ID}"
    assert not item.untrusted  # the user's own session -> trusted context
    value = item.value
    assert "title: Investigate the flaky test in the parser" in value
    assert "status: completed" in value
    assert "model: claude_code:opus" in value
    assert "messages: 2" in value
    assert "tokens: 2100" in value
    assert "last user: Now fix it and add a regression test" in value
    assert "last assistant: Done, patched _tokenize and added test_flaky" in value
    # A compact summary, NOT a transcript: the earlier assistant turn is dropped.
    assert "I looked at parser.py" not in value


def test_capture_session_reports_active_status(monkeypatch):
    events = [_ev("session_start", agent="odyn", model="m"), _ev("user_input", text="hi")]
    _wire(monkeypatch, backend=_FakeBackend({SESSION_ID: events}))

    value = id_ctx.capture_session(SESSION_ID, {})[0].value

    assert "status: active" in value


def test_capture_session_unknown_id_returns_empty(monkeypatch):
    _wire(monkeypatch, backend=_FakeBackend({}))

    assert id_ctx.capture_session(SESSION_ID, {}) == []


def test_capture_session_empty_arg_returns_empty(monkeypatch):
    _wire(monkeypatch, backend=_FakeBackend({SESSION_ID: _session_events()}))

    assert id_ctx.capture_session(None, {}) == []
    assert id_ctx.capture_session("", {}) == []


def test_capture_session_value_is_capped(monkeypatch):
    events = [
        _ev("session_start", agent="odyn", model="m"),
        _ev("user_input", text="hi"),
        _ev("model_response", raw_content="z" * 10000, usage={"total_tokens": 1}),
    ]
    _wire(monkeypatch, backend=_FakeBackend({SESSION_ID: events}))

    value = id_ctx.capture_session(SESSION_ID, {})[0].value

    assert len(value) <= id_ctx._MAX_VALUE_CHARS
    # The last reply is previewed, not dumped whole.
    assert "z" * id_ctx._PREVIEW_CHARS in value
    assert "z" * (id_ctx._PREVIEW_CHARS + 1) not in value


def test_capture_job_returns_item(monkeypatch):
    job = Job(
        id=JOB_ID,
        parent_session_id="p",
        prompt="Add rate limiting to the API",
        state="running",
        executor="agent",
        verify_attempts=1,
        max_attempts=3,
    )
    _wire(monkeypatch, orchestrator=_FakeOrchestrator([job]))

    items = id_ctx.capture_job(JOB_ID, {})

    assert len(items) == 1
    item = items[0]
    assert item.key == f"job:{JOB_ID}"
    assert not item.untrusted
    assert "state: running" in item.value
    assert "executor: agent" in item.value
    assert "attempts: 1/3" in item.value
    assert "prompt: Add rate limiting to the API" in item.value


def test_capture_job_includes_last_error_first_line(monkeypatch):
    job = Job(
        id=JOB_ID,
        parent_session_id="p",
        prompt="x",
        state="errored",
        executor="cc",
        error="worker died (code 1)\ntraceback frame two",
    )
    _wire(monkeypatch, orchestrator=_FakeOrchestrator([job]))

    value = id_ctx.capture_job(JOB_ID, {})[0].value

    assert "last error: worker died (code 1)" in value
    assert "traceback frame two" not in value


def test_capture_job_unknown_id_returns_empty(monkeypatch):
    _wire(monkeypatch, orchestrator=_FakeOrchestrator([]))

    assert id_ctx.capture_job(JOB_ID, {}) == []


def test_capture_job_empty_arg_returns_empty(monkeypatch):
    job = Job(id=JOB_ID, parent_session_id="p", prompt="p", state="done", executor="agent")
    _wire(monkeypatch, orchestrator=_FakeOrchestrator([job]))

    assert id_ctx.capture_job(None, {}) == []
    assert id_ctx.capture_job("", {}) == []


def test_capture_job_orchestrator_unwired_returns_empty(monkeypatch):
    _wire(monkeypatch, orchestrator=None)  # no daemon -> no orchestrator

    assert id_ctx.capture_job(JOB_ID, {}) == []


def test_registers_capture_only_session_and_job_providers(monkeypatch):
    reset_context_providers()
    importlib.reload(id_ctx)

    for key, label in (("session", "Session"), ("job", "Job")):
        provider = get_context_provider(key)
        assert provider is not None
        assert provider.label == label
        assert provider.capture is not None
        assert provider.detect is None
        # Capture-only: an explicit action runs it, so it stays out of the menu.
        assert not provider.in_menu

    # Still reachable through the registry's capture path despite menu=False.
    monkeypatch.setattr(id_ctx, "get_history_backend", lambda: _FakeBackend({SESSION_ID: _session_events()}))
    items = run_capture("session", SESSION_ID, {})
    assert [i.key for i in items] == [f"session:{SESSION_ID}"]
