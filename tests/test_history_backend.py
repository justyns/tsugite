"""Phase 1a: the history store sits behind a swappable HistoryBackend seam."""

import pytest

from tsugite.history import (
    Event,
    HistoryBackend,
    Session,
    get_history_backend,
    reset_history_backend,
    set_history_backend,
)


@pytest.fixture(autouse=True)
def _reset_backend():
    reset_history_backend()
    yield
    reset_history_backend()


def test_default_backend_is_jsonl_and_round_trips(isolate_config_files):
    backend = get_history_backend()
    assert isinstance(backend, HistoryBackend)

    session = backend.create(agent_name="tester", model="test:model")
    assert isinstance(session, Session)
    sid = session.session_id
    session.record("user_input", text="hello")

    assert backend.exists(sid)
    assert sid in backend.list_sessions()

    meta = backend.get_meta(sid)
    assert meta is not None and meta.data.get("agent") == "tester"

    loaded = backend.load(sid)
    types = [e.type for e in loaded.load_events()]
    assert types == ["session_start", "user_input"]


class _DummyBackend:
    """Minimal in-memory backend used to prove the seam routes away from jsonl."""

    def __init__(self):
        self.sessions: dict[str, "_DummySession"] = {}

    def create(self, agent_name, model, *, workspace=None, parent_session=None, session_id=None, timestamp=None):
        sid = session_id or f"dummy-{len(self.sessions)}"
        s = _DummySession(sid)
        s.record("session_start", agent=agent_name, model=model)
        self.sessions[sid] = s
        return s

    def load(self, session_id):
        return self.sessions[session_id]

    def exists(self, session_id):
        return session_id in self.sessions

    def get_meta(self, session_id):
        events = self.sessions[session_id].events
        return events[0] if events else None

    def list_sessions(self):
        return list(self.sessions)


class _DummySession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.events: list[Event] = []

    def record(self, type, *, ts=None, **data):
        from datetime import datetime, timezone

        self.events.append(
            Event(type=type, ts=ts or datetime.now(timezone.utc), data={k: v for k, v in data.items() if v is not None})
        )

    def record_many(self, events):
        self.events.extend(events)

    def iter_events(self, types=None):
        wanted = set(types) if types is not None else None
        return (e for e in self.events if wanted is None or e.type in wanted)

    def load_events(self):
        return list(self.events)

    def summary(self):
        from tsugite.history import SessionSummary

        return SessionSummary.from_events(self.events)


def test_set_backend_routes_sessions_to_it():
    dummy = _DummyBackend()
    set_history_backend(dummy)

    backend = get_history_backend()
    assert backend is dummy

    session = backend.create(agent_name="x", model="m", session_id="abc")
    assert dummy.sessions["abc"] is session
    assert isinstance(session, Session)


def test_backend_resolves_named_plugin_from_config(monkeypatch):
    """config.history.backend selects a backend registered under tsugite.history."""
    import tsugite.history.registry as registry
    import tsugite.plugins as plugins
    from tsugite.config import Config, HistoryConfig

    dummy = _DummyBackend()

    class _EP:
        name = "memory"

        def load(self):
            return lambda config: dummy

    monkeypatch.setattr(registry, "load_config", lambda: Config(history=HistoryConfig(backend="memory")))
    monkeypatch.setattr(
        plugins.importlib.metadata, "entry_points", lambda group: [_EP()] if group == "tsugite.history" else []
    )

    reset_history_backend()
    assert get_history_backend() is dummy
