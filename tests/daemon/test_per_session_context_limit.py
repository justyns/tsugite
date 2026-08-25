"""Per-session context_limit storage.

Issue #315 Fix 1: context_limit is naturally per-session (each session's model
can differ via `model_override`, and even within the same model the provider's
reported window can drift). Storing it agent-wide means any code path that
updates it can clobber another session's value. Moving the storage to the
`Session` dataclass eliminates that whole bug class — sessions cannot leak into
each other through a shared scalar.
"""

import pytest
from tsugite_daemon.session_store import Session, SessionSource, SessionStore


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "store.json", default_context_limit=128_000)


def _make_session(store: SessionStore, sid: str, agent: str = "agent-a") -> Session:
    s = Session(id=sid, source=SessionSource.INTERACTIVE.value, user_id="u1")
    store.create_session(s)
    return s


def test_session_context_limit_falls_back_to_agent_default(store):
    """A fresh session has no per-session limit, so it inherits the agent-wide
    default until the first turn reports a window.
    """
    _make_session(store, "s1")
    assert store.get_session_context_limit("s1") == 128_000


def test_session_context_limit_is_isolated_between_sessions(store):
    _make_session(store, "s1")
    _make_session(store, "s2")

    store.update_session_context_limit("s1", 1_000_000)

    assert store.get_session_context_limit("s1") == 1_000_000
    assert store.get_session_context_limit("s2") == 128_000, (
        "setting s1's limit must not affect s2; that's the whole point of moving to per-session storage"
    )


def test_session_context_limit_survives_agent_wide_mutation(store):
    """The original #315 scenario: a compaction-model call mutates the agent-wide
    limit. After per-session migration, that mutation has no effect on a session
    that already has its own tracked limit.
    """
    _make_session(store, "s1")
    store.update_session_context_limit("s1", 1_000_000)

    store.update_context_limit(200_000)

    assert store.get_session_context_limit("s1") == 1_000_000, (
        "agent-wide context_limit mutation should not clobber a session's tracked limit"
    )


def test_needs_compaction_uses_per_session_limit(store):
    """Per-session limit must drive `needs_compaction`. Otherwise a session on a
    1M model with 700k tokens would falsely trigger compaction because the
    agent-wide default (128k) says it's over threshold.
    """
    _make_session(store, "big")
    _make_session(store, "small")
    store.update_session_context_limit("big", 1_000_000)

    store._sessions["big"].cumulative_tokens = 700_000
    store._sessions["small"].cumulative_tokens = 110_000

    assert not store.needs_compaction("big"), (
        "1M-window session at 700k must not trigger compaction (threshold = 80% = 800k)"
    )
    assert store.needs_compaction("small"), (
        "128k-window session at 110k must trigger compaction (threshold = 80% = 102.4k)"
    )


def test_compacted_session_inherits_predecessor_limit(store):
    _make_session(store, "s1")
    store.update_session_context_limit("s1", 1_000_000)
    store._sessions["s1"].message_count = 1  # compact_session requires non-empty

    new_session = store.compact_session("s1")

    assert store.get_session_context_limit(new_session.id) == 1_000_000, (
        "post-compaction successor must inherit the predecessor's tracked window — the agent's model didn't change"
    )


def test_per_session_limit_persists_across_store_reload(store, tmp_path):
    _make_session(store, "s1")
    store.update_session_context_limit("s1", 1_000_000)

    reloaded = SessionStore(store._path, default_context_limit=128_000)

    assert reloaded.get_session_context_limit("s1") == 1_000_000


def test_session_detail_exposes_resolved_context_limit(store):
    """A fresh session's raw `context_limit` is None until the first turn reports a
    provider window, but the detail payload must ALSO carry the RESOLVED limit
    (agent-default fallback) so the web UI paints a context meter from session open
    instead of showing nothing. The raw field keeps its meaning (unset == None).
    """
    _make_session(store, "s1")

    detail = store.session_detail("s1")
    assert detail["context_limit"] is None, "raw per-session field stays None pre-first-turn"
    assert detail["context_limit_resolved"] == 128_000, (
        "resolved limit falls back to the agent default so a fresh session still gets a meter"
    )

    store.update_session_context_limit("s1", 1_000_000)
    detail = store.session_detail("s1")
    assert detail["context_limit"] == 1_000_000, "raw field now carries the tracked window"
    assert detail["context_limit_resolved"] == 1_000_000, "resolved tracks the per-session window once set"
