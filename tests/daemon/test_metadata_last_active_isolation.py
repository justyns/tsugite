"""Metadata-only updates must not bump `last_active`.

Issue #313: agents emit `session_metadata(key='status_text', ...)` and similar
housekeeping calls many times per turn. Bumping `last_active` on every one
defeats the post-`mark-viewed` clear because the unread computation in
`http.py:741` reads `last_active > last_viewed_at`. The sidebar badge then
re-appears moments after stream end via the `metadata_updated` SSE → re-fetch
loop. The fix: keep `last_active` driven by message-level activity, not by
metadata housekeeping.
"""

from datetime import datetime, timedelta, timezone

import pytest

from tsugite.daemon.session_store import Session, SessionSource, SessionStore


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "store.json")


def _make_session(store: SessionStore, user_id: str = "u1") -> Session:
    s = Session(id="sess-1", agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id=user_id)
    store.create_session(s)
    return s


def test_set_metadata_does_not_bump_last_active(store):
    s = _make_session(store)
    before = s.last_active

    store.set_metadata("sess-1", "status_text", "idle")

    assert store._sessions["sess-1"].last_active == before, (
        "set_metadata bumped last_active; metadata housekeeping shouldn't count as new activity"
    )


def test_set_metadata_bulk_does_not_bump_last_active(store):
    s = _make_session(store)
    before = s.last_active

    store.set_metadata_bulk("sess-1", {"status_text": "thinking", "topic": "fixing bugs"})

    assert store._sessions["sess-1"].last_active == before, (
        "set_metadata_bulk bumped last_active; metadata housekeeping shouldn't count as new activity"
    )


def test_delete_metadata_does_not_bump_last_active(store):
    _make_session(store)
    store.set_metadata("sess-1", "task", "task-123")
    before = store._sessions["sess-1"].last_active

    store.delete_metadata("sess-1", "task")

    assert store._sessions["sess-1"].last_active == before, (
        "delete_metadata bumped last_active; metadata housekeeping shouldn't count as new activity"
    )


def test_message_activity_still_bumps_last_active(store):
    """Sanity check: real message activity (update_token_count) must still bump
    last_active, otherwise we'd break unread tracking for actual replies.
    """
    _make_session(store)
    past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    store._sessions["sess-1"].last_active = past

    store.update_token_count("sess-1", 100)

    assert store._sessions["sess-1"].last_active > past, (
        "update_token_count must keep bumping last_active so genuine new activity surfaces in the UI"
    )


def test_unread_does_not_revive_after_mark_viewed_then_metadata_update(store):
    """End-to-end of the issue: after `mark-viewed`, a metadata update for the
    same session must not re-flip `last_active` past `last_viewed_at` and make
    the unread re-derivation flip back to True.
    """
    _make_session(store)
    # Simulate: user just finished a turn (last_active advanced) and the
    # stream-end mark-viewed handler stamped last_viewed_at to the current time.
    store._sessions["sess-1"].last_active = "2026-05-18T10:00:00+00:00"
    store._sessions["sess-1"].last_viewed_at = "2026-05-18T10:00:01+00:00"

    store.set_metadata("sess-1", "status_text", "idle")

    sess = store._sessions["sess-1"]
    unread = bool(sess.last_active and (not sess.last_viewed_at or sess.last_active > sess.last_viewed_at))
    assert not unread, (
        "metadata update revived the unread flag — sidebar will flash bold again after the user has already seen the reply"
    )
