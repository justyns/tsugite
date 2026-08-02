"""Daemon UI events should write to the same per-session JSONL as conversation
history — there shouldn't be a separate `daemon/sessions/` log."""

import json
from pathlib import Path

import pytest
from tsugite_daemon.session_store import SessionStore

from tests.history_helpers import load_history_session, seed_history_session
from tsugite.history import Session, get_history_backend


@pytest.fixture
def store(tmp_path: Path, history_dir):
    return SessionStore(tmp_path / "session_store.json")


def _make_history_session(history_dir: Path, session_id: str) -> Session:
    """Pre-create a history file so append_event has a target."""
    return seed_history_session(session_id, agent="test", model="m")


def test_append_event_writes_to_history_file(store, history_dir):
    sid = "test-session"
    _make_history_session(history_dir, sid)
    store.append_event(sid, {"type": "reaction", "emoji": "👍", "timestamp": "2026-01-01T00:00:00+00:00"})

    storage = load_history_session(sid)
    types = [e.type for e in storage.iter_events()]
    assert "reaction" in types
    reaction = next(e for e in storage.iter_events() if e.type == "reaction")
    assert reaction.data["emoji"] == "👍"


def test_no_separate_daemon_sessions_dir(store, history_dir, tmp_path):
    """The legacy daemon/sessions/ directory should not be created any more."""
    sid = "another-session"
    _make_history_session(history_dir, sid)
    store.append_event(sid, {"type": "prompt_snapshot", "token_breakdown": {"total": 100}})
    assert not (tmp_path / "sessions").exists()


def test_read_events_returns_flat_dict_shape(store, history_dir):
    """Consumers iterate `e.get('type')`, `e.get('emoji')` etc. — preserve that shape."""
    sid = "shape-test"
    _make_history_session(history_dir, sid)
    store.append_event(sid, {"type": "reaction", "emoji": "🎉"})

    events = store.read_events(sid)
    assert any(e.get("type") == "reaction" and e.get("emoji") == "🎉" for e in events)


def test_event_count_matches_appends(store, history_dir):
    sid = "count-test"
    _make_history_session(history_dir, sid)
    for _ in range(3):
        store.append_event(sid, {"type": "prompt_snapshot", "token_breakdown": {"total": 1}})
    # session_start + 3 appends
    assert store.event_count(sid) == 4


def test_append_event_handles_missing_session_gracefully(store, history_dir):
    """If the session doesn't exist yet, append_event should create it."""

    sid = "fresh-session"
    store.append_event(sid, {"type": "reaction", "emoji": "✨"})

    assert get_history_backend().exists(sid)
    assert [e.type for e in get_history_backend().load(sid).load_events()] == ["reaction"]


class TestExtendedMigration:
    """The migrate command should also pick up workspace archives and old
    daemon/sessions/ files."""

    def test_migrate_workspace_archives(self, tmp_path):
        from tsugite.cli.history import migrate_path

        workspaces = tmp_path / "workspaces"
        ws = workspaces / "demo" / "sessions"
        ws.mkdir(parents=True)
        old_file = ws / "20260101-120000.jsonl"
        old_file.write_text(
            json.dumps(
                {
                    "type": "session_meta",
                    "agent": "demo",
                    "model": "m",
                    "machine": "h",
                    "created_at": "2026-01-01T12:00:00+00:00",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "turn",
                    "timestamp": "2026-01-01T12:00:01+00:00",
                    "messages": [{"role": "user", "content": "hi"}],
                    "final_answer": "hello",
                }
            )
            + "\n"
        )

        migrated, skipped, failed = migrate_path(workspaces, backup=False, dry_run=False, recursive=True)
        assert migrated == 1
        # Verify it's now in the new format
        first = json.loads(old_file.read_text().splitlines()[0])
        assert first["type"] == "session_start"

    def test_migrate_daemon_sessions_merges_into_history(self, tmp_path):
        """Old daemon/sessions/{id}.jsonl events get merged (sorted by ts) into
        the matching history/{id}.jsonl file. The daemon source file is then
        removed (or backed up)."""
        from tsugite.cli.history import migrate_daemon_sessions

        history_dir = tmp_path / "history"
        history_dir.mkdir()
        daemon_dir = tmp_path / "daemon" / "sessions"
        daemon_dir.mkdir(parents=True)

        sid = "test-session"
        # Pre-existing history file (already migrated to new format)
        history_file = history_dir / f"{sid}.jsonl"
        history_file.write_text(
            json.dumps({"type": "session_start", "ts": "2026-01-01T12:00:00+00:00", "data": {"agent": "x"}})
            + "\n"
            + json.dumps({"type": "user_input", "ts": "2026-01-01T12:00:01+00:00", "data": {"text": "hi"}})
            + "\n"
        )
        # Daemon-only events from the old separate log
        daemon_file = daemon_dir / f"{sid}.jsonl"
        daemon_file.write_text(
            json.dumps({"type": "reaction", "emoji": "👍", "timestamp": "2026-01-01T12:00:02+00:00"})
            + "\n"
            + json.dumps(
                {"type": "prompt_snapshot", "token_breakdown": {"total": 100}, "timestamp": "2026-01-01T12:00:03+00:00"}
            )
            + "\n"
        )

        merged, skipped = migrate_daemon_sessions(daemon_dir, history_dir, backup=False, dry_run=False)
        assert merged == 1
        # Daemon file is gone
        assert not daemon_file.exists()
        # History file now has both original events + merged daemon events
        events = [json.loads(line) for line in history_file.read_text().splitlines() if line.strip()]
        types = [e["type"] for e in events]
        assert types == ["session_start", "user_input", "reaction", "prompt_snapshot"]
        reaction = next(e for e in events if e["type"] == "reaction")
        assert reaction["data"]["emoji"] == "👍"

    def test_migrate_daemon_sessions_preserves_timestamp_order(self, tmp_path):
        """Daemon events whose ts fall *within* the existing conversation range must
        interleave by timestamp, not append after the last existing event (which
        broke chronological order in `history show` / export)."""
        from tsugite.cli.history import migrate_daemon_sessions

        history_dir = tmp_path / "history"
        history_dir.mkdir()
        daemon_dir = tmp_path / "daemon" / "sessions"
        daemon_dir.mkdir(parents=True)

        sid = "interleave"
        (history_dir / f"{sid}.jsonl").write_text(
            json.dumps({"type": "user_input", "ts": "2026-01-01T12:00:05+00:00", "data": {"text": "a"}})
            + "\n"
            + json.dumps({"type": "model_response", "ts": "2026-01-01T12:00:30+00:00", "data": {"raw_content": "b"}})
            + "\n"
        )
        (daemon_dir / f"{sid}.jsonl").write_text(
            json.dumps({"type": "reaction", "emoji": "👍", "timestamp": "2026-01-01T12:00:08+00:00"})
            + "\n"
            + json.dumps({"type": "prompt_snapshot", "token_breakdown": {}, "timestamp": "2026-01-01T12:00:10+00:00"})
            + "\n"
        )

        migrate_daemon_sessions(daemon_dir, history_dir, backup=False, dry_run=False)

        events = [json.loads(line) for line in (history_dir / f"{sid}.jsonl").read_text().splitlines() if line.strip()]
        tss = [e["ts"] for e in events]
        assert tss == sorted(tss), f"events not in timestamp order: {tss}"
        assert [e["type"] for e in events] == ["user_input", "reaction", "prompt_snapshot", "model_response"]

    def test_migrate_daemon_sessions_dry_run_does_nothing(self, tmp_path):
        from tsugite.cli.history import migrate_daemon_sessions

        history_dir = tmp_path / "history"
        history_dir.mkdir()
        daemon_dir = tmp_path / "daemon" / "sessions"
        daemon_dir.mkdir(parents=True)
        (history_dir / "s.jsonl").write_text(
            json.dumps({"type": "session_start", "ts": "2026-01-01T00:00:00+00:00", "data": {}}) + "\n"
        )
        (daemon_dir / "s.jsonl").write_text(
            json.dumps({"type": "reaction", "emoji": "x", "timestamp": "2026-01-01T00:00:01+00:00"}) + "\n"
        )

        merged, skipped = migrate_daemon_sessions(daemon_dir, history_dir, backup=False, dry_run=True)
        assert merged == 1
        # Source still exists in dry-run
        assert (daemon_dir / "s.jsonl").exists()
        # History wasn't modified
        types = [
            json.loads(line)["type"] for line in (history_dir / "s.jsonl").read_text().splitlines() if line.strip()
        ]
        assert types == ["session_start"]
