"""Tests for the tsu history migrate command (old Turn → new event log)."""

import json
from pathlib import Path

import pytest

from tsugite.cli.history import migrate_session_file


def _write_old_session(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture
def session_path(tmp_path: Path) -> Path:
    return tmp_path / "20260101_120000_chat_abc123.jsonl"


def test_migrates_basic_session(session_path):
    _write_old_session(
        session_path,
        [
            {"type": "session_meta", "agent": "chat", "model": "openai:gpt-4o", "machine": "mac", "created_at": "2026-01-01T12:00:00+00:00"},
            {"type": "turn", "timestamp": "2026-01-01T12:00:05+00:00", "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ], "final_answer": "hi there", "tokens": 100, "cost": 0.001},
            {"type": "session_status", "status": "success", "timestamp": "2026-01-01T12:00:10+00:00"},
        ],
    )

    result = migrate_session_file(session_path, backup=False, dry_run=False)
    assert result.startswith("migrated")

    events = _read_events(session_path)
    types = [e["type"] for e in events]
    assert types == ["session_start", "user_input", "model_response", "session_end"]
    assert events[0]["data"]["agent"] == "chat"
    assert events[0]["data"]["model"] == "openai:gpt-4o"
    assert events[1]["data"]["text"] == "hello"
    assert events[2]["data"]["raw_content"] == "hi there"
    assert events[2]["data"]["usage"]["total_tokens"] == 100
    assert events[3]["data"]["status"] == "success"


def test_skips_already_new_format(session_path):
    _write_old_session(
        session_path,
        [
            {"type": "session_start", "ts": "2026-01-01T12:00:00+00:00", "data": {"agent": "x", "model": "y"}},
        ],
    )
    result = migrate_session_file(session_path, backup=False, dry_run=False)
    assert result.startswith("skipped:already_new")


def test_skips_empty_file(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    result = migrate_session_file(p, backup=False, dry_run=False)
    assert result.startswith("skipped")


def test_dry_run_does_not_modify(session_path):
    original_content = json.dumps({"type": "session_meta", "agent": "x", "model": "y", "machine": "m", "created_at": "2026-01-01T00:00:00+00:00"})
    session_path.write_text(original_content + "\n")
    migrate_session_file(session_path, backup=False, dry_run=True)
    assert session_path.read_text().strip() == original_content


def test_creates_backup_when_requested(session_path):
    _write_old_session(
        session_path,
        [
            {"type": "session_meta", "agent": "x", "model": "y", "machine": "m", "created_at": "2026-01-01T00:00:00+00:00"},
        ],
    )
    original = session_path.read_text()
    migrate_session_file(session_path, backup=True, dry_run=False)
    bak = session_path.with_suffix(session_path.suffix + ".bak")
    assert bak.exists()
    assert bak.read_text() == original


def test_preserves_compaction_summary(session_path):
    _write_old_session(
        session_path,
        [
            {"type": "session_meta", "agent": "x", "model": "y", "machine": "m", "created_at": "2026-01-01T00:00:00+00:00", "compacted_from": "older-id"},
            {"type": "compaction_summary", "summary": "we discussed cats", "previous_turns": 10, "retained_turns": 2, "compaction_reason": "token_threshold"},
            {"type": "turn", "timestamp": "2026-01-01T00:00:01+00:00", "messages": [
                {"role": "user", "content": "next message"},
                {"role": "assistant", "content": "ok"},
            ], "final_answer": "ok"},
        ],
    )
    migrate_session_file(session_path, backup=False, dry_run=False)
    events = _read_events(session_path)
    types = [e["type"] for e in events]
    assert "compaction" in types
    compaction = next(e for e in events if e["type"] == "compaction")
    assert compaction["data"]["summary"] == "we discussed cats"
    assert compaction["data"]["replaced_count"] == 10
    assert compaction["data"]["retained_count"] == 2
    assert compaction["data"]["reason"] == "token_threshold"
    # parent_session preserved on session_start
    assert events[0]["data"]["parent_session"] == "older-id"


def test_drops_tool_calls_and_context(session_path):
    """User said it's OK to lose tool calls/context as long as the conversation is preserved."""
    _write_old_session(
        session_path,
        [
            {"type": "session_meta", "agent": "x", "model": "y", "machine": "m", "created_at": "2026-01-01T00:00:00+00:00"},
            {"type": "context", "attachments": {"foo": {"hash": "abc"}}, "skills": ["sk1"], "hash": "x"},
            {"type": "context_update", "changed": {}, "removed": [], "added_skills": ["sk2"], "removed_skills": [], "timestamp": "2026-01-01T00:00:01+00:00", "hash": "y"},
            {"type": "hook_execution", "phase": "pre", "command": "echo hi", "exit_code": 0, "timestamp": "2026-01-01T00:00:02+00:00"},
            {"type": "turn", "timestamp": "2026-01-01T00:00:03+00:00", "messages": [
                {"role": "user", "content": "hey"},
                {"role": "assistant", "content": "yo"},
            ], "final_answer": "yo"},
        ],
    )
    migrate_session_file(session_path, backup=False, dry_run=False)
    events = _read_events(session_path)
    types = [e["type"] for e in events]
    assert "context" not in types
    assert "context_update" not in types
    assert "hook_execution" not in types
    # Conversation preserved.
    assert "user_input" in types
    assert "model_response" in types


def test_handles_user_summary_when_messages_missing(session_path):
    """Some old turns only have user_summary, not full messages."""
    _write_old_session(
        session_path,
        [
            {"type": "session_meta", "agent": "x", "model": "y", "machine": "m", "created_at": "2026-01-01T00:00:00+00:00"},
            {"type": "turn", "timestamp": "2026-01-01T00:00:01+00:00", "user_summary": "what time?", "final_answer": "noon", "messages": []},
        ],
    )
    migrate_session_file(session_path, backup=False, dry_run=False)
    events = _read_events(session_path)
    user = next(e for e in events if e["type"] == "user_input")
    assert user["data"]["text"] == "what time?"


def test_atomic_rewrite_on_failure(session_path, monkeypatch):
    """If write fails partway, the original file must remain intact."""
    _write_old_session(
        session_path,
        [
            {"type": "session_meta", "agent": "x", "model": "y", "machine": "m", "created_at": "2026-01-01T00:00:00+00:00"},
            {"type": "turn", "timestamp": "2026-01-01T00:00:01+00:00", "messages": [
                {"role": "user", "content": "hi"},
            ], "final_answer": "ok"},
        ],
    )
    original = session_path.read_text()

    import tsugite.cli.history as hmod

    real_open = open

    def boom_open(path, *a, **kw):
        # Block writes to the .tmp file
        if str(path).endswith(".tmp"):
            raise OSError("disk full (simulated)")
        return real_open(path, *a, **kw)

    monkeypatch.setattr(hmod, "open", boom_open, raising=False)
    with pytest.raises(OSError):
        migrate_session_file(session_path, backup=False, dry_run=False)
    monkeypatch.undo()

    # Original is unchanged
    assert session_path.read_text() == original
