"""One-time JSONL -> SQLite import and on-demand JSONL export."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tsugite.history.models import Event
from tsugite.history.sqlite_backend import SqliteHistoryBackend
from tsugite.history.sqlite_conn import close_all
from tsugite.history.storage import SessionStorage


@pytest.fixture
def backend(tmp_path):
    b = SqliteHistoryBackend(db_path=tmp_path / "history.db")
    yield b
    close_all()


def _write_legacy(dirpath: Path, name: str) -> Path:
    path = dirpath / f"{name}.jsonl"
    st = SessionStorage.create(agent_name="legacy", model="m", session_path=path)
    st.record("user_input", text="old question")
    st.record("model_response", raw_content="old answer")
    return path


def test_import_strips_model_request_messages(tmp_path, backend):
    """Legacy model_request events carry the full messages array; import drops it for a count."""
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    p = legacy / "20260101_000003_legacy_ddd.jsonl"
    st = SessionStorage.create(agent_name="legacy", model="m", session_path=p)
    st.record(
        "model_request",
        turn=0,
        provider="claude_code",
        model="opus",
        messages=[{"role": "system", "content": "x" * 5000}, {"role": "user", "content": "hi"}],
        tool_names=["read_file"],
    )

    backend.import_jsonl([p])

    mr = next(e for e in backend.load(p.stem).load_events() if e.type == "model_request")
    assert "messages" not in mr.data  # the bloat is gone
    assert "messages_sha256" not in mr.data
    assert mr.data["message_count"] == 2
    assert mr.data["tool_names"] == ["read_file"]


def test_export_round_trips_and_excludes_id(backend):
    s = backend.create("chat", "m")
    s.record("user_input", text="hello")
    s.record("model_response", raw_content="hi")

    parsed = [json.loads(line) for line in backend.export_jsonl(s.session_id)]
    assert all("id" not in p for p in parsed)
    assert [Event.model_validate(p).type for p in parsed] == ["session_start", "user_input", "model_response"]


def test_import_then_idempotent(tmp_path, backend):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    p = _write_legacy(legacy, "20260101_000000_legacy_aaa")

    rep = backend.import_jsonl([p])
    assert rep["imported"] == 1
    assert [e.type for e in backend.load(p.stem).load_events()] == ["session_start", "user_input", "model_response"]

    rep2 = backend.import_jsonl([p])
    assert rep2["skipped"] == 1 and rep2["imported"] == 0


def test_import_tolerates_malformed_line(tmp_path, backend):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    p = _write_legacy(legacy, "20260101_000001_legacy_bbb")
    with open(p, "a") as f:
        f.write("{ not valid json\n")

    rep = backend.import_jsonl([p])
    assert rep["imported"] == 1
    assert backend.count_events(p.stem) == 3  # the valid events still imported


def test_import_no_session_start_synthesizes_row(tmp_path, backend):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    p = legacy / "20260101_000002_legacy_ccc.jsonl"
    orphan = Event(type="user_input", ts=datetime(2026, 1, 1, tzinfo=timezone.utc), data={"text": "orphan"})
    p.write_text(orphan.model_dump_json(exclude={"id"}, exclude_none=True) + "\n")

    rep = backend.import_jsonl([p])
    assert rep["imported"] == 1 and rep["no_session_start"] == 1
    assert backend.exists(p.stem)
    assert backend.count_events(p.stem) == 1


def test_import_recency_reflects_event_time_not_import_order(tmp_path, backend):
    """Imported sessions sort by their real last activity, not by when they were imported."""
    legacy = tmp_path / "legacy"
    legacy.mkdir()

    def _write(name: str, ts: datetime) -> Path:
        p = legacy / f"{name}.jsonl"
        lines = [
            Event(type="session_start", ts=ts, data={"agent": "a", "model": "m"}),
            Event(type="user_input", ts=ts, data={"text": "hi"}),
        ]
        p.write_text("".join(e.model_dump_json(exclude={"id"}, exclude_none=True) + "\n" for e in lines))
        return p

    older = _write("20200101_000000_a_old", datetime(2020, 1, 1, tzinfo=timezone.utc))
    newer = _write("20240101_000000_a_new", datetime(2024, 1, 1, tzinfo=timezone.utc))

    # Import the newer conversation first, then the older one: import order != chronology.
    backend.import_jsonl([newer])
    backend.import_jsonl([older])

    # list_sessions is recency-ordered; the chronologically newer conversation must lead.
    assert backend.list_sessions()[0] == newer.stem


def test_cli_import_and_export_round_trip(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import tsugite.history as history_pkg
    from tsugite.cli.history import history_app

    hist_dir = tmp_path / "history"
    hist_dir.mkdir()
    monkeypatch.setattr(history_pkg, "get_history_dir", lambda: hist_dir)
    monkeypatch.setenv("TSUGITE_HISTORY_DB", str(hist_dir / "history.db"))
    st = SessionStorage.create(
        agent_name="legacy", model="m", session_path=hist_dir / "20260101_000000_legacy_aaa.jsonl"
    )
    st.record("user_input", text="cli question")

    runner = CliRunner()
    try:
        res = runner.invoke(history_app, ["import"])
        assert res.exit_code == 0, res.output
        assert "Imported:" in res.stdout

        res2 = runner.invoke(history_app, ["export", "20260101_000000_legacy_aaa"])
        assert res2.exit_code == 0, res2.output
        assert "user_input" in res2.stdout and "cli question" in res2.stdout
    finally:
        close_all()
