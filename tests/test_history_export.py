"""On-demand JSONL export of a stored session.

Export is a portability feature, not a storage format: nothing reads these files
back. The one-time JSONL import that used to live alongside it is gone.
"""

import json

import pytest

from tsugite.history.models import Event
from tsugite.history.sqlite_backend import SqliteHistoryBackend
from tsugite.history.sqlite_conn import close_all


@pytest.fixture
def backend(tmp_path):
    b = SqliteHistoryBackend(db_path=tmp_path / "history.db")
    yield b
    close_all()


def test_export_round_trips_and_excludes_id(backend):
    s = backend.create("chat", "m")
    s.record("user_input", text="hello")
    s.record("model_response", raw_content="hi")

    parsed = [json.loads(line) for line in backend.export_jsonl(s.session_id)]

    assert all("id" not in p for p in parsed), "the db row id is an implementation detail"
    assert [Event.model_validate(p).type for p in parsed] == ["session_start", "user_input", "model_response"]


def test_cli_export_emits_the_event_stream(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from tsugite.cli.history import history_app

    hist_dir = tmp_path / "history"
    hist_dir.mkdir()
    monkeypatch.setenv("TSUGITE_HISTORY_DB", str(hist_dir / "history.db"))

    backend = SqliteHistoryBackend(db_path=hist_dir / "history.db")
    session = backend.create("legacy", "m")
    session.record("user_input", text="cli question")

    try:
        result = CliRunner().invoke(history_app, ["export", session.session_id])
        assert result.exit_code == 0, result.output
        assert "user_input" in result.stdout
        assert "cli question" in result.stdout
    finally:
        close_all()
