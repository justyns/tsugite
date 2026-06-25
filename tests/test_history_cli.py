"""history list/search/show read through the configured backend (sqlite by default)."""

import pytest
from typer.testing import CliRunner

from tsugite.cli.history import history_app
from tsugite.history import get_history_backend
from tsugite.history.sqlite_conn import close_all

runner = CliRunner()


@pytest.fixture
def seeded():
    # get_history_backend() is sqlite by default; XDG is isolated per test by conftest.
    backend = get_history_backend()
    session = backend.create("chat", "openai:gpt-4o-mini")
    session.record("user_input", text="deploy the service")
    session.record("model_response", raw_content="deploying now")
    yield session.session_id
    close_all()


def test_list_shows_session(seeded):
    res = runner.invoke(history_app, ["list"])
    assert res.exit_code == 0, res.output
    assert "chat" in res.stdout


def test_search_finds_text(seeded):
    res = runner.invoke(history_app, ["search", "deploy"])
    assert res.exit_code == 0, res.output
    assert "deploy" in res.stdout


def test_search_reports_no_match(seeded):
    res = runner.invoke(history_app, ["search", "zzz-no-such-text"])
    assert res.exit_code == 0
    assert "No matches" in res.stdout


def test_show_renders_conversation(seeded):
    res = runner.invoke(history_app, ["show", seeded])
    assert res.exit_code == 0, res.output
    assert "deploy the service" in res.stdout


def test_show_missing_exits_nonzero():
    res = runner.invoke(history_app, ["show", "no-such-conversation"])
    assert res.exit_code != 0
