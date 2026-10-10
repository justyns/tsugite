"""Daemon-spawned sessions get their model; failed runs record no model_response."""

from types import SimpleNamespace

import pytest

from tsugite.agent_runner import history_integration
from tsugite.history.sqlite_backend import SqliteHistoryBackend
from tsugite.history.sqlite_conn import close_all


@pytest.fixture
def backend(tmp_path, monkeypatch):
    b = SqliteHistoryBackend(db_path=tmp_path / "history.db")
    monkeypatch.delenv("TSUGITE_SUBAGENT_MODE", raising=False)
    monkeypatch.setattr(history_integration, "get_history_backend", lambda: b)
    monkeypatch.setattr(history_integration, "load_config", lambda: SimpleNamespace(history_enabled=True))
    yield b
    close_all()


def _open(tmp_path, session_id):
    return history_integration.open_or_create_session(
        agent_path=tmp_path / "agent.md",
        agent_name="worker",
        model="codex_cli:gpt-5.6-sol",
        continue_conversation_id=session_id,
        workspace="ws",
    )


def test_continuing_a_bare_daemon_session_records_agent_and_model(backend, tmp_path):
    backend.ensure_session("session-1").record("session_start", prompt="do the thing")

    _open(tmp_path, "session-1")

    summary = backend.load("session-1").summary()
    assert (summary.agent, summary.model, summary.workspace) == ("worker", "codex_cli:gpt-5.6-sol", "ws")


def test_continuing_a_session_with_a_model_leaves_it_alone(backend, tmp_path):
    backend.create("first", "anthropic:claude-opus-5-5", session_id="session-2")

    _open(tmp_path, "session-2")

    summary = backend.load("session-2").summary()
    assert (summary.agent, summary.model) == ("first", "anthropic:claude-opus-5-5")
    assert backend.count_events("session-2", type="session_start") == 1


def test_session_start_without_metadata_keeps_existing_values(backend):
    s = backend.create("first", "m", session_id="session-3")
    created_at = backend.load("session-3").summary().created_at
    s.record("session_start", prompt="later")

    summary = backend.load("session-3").summary()
    assert (summary.agent, summary.model) == ("first", "m")
    assert summary.created_at == created_at


@pytest.mark.parametrize("status", ["error", "interrupted"])
def test_failed_run_records_no_model_response(backend, tmp_path, status):
    sid = history_integration.save_run_to_history(
        agent_path=tmp_path / "agent.md",
        agent_name="worker",
        prompt="hi",
        result="",
        model="codex_cli/gpt-5.6-sol",
        status=status,
        error_message="Invalid model string format",
    )

    session = backend.load(sid)
    assert backend.count_events(sid, type="model_response") == 0
    assert session.summary().status == status


def test_successful_run_records_split_provider_and_model(backend, tmp_path):
    sid = history_integration.save_run_to_history(
        agent_path=tmp_path / "agent.md",
        agent_name="worker",
        prompt="hi",
        result="done",
        model="codex_cli:gpt-5.6-sol",
    )

    (event,) = backend.load(sid).iter_events(types=["model_response"])
    assert (event.data["provider"], event.data["model"]) == ("codex_cli", "gpt-5.6-sol")
