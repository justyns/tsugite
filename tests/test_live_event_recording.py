"""Verify that running an agent through `run_agent` records events live to storage,
and that `save_run_to_history` doesn't double-write when the agent already did."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tsugite.agent_runner.history_integration import save_run_to_history
from tsugite.history import SessionStorage


@pytest.fixture
def history_dir(tmp_path: Path):
    history = tmp_path / "history"
    history.mkdir()
    with (
        patch("tsugite.history.storage.get_history_dir", return_value=history),
        patch("tsugite.agent_runner.history_integration.get_history_dir", return_value=history),
    ):
        yield history


@pytest.fixture
def fake_agent_path(tmp_path: Path) -> Path:
    p = tmp_path / "fake.md"
    p.write_text("---\nname: fake\nmodel: openai:gpt-4o-mini\n---\nbe brief")
    return p


def test_save_run_to_history_idempotent_when_agent_already_recorded(history_dir, fake_agent_path):
    """If the agent recorded events live, save_run_to_history must not duplicate."""
    storage = SessionStorage.create(
        agent_name="fake",
        model="openai:gpt-4o-mini",
        session_path=history_dir / "live.jsonl",
    )
    storage.record("user_input", text="hello")
    storage.record("code_execution", code="print('x')", output="x\n", duration_ms=5)
    storage.record("model_response", raw_content="Done.", usage={"total_tokens": 50})
    storage.record("session_end", status="success")

    with patch("tsugite.config.load_config") as cfg:
        cfg.return_value = MagicMock(history_enabled=True)
        save_run_to_history(
            agent_path=fake_agent_path,
            agent_name="fake",
            prompt="hello",
            result="Done.",
            model="openai:gpt-4o-mini",
            continue_conversation_id="live",
        )

    events = list(storage.iter_events())
    types = [e.type for e in events]
    assert types == ["session_start", "user_input", "code_execution", "model_response", "session_end"]


def test_save_run_to_history_finalizes_when_agent_recorded_but_no_session_end(history_dir, fake_agent_path):
    """If the agent crashed mid-run after recording events, save adds session_end."""
    storage = SessionStorage.create(
        agent_name="fake",
        model="openai:gpt-4o-mini",
        session_path=history_dir / "crash.jsonl",
    )
    storage.record("user_input", text="hello")
    storage.record("model_response", raw_content="partial")

    with patch("tsugite.config.load_config") as cfg:
        cfg.return_value = MagicMock(history_enabled=True)
        save_run_to_history(
            agent_path=fake_agent_path,
            agent_name="fake",
            prompt="hello",
            result="partial",
            model="openai:gpt-4o-mini",
            continue_conversation_id="crash",
            status="error",
            error_message="boom",
        )

    events = list(storage.iter_events())
    types = [e.type for e in events]
    assert types[-1] == "session_end"
    end_event = events[-1]
    assert end_event.data["status"] == "error"
    assert end_event.data["error_message"] == "boom"
    # And we didn't double-record user_input or model_response
    assert types.count("user_input") == 1
    assert types.count("model_response") == 1


def test_save_run_to_history_legacy_path_writes_full_events(history_dir, fake_agent_path):
    """Call sites that don't use storage threading still get a complete event log."""
    with patch("tsugite.config.load_config") as cfg:
        cfg.return_value = MagicMock(history_enabled=True)
        session_id = save_run_to_history(
            agent_path=fake_agent_path,
            agent_name="fake",
            prompt="hello",
            result="hi",
            model="openai:gpt-4o-mini",
        )

    storage = SessionStorage.load(history_dir / f"{session_id}.jsonl")
    types = [e.type for e in storage.iter_events()]
    assert types == ["session_start", "user_input", "model_response", "session_end"]


def test_run_agent_creates_storage_at_runtime(history_dir, fake_agent_path):
    """The TsugiteAgent constructor accepts a `storage` keyword and stores it
    on the instance — confirms the plumbing endpoint exists. (Integration is
    covered by the live CLI smoke run.)"""
    from tsugite.core.agent import TsugiteAgent
    from tsugite.history import SessionStorage

    storage = SessionStorage.create(
        agent_name="t",
        model="openai:gpt-4o-mini",
        session_path=history_dir / "t.jsonl",
    )
    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="",
        max_turns=1,
        storage=storage,
    )
    assert agent.storage is storage
    assert agent._user_input_recorded is False
