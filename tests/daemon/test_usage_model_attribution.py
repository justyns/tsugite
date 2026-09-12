"""Usage rows must record the model the turn actually ran on.

A session running under `/model` bills against the override. A session or
schedule pinned to an agent file bills against that file's model. Otherwise the
Usage tab attributes one provider's spend to another.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from tsugite_daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite_daemon.config import RuntimeDefaults
from tsugite_daemon.session_store import SessionStore

from tsugite.exceptions import AgentExecutionError

AGENT_MODEL = "codex_cli:gpt-5.5"
SESSION_MODEL = "anthropic:claude-opus-4"
TURN_MODEL = "openai:gpt-4o-mini"


class _StubAdapter(BaseAdapter):
    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


@pytest.fixture
def history_calls(monkeypatch):
    """Captures what the real _save_history hands to save_run_to_history."""
    calls = {}
    monkeypatch.setattr(
        "tsugite.agent_runner.history_integration.save_run_to_history",
        lambda **kw: calls.update(kw),
    )
    return calls


@pytest.fixture
def adapter(tmp_path, monkeypatch, history_calls):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent_file = workspace / "agent.md"
    agent_file.write_text("---\nname: test-agent\n---\n\nHi.\n")
    runtime = RuntimeDefaults(workspace_dir=workspace, agent_file=str(agent_file), model=AGENT_MODEL)
    adapter = _StubAdapter(runtime, SessionStore(tmp_path / "store.json"))

    monkeypatch.setattr(adapter, "_resolve_agent_path", lambda: agent_file)
    monkeypatch.setattr(adapter, "_build_message_context", lambda message, *a, **kw: message)
    monkeypatch.setattr(adapter, "_build_agent_context", lambda *a, **kw: {})
    monkeypatch.setattr(adapter, "_update_skill_ttl", lambda *a, **kw: None)
    monkeypatch.setattr(
        "tsugite_daemon.adapters.base.run_agent",
        lambda *a, **kw: SimpleNamespace(token_count=1234, cost=4.25, provider_state={}),
    )
    return adapter


async def _run_turn(
    adapter, monkeypatch, *, session_override=None, turn_override=None, source="http", metadata=None
) -> dict:
    """Drive one full turn, returning the kwargs it recorded to the usage store."""
    captured = {}
    store = MagicMock()
    store.record.side_effect = lambda **kw: captured.update(kw)
    monkeypatch.setattr("tsugite.usage.get_usage_store", lambda: store)

    session = adapter.session_store.get_or_create_interactive("alice")
    if session_override:
        adapter.session_store.set_model_override(session.id, session_override)

    meta = dict(metadata or {})
    if turn_override:
        meta["model_override"] = turn_override

    await adapter.handle_message(
        user_id="alice",
        message="hi",
        channel_context=ChannelContext(
            source=source,
            channel_id=None,
            user_id="alice",
            reply_to="http:alice",
            metadata=meta,
        ),
    )
    return captured


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "session_override,turn_override,expected",
    [
        pytest.param(SESSION_MODEL, None, SESSION_MODEL, id="session-override"),
        pytest.param(SESSION_MODEL, TURN_MODEL, TURN_MODEL, id="per-turn-override-wins"),
        pytest.param(None, None, AGENT_MODEL, id="no-override-bills-agent-default"),
    ],
)
async def test_usage_row_records_the_model_the_turn_ran_on(
    adapter, monkeypatch, session_override, turn_override, expected
):
    row = await _run_turn(adapter, monkeypatch, session_override=session_override, turn_override=turn_override)

    assert row["model"] == expected


@pytest.mark.asyncio
async def test_failed_turn_records_history_against_the_override(adapter, monkeypatch, history_calls):
    """A turn that dies before the agent records a model_response gets its
    model_response written post-hoc by save_run_to_history, off this model."""
    monkeypatch.setattr(
        "tsugite_daemon.adapters.base.run_agent",
        MagicMock(side_effect=AgentExecutionError("boom")),
    )

    with pytest.raises(AgentExecutionError):
        await _run_turn(adapter, monkeypatch, session_override=SESSION_MODEL)

    assert history_calls["model"] == SESSION_MODEL


@pytest.mark.asyncio
async def test_history_session_opens_on_the_override_too(adapter, monkeypatch):
    """The history session and the usage row must agree on the turn's model."""
    opened = {}
    monkeypatch.setattr(
        "tsugite.agent_runner.history_integration.open_or_create_session",
        lambda **kw: opened.update(kw) or None,
    )

    await _run_turn(adapter, monkeypatch, session_override=SESSION_MODEL)

    assert opened["model"] == SESSION_MODEL


PINNED_AGENT_MODEL = "ollama:pinned-agent-model"


@pytest.fixture
def pinned_adapter(tmp_path, monkeypatch, history_calls):
    """Adapter with no daemon-wide model and two agent files pinning different models."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "default.md").write_text(f"---\nname: default-agent\nmodel: {AGENT_MODEL}\n---\n\nHi.\n")
    (workspace / "pinned.md").write_text(f"---\nname: pinned-agent\nmodel: {PINNED_AGENT_MODEL}\n---\n\nHi.\n")
    runtime = RuntimeDefaults(workspace_dir=workspace, agent_file=str(workspace / "default.md"), model=None)
    adapter = _StubAdapter(runtime, SessionStore(tmp_path / "store.json"))

    monkeypatch.setattr(adapter, "_build_message_context", lambda message, *a, **kw: message)
    monkeypatch.setattr(adapter, "_build_agent_context", lambda *a, **kw: {})
    monkeypatch.setattr(adapter, "_update_skill_ttl", lambda *a, **kw: None)
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source,session_override,pin_agent,expected",
    [
        pytest.param("http", None, True, PINNED_AGENT_MODEL, id="session-pinned-to-agent-file"),
        pytest.param("http", SESSION_MODEL, True, SESSION_MODEL, id="model-command-beats-agent-file"),
        pytest.param("http", None, False, AGENT_MODEL, id="no-pin-bills-default-agent"),
        pytest.param("scheduler", None, True, PINNED_AGENT_MODEL, id="schedule-pinned-to-agent-file"),
    ],
)
async def test_turn_pinned_to_an_agent_file_runs_on_that_file_model(
    pinned_adapter, monkeypatch, source, session_override, pin_agent, expected
):
    runs = []
    monkeypatch.setattr(
        "tsugite_daemon.adapters.base.run_agent",
        lambda *a, **kw: runs.append(kw) or SimpleNamespace(token_count=1234, cost=4.25, provider_state={}),
    )

    opened = {}
    monkeypatch.setattr(
        "tsugite.agent_runner.history_integration.open_or_create_session",
        lambda **kw: opened.update(kw) or None,
    )

    metadata = {}
    if pin_agent:
        metadata["agent_file_override"] = str(pinned_adapter.runtime.workspace_dir / "pinned.md")
    if source == "scheduler":
        metadata["schedule_id"] = "nightly"

    row = await _run_turn(
        pinned_adapter, monkeypatch, session_override=session_override, source=source, metadata=metadata
    )

    assert row["model"] == expected
    assert row["source"] == source
    assert opened["model"] == expected
    assert runs[0]["exec_options"].model_override == expected


@pytest.mark.asyncio
async def test_turn_with_no_model_configured_reports_no_model_specified(pinned_adapter, monkeypatch):
    """The runner raises its own error when no model is configured anywhere.

    "unknown" is only resolve_model's display sentinel. The agent opts out of inheritance so
    it picks up no model from the workspace default.
    """
    bare_agent = pinned_adapter.runtime.workspace_dir / "bare.md"
    bare_agent.write_text("---\nname: bare-agent\nextends: none\n---\n\nHi.\n")
    pinned_adapter.runtime.agent_file = str(bare_agent)

    with pytest.raises(RuntimeError, match="No model specified"):
        await _run_turn(pinned_adapter, monkeypatch)
