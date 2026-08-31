"""Usage rows must record the model the turn actually ran on.

A session running under `/model` bills against the override; otherwise the Usage
tab attributes one provider's spend to another.
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


async def _run_turn(adapter, monkeypatch, *, session_override=None, turn_override=None) -> dict:
    """Drive one full turn, returning the kwargs it recorded to the usage store."""
    captured = {}
    store = MagicMock()
    store.record.side_effect = lambda **kw: captured.update(kw)
    monkeypatch.setattr("tsugite.usage.get_usage_store", lambda: store)

    session = adapter.session_store.get_or_create_interactive("alice")
    if session_override:
        adapter.session_store.set_model_override(session.id, session_override)

    await adapter.handle_message(
        user_id="alice",
        message="hi",
        channel_context=ChannelContext(
            source="http",
            channel_id=None,
            user_id="alice",
            reply_to="http:alice",
            metadata={"model_override": turn_override} if turn_override else {},
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
