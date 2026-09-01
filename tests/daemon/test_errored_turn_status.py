"""A turn killed by a provider error must record session_end status=error.

Both `_recorded_run_outcome` (scheduler) and the Activity view read that status,
so a turn that dies has to say so in history rather than only on the live SSE
frame, which is gone after a reload.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from tsugite_daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite_daemon.config import RuntimeDefaults
from tsugite_daemon.session_store import SessionStore

from tsugite.exceptions import AgentExecutionError
from tsugite.history import get_history_backend

POISON = "API Error: 400 messages: text content blocks must be non-empty (subtype=success)"
TOO_LONG = "prompt is too long: 300000 tokens > 200000 maximum"


class _StubAdapter(BaseAdapter):
    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent_file = workspace / "agent.md"
    agent_file.write_text("---\nname: test-agent\n---\n\nHi.\n")
    runtime = RuntimeDefaults(workspace_dir=workspace, agent_file=str(agent_file), model="anthropic:claude-opus-4")
    adapter = _StubAdapter(runtime, SessionStore(tmp_path / "store.json"))

    monkeypatch.setattr(adapter, "_resolve_agent_path", lambda: agent_file)
    monkeypatch.setattr(adapter, "_build_message_context", lambda message, *a, **kw: message)
    monkeypatch.setattr(adapter, "_build_agent_context", lambda *a, **kw: {})
    monkeypatch.setattr(adapter, "_update_skill_ttl", lambda *a, **kw: None)
    return adapter


async def _run_turn(adapter) -> str:
    """Drive one full turn, returning the session it ran in."""
    await adapter.handle_message(
        user_id="alice",
        message="do the thing",
        channel_context=ChannelContext(
            source="http", channel_id=None, user_id="alice", reply_to="http:alice", metadata={}
        ),
    )
    return adapter.session_store.get_or_create_interactive("alice").id


def _session_ends(session_id):
    return [e for e in get_history_backend().load(session_id).iter_events() if e.type == "session_end"]


@pytest.mark.asyncio
async def test_a_turn_killed_by_a_provider_error_records_the_error(adapter, monkeypatch):
    monkeypatch.setattr(
        "tsugite_daemon.adapters.base.run_agent",
        MagicMock(side_effect=AgentExecutionError(POISON)),
    )
    session = adapter.session_store.get_or_create_interactive("alice")

    with pytest.raises(AgentExecutionError):
        await _run_turn(adapter)

    ends = _session_ends(session.id)

    assert ends, "the failed turn recorded no session_end at all"
    assert ends[-1].data["status"] == "error"
    assert POISON in (ends[-1].data.get("error_message") or "")


@pytest.mark.asyncio
async def test_a_clean_turn_still_records_success(adapter, monkeypatch):
    monkeypatch.setattr(
        "tsugite_daemon.adapters.base.run_agent",
        lambda *a, **kw: SimpleNamespace(token_count=10, cost=0.0, provider_state={}),
    )

    ends = _session_ends(await _run_turn(adapter))

    assert ends[-1].data["status"] == "success"


@pytest.mark.asyncio
async def test_giving_up_before_compacting_records_the_error(adapter, monkeypatch):
    """The guard against re-issuing side effects re-raises instead of retrying,
    so that route has to record the failure too."""
    session = adapter.session_store.get_or_create_interactive("alice")

    def run_agent(*a, **kw):
        adapter.session_store.append_event(session.id, {"type": "code_execution", "code": "x=1", "output": "ok"})
        raise AgentExecutionError(TOO_LONG)

    monkeypatch.setattr("tsugite_daemon.adapters.base.run_agent", run_agent)

    with pytest.raises(AgentExecutionError):
        await _run_turn(adapter)

    ends = _session_ends(session.id)

    assert ends, "the abandoned turn recorded no session_end at all"
    assert ends[-1].data["status"] == "error"


@pytest.mark.asyncio
async def test_a_failed_post_compaction_retry_records_the_error(adapter, monkeypatch):
    """Compaction rewrites conv_id, so the retry's failure has to be recorded
    against the session it actually ran in."""
    session = adapter.session_store.get_or_create_interactive("alice")

    async def compact(*a, **kw):
        return session.id

    monkeypatch.setattr(
        "tsugite_daemon.adapters.base.run_agent",
        MagicMock(side_effect=AgentExecutionError(TOO_LONG)),
    )
    monkeypatch.setattr(adapter, "_run_compaction", compact)

    with pytest.raises(AgentExecutionError):
        await _run_turn(adapter)

    ends = _session_ends(session.id)

    assert ends, "the failed retry recorded no session_end at all"
    assert ends[-1].data["status"] == "error"


@pytest.mark.asyncio
async def test_a_bare_runtime_error_records_the_error(adapter, monkeypatch):
    """The runner flattens anything without execution_steps into a plain
    RuntimeError, and a turn with no session_end reads as a successful run."""
    monkeypatch.setattr(
        "tsugite_daemon.adapters.base.run_agent",
        MagicMock(side_effect=RuntimeError("Agent execution failed: connection reset")),
    )
    session = adapter.session_store.get_or_create_interactive("alice")

    with pytest.raises(RuntimeError):
        await _run_turn(adapter)

    ends = _session_ends(session.id)

    assert ends, "the failed turn recorded no session_end at all"
    assert ends[-1].data["status"] == "error"


@pytest.mark.asyncio
async def test_a_turn_that_already_answered_once_still_ends_as_error(adapter, monkeypatch):
    """A turn dying after the agent loop recorded a model_response takes
    save_run_to_history's other branch, which writes only the session_end."""
    session = adapter.session_store.get_or_create_interactive("alice")

    def run_agent(*a, **kw):
        get_history_backend().load(session.id).record("model_response", raw_content="partial answer")
        raise AgentExecutionError(POISON)

    monkeypatch.setattr("tsugite_daemon.adapters.base.run_agent", run_agent)

    with pytest.raises(AgentExecutionError):
        await _run_turn(adapter)

    ends = _session_ends(session.id)

    assert ends, "the failed turn recorded no session_end at all"
    assert ends[-1].data["status"] == "error"
