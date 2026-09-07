"""A turn killed by a provider error must record session_end status=error.

Both `_recorded_run_outcome` (scheduler) and the Activity view read that status,
so a turn that dies has to say so in history rather than only on the live SSE
frame, which is gone after a reload.
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from tsugite_daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite_daemon.config import RuntimeDefaults
from tsugite_daemon.session_runner import SessionRunner
from tsugite_daemon.session_store import Session, SessionSource, SessionStore

from tsugite.agent_runner.history_integration import record_session_end
from tsugite.agent_runner.models import AgentSkippedError
from tsugite.cancellation import is_cancelled
from tsugite.exceptions import AgentExecutionError
from tsugite.history import get_history_backend

from .conftest import _wait_until

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


@pytest.mark.asyncio
async def test_a_skip_on_the_first_turn_leaves_no_history_session(adapter, monkeypatch):
    """handle_message opens the session to hold the prompt before the guard runs,
    so a first turn that never happens must take that row with it."""
    monkeypatch.setattr(
        "tsugite_daemon.adapters.base.run_agent",
        MagicMock(side_effect=AgentSkippedError("run_if guard")),
    )
    session = adapter.session_store.get_or_create_interactive("alice")

    with pytest.raises(AgentSkippedError):
        await _run_turn(adapter)

    assert not get_history_backend().exists(session.id), "the skipped run left a history session behind"


@pytest.mark.asyncio
async def test_a_skip_keeps_a_conversation_that_already_has_turns(adapter, monkeypatch):
    """Only the turn that opened the session may discard it, or a guard declining
    on turn five would delete the first four."""
    monkeypatch.setattr(
        "tsugite_daemon.adapters.base.run_agent",
        lambda *a, **kw: SimpleNamespace(token_count=10, cost=0.0, provider_state={}),
    )
    session_id = await _run_turn(adapter)

    monkeypatch.setattr(
        "tsugite_daemon.adapters.base.run_agent",
        MagicMock(side_effect=AgentSkippedError("run_if guard")),
    )
    with pytest.raises(AgentSkippedError):
        await _run_turn(adapter)

    assert get_history_backend().exists(session_id), "a skip deleted a conversation that had already run"


@pytest.mark.asyncio
async def test_a_post_compaction_retry_runs_as_the_successor_session(adapter, monkeypatch):
    """Compaction hands the retry a new session; tools that stamp the current
    session (open_artifact, spawn_job, session_reply) must see that one."""
    from tsugite_daemon.session_runner import get_current_session_id

    first = adapter.session_store.get_or_create_interactive("alice")
    successor = adapter.session_store.get_or_create_interactive("alice-successor")
    seen = []

    async def compact(*a, **kw):
        return successor.id

    def run(*a, **kw):
        seen.append(get_current_session_id())
        raise AgentExecutionError(TOO_LONG)

    monkeypatch.setattr("tsugite_daemon.adapters.base.run_agent", MagicMock(side_effect=run))
    monkeypatch.setattr(adapter, "_run_compaction", compact)

    with pytest.raises(AgentExecutionError):
        await _run_turn(adapter)

    assert seen == [first.id, successor.id]


@pytest.mark.asyncio
async def test_cancelling_a_background_session_records_it_as_cancelled(adapter, monkeypatch):
    """Without a bound cancel Event the turn runs to completion and records a successful run."""
    entered = threading.Event()

    def run_agent(*a, **kw):
        entered.set()
        deadline = time.monotonic() + 3.0
        status = "success"
        while time.monotonic() < deadline:
            if is_cancelled():
                status = "cancelled"
                break
            time.sleep(0.01)
        storage = get_history_backend().load("bg-1")
        storage.record("model_response", raw_content="partial answer")
        record_session_end(storage, status=status, error_message=None)
        return SimpleNamespace(token_count=10, cost=0.0, provider_state={})

    monkeypatch.setattr("tsugite_daemon.adapters.base.run_agent", run_agent)

    runner = SessionRunner(adapter.session_store, adapter)
    runner.start_session(Session(id="bg-1", source=SessionSource.BACKGROUND.value, prompt="do the thing"))

    assert await _wait_until(entered.is_set), "the worker never started"
    runner.cancel_session("bg-1")

    assert await _wait_until(lambda: bool(_session_ends("bg-1")), timeout=3.0), "the run recorded no session_end"
    ends = _session_ends("bg-1")

    assert [e.data["status"] for e in ends] == ["cancelled"]
