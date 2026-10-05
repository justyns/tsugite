"""Every daemon turn records one turn id on its user_input and on its outcome events."""

import asyncio
import threading
from contextlib import suppress
from types import SimpleNamespace

import pytest
from tsugite_daemon.adapters.base import BaseAdapter, ChannelContext, build_session_event_persister
from tsugite_daemon.adapters.http.sse import SSEProgressHandler
from tsugite_daemon.config import RuntimeDefaults
from tsugite_daemon.session_store import SessionStore

from tsugite.agent_runner.history_integration import record_final_result, record_session_end, record_user_input
from tsugite.history import get_history_backend
from tsugite.providers.base import CompletionResponse, Usage

TURN_TYPES = ("user_input", "final_result", "session_end")


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
    agent_file.write_text("---\nname: test-agent\nextends: none\ntools: []\n---\n\nHi.\n")
    runtime = RuntimeDefaults(workspace_dir=workspace, agent_file=str(agent_file), model="openai:gpt-4o-mini")
    adapter = _StubAdapter(runtime, SessionStore(tmp_path / "store.json"))

    monkeypatch.setattr(adapter, "_resolve_agent_path", lambda *a: agent_file)
    monkeypatch.setattr(adapter, "_build_message_context", lambda message, *a, **kw: message)
    monkeypatch.setattr(adapter, "_build_agent_context", lambda *a, **kw: {})
    monkeypatch.setattr(adapter, "_update_skill_ttl", lambda *a, **kw: None)
    return adapter


def _send(adapter, message, custom_logger=None):
    return adapter.handle_message(
        user_id="alice",
        message=message,
        channel_context=ChannelContext(
            source="http", channel_id=None, user_id="alice", reply_to="http:alice", metadata={}
        ),
        custom_logger=custom_logger,
    )


def _turn_events(session_id):
    return [e for e in get_history_backend().load(session_id).iter_events() if e.type in TURN_TYPES]


@pytest.mark.asyncio
async def test_each_turn_ties_its_user_input_to_its_outcome(adapter, mock_provider):
    mock_provider.acompletion.return_value = CompletionResponse(
        content='```python-exec\nreturn_value("done")\n```', usage=Usage(total_tokens=10), cost=0.0
    )
    session_id = adapter.session_store.get_or_create_interactive("alice").id

    for message in ("first", "second"):
        progress = SSEProgressHandler()
        progress.set_loop(asyncio.get_running_loop())
        progress.set_event_persister(build_session_event_persister(adapter.session_store, session_id))
        await _send(adapter, message, SimpleNamespace(ui_handler=progress))

    events = _turn_events(session_id)
    assert [e.type for e in events] == list(TURN_TYPES) * 2
    first, second = {e.data.get("turn_id") for e in events[:3]}, {e.data.get("turn_id") for e in events[3:]}
    assert len(first) == len(second) == 1
    assert None not in first | second
    assert first != second


@pytest.mark.asyncio
async def test_message_after_a_cancelled_turn_with_a_running_worker(adapter, monkeypatch):
    """task.cancel() leaves the first turn's worker running; it records its outcome
    after the second turn has started."""
    started, release, first_done = threading.Event(), threading.Event(), threading.Event()

    def run_agent(*, prompt, continue_conversation_id, **kw):
        storage = get_history_backend().load(continue_conversation_id)
        record_user_input(storage, prompt)
        if prompt == "first":
            started.set()
            release.wait(5)
        else:
            release.set()
            first_done.wait(5)
        record_final_result(storage, result=f"answer to {prompt}")
        record_session_end(storage)
        if prompt == "first":
            first_done.set()
        return SimpleNamespace(token_count=1, cost=0.0, provider_state={})

    monkeypatch.setattr("tsugite_daemon.adapters.base.run_agent", run_agent)
    monkeypatch.setattr(adapter, "_save_history", lambda **kw: None)
    session_id = adapter.session_store.get_or_create_interactive("alice").id

    first = asyncio.create_task(_send(adapter, "first"))
    await asyncio.to_thread(started.wait, 5)
    first.cancel()
    with suppress(asyncio.CancelledError):
        await first
    await _send(adapter, "second")

    events = _turn_events(session_id)
    assert [e.data["text"] for e in events if e.type == "user_input"] == ["first", "second"]
    assert [e.data["result"] for e in events if e.type == "final_result"] == ["answer to first", "answer to second"]
