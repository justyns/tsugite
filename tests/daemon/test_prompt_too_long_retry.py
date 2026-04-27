"""The daemon auto-compacts-and-retries on "prompt too long" at
`tsugite/daemon/adapters/base.py:486`. If the first attempt already executed
tool calls with side effects before overflowing, a second full agent.run with
the same user message will happily re-issue those calls. Gate the retry on
whether any code_execution event was recorded during the first pass.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tsugite.daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite.daemon.config import AgentConfig
from tsugite.daemon.session_store import SessionStore
from tsugite.exceptions import AgentExecutionError


class _StubAdapter(BaseAdapter):
    def get_platform_name(self) -> str:
        return "test"

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


@pytest.fixture
def adapter(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "agent.md").write_text("---\nname: agent\n---\n\nHi.\n")

    store = SessionStore(tmp_path / "store.json")
    config = AgentConfig(workspace_dir=ws, agent_file=str(ws / "agent.md"))

    return _StubAdapter(
        agent_name="test-agent",
        agent_config=config,
        session_store=store,
    )


@pytest.mark.asyncio
async def test_retry_skipped_after_side_effecting_code_ran(adapter, tmp_path, monkeypatch):
    """run_agent raises 'prompt too long' AFTER a code_execution event was
    recorded. The adapter must not silently retry — that would re-issue the
    tool call.
    """
    session = adapter.session_store.get_or_create_interactive("alice", "test-agent")
    call_count = {"n": 0}

    def fake_run_agent(*args, **kwargs):
        call_count["n"] += 1
        # First call: record a code_execution event (simulating a POST tool call
        # that already committed side effects), then raise prompt-too-long.
        if call_count["n"] == 1:
            adapter.session_store.append_event(
                session.id,
                {"type": "code_execution", "code": "x=1", "output": "ok"},
            )
            raise AgentExecutionError("context length exceeded, prompt too long")
        # If we get here, the guard failed — the retry ran.
        return MagicMock(token_count=0, cost=0, execution_steps=[])

    # Make _run_compaction a no-op that returns the same conv_id (we don't care
    # about compaction mechanics for this test).
    async def fake_compaction(*args, **kwargs):
        return session.id

    monkeypatch.setattr("tsugite.daemon.adapters.base.run_agent", fake_run_agent)
    monkeypatch.setattr(adapter, "_run_compaction", fake_compaction)
    # Avoid schema / agent-file complexity.
    monkeypatch.setattr(adapter, "_resolve_agent_path", lambda: Path(adapter.agent_config.agent_file))
    monkeypatch.setattr(adapter, "_build_message_context", lambda msg, *a, **kw: msg)
    monkeypatch.setattr(adapter, "_build_agent_context", lambda *a, **kw: {})
    monkeypatch.setattr(adapter, "_get_workspace_attachments", lambda: [])
    monkeypatch.setattr(adapter, "_save_history", lambda **kw: None)
    monkeypatch.setattr(adapter, "_update_skill_ttl", lambda *a, **kw: None)

    channel_context = ChannelContext(
        source="http",
        channel_id=None,
        user_id="alice",
        reply_to="http:alice",
        metadata={},
    )

    with pytest.raises(AgentExecutionError):
        await adapter.handle_message(
            user_id="alice",
            message="post something",
            channel_context=channel_context,
        )

    assert call_count["n"] == 1, (
        f"run_agent fired {call_count['n']} times; should stop after first attempt recorded a code_execution event"
    )


@pytest.mark.asyncio
async def test_retry_still_fires_when_no_side_effects_yet(adapter, tmp_path, monkeypatch):
    """If prompt-too-long fires before any code executed, the retry is safe
    and should still happen (preserves the existing recovery behavior).
    """
    session = adapter.session_store.get_or_create_interactive("bob", "test-agent")
    call_count = {"n": 0}

    def fake_run_agent(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # No code_execution event recorded — purely a prompt-size failure.
            raise AgentExecutionError("prompt too long")
        return MagicMock(token_count=0, cost=0, execution_steps=[], __str__=lambda self: "ok")

    async def fake_compaction(*args, **kwargs):
        return session.id

    monkeypatch.setattr("tsugite.daemon.adapters.base.run_agent", fake_run_agent)
    monkeypatch.setattr(adapter, "_run_compaction", fake_compaction)
    monkeypatch.setattr(adapter, "_resolve_agent_path", lambda: Path(adapter.agent_config.agent_file))
    monkeypatch.setattr(adapter, "_build_message_context", lambda msg, *a, **kw: msg)
    monkeypatch.setattr(adapter, "_build_agent_context", lambda *a, **kw: {})
    monkeypatch.setattr(adapter, "_get_workspace_attachments", lambda: [])
    monkeypatch.setattr(adapter, "_save_history", lambda **kw: None)
    monkeypatch.setattr(adapter, "_update_skill_ttl", lambda *a, **kw: None)

    channel_context = ChannelContext(
        source="http",
        channel_id=None,
        user_id="bob",
        reply_to="http:bob",
        metadata={},
    )

    await adapter.handle_message(
        user_id="bob",
        message="anything",
        channel_context=channel_context,
    )

    assert call_count["n"] == 2, "retry should still fire when no side effects committed"
