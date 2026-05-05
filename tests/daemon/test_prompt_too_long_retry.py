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


@pytest.fixture
def stub_adapter_internals(adapter, monkeypatch):
    """Stub out adapter side-effect methods irrelevant to retry/compaction
    behavior so tests can focus on `run_agent` + `_run_compaction` interplay.
    """
    monkeypatch.setattr(adapter, "_resolve_agent_path", lambda: Path(adapter.agent_config.agent_file))
    monkeypatch.setattr(adapter, "_build_message_context", lambda msg, *a, **kw: msg)
    monkeypatch.setattr(adapter, "_build_agent_context", lambda *a, **kw: {})
    monkeypatch.setattr(adapter, "_get_workspace_attachments", lambda: [])
    monkeypatch.setattr(adapter, "_save_history", lambda **kw: None)
    monkeypatch.setattr(adapter, "_update_skill_ttl", lambda *a, **kw: None)
    return adapter


def _channel_context(user_id: str, *, conv_id_override: str | None = None) -> ChannelContext:
    metadata = {"conv_id_override": conv_id_override} if conv_id_override else {}
    return ChannelContext(
        source="http",
        channel_id=None,
        user_id=user_id,
        reply_to=f"http:{user_id}",
        metadata=metadata,
    )


@pytest.mark.asyncio
async def test_retry_skipped_after_side_effecting_code_ran(stub_adapter_internals, monkeypatch):
    """run_agent raises 'prompt too long' AFTER a code_execution event was
    recorded. The adapter must not silently retry — that would re-issue the
    tool call.
    """
    adapter = stub_adapter_internals
    session = adapter.session_store.get_or_create_interactive("alice", "test-agent")
    call_count = {"n": 0}

    def fake_run_agent(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            adapter.session_store.append_event(
                session.id,
                {"type": "code_execution", "code": "x=1", "output": "ok"},
            )
            raise AgentExecutionError("Prompt is too long (subtype=success)")
        return MagicMock(token_count=0, cost=0, execution_steps=[])

    async def fake_compaction(*args, **kwargs):
        return session.id

    monkeypatch.setattr("tsugite.daemon.adapters.base.run_agent", fake_run_agent)
    monkeypatch.setattr(adapter, "_run_compaction", fake_compaction)

    with pytest.raises(AgentExecutionError):
        await adapter.handle_message(
            user_id="alice",
            message="post something",
            channel_context=_channel_context("alice"),
        )

    assert call_count["n"] == 1, (
        f"run_agent fired {call_count['n']} times; should stop after first attempt recorded a code_execution event"
    )


@pytest.mark.asyncio
async def test_retry_still_fires_when_no_side_effects_yet(stub_adapter_internals, monkeypatch):
    """If prompt-too-long fires before any code executed, the retry is safe
    and should still happen (preserves the existing recovery behavior).
    """
    adapter = stub_adapter_internals
    session = adapter.session_store.get_or_create_interactive("bob", "test-agent")
    call_count = {"n": 0}

    def fake_run_agent(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise AgentExecutionError("Prompt is too long (subtype=success)")
        return MagicMock(token_count=0, cost=0, execution_steps=[], __str__=lambda self: "ok")

    async def fake_compaction(*args, **kwargs):
        return session.id

    monkeypatch.setattr("tsugite.daemon.adapters.base.run_agent", fake_run_agent)
    monkeypatch.setattr(adapter, "_run_compaction", fake_compaction)

    await adapter.handle_message(
        user_id="bob",
        message="anything",
        channel_context=_channel_context("bob"),
    )

    assert call_count["n"] == 2, "retry should still fire when no side effects committed"


@pytest.mark.asyncio
async def test_retry_fires_for_pinned_session_when_no_side_effects(stub_adapter_internals, monkeypatch):
    """Pinned (and any explicitly-selected) sessions go through the daemon with
    metadata["conv_id_override"] set. The reactive auto-compact-and-retry must
    still fire for them when no code_execution has happened yet — otherwise the
    user just sees a raw 'Prompt is too long' error with no recovery.
    """
    adapter = stub_adapter_internals
    session = adapter.session_store.get_or_create_interactive("carol", "test-agent")
    call_count = {"n": 0}

    def fake_run_agent(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise AgentExecutionError("Prompt is too long (subtype=success)")
        return MagicMock(token_count=0, cost=0, execution_steps=[], __str__=lambda self: "ok")

    async def fake_compaction(*args, **kwargs):
        return session.id

    monkeypatch.setattr("tsugite.daemon.adapters.base.run_agent", fake_run_agent)
    monkeypatch.setattr(adapter, "_run_compaction", fake_compaction)

    await adapter.handle_message(
        user_id="carol",
        message="anything",
        channel_context=_channel_context("carol", conv_id_override=session.id),
    )

    assert call_count["n"] == 2, "retry should fire even when conv_id_override is set"


@pytest.mark.asyncio
async def test_retry_skipped_for_pinned_session_after_side_effects(stub_adapter_internals, monkeypatch):
    """Even with conv_id_override set (pinned session), the side-effect guard
    must still win: if a code_execution event was recorded before the prompt
    overflow, do not retry — that would re-issue the tool call.
    """
    adapter = stub_adapter_internals
    session = adapter.session_store.get_or_create_interactive("dave", "test-agent")
    call_count = {"n": 0}

    def fake_run_agent(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            adapter.session_store.append_event(
                session.id,
                {"type": "code_execution", "code": "x=1", "output": "ok"},
            )
            raise AgentExecutionError("Prompt is too long (subtype=success)")
        return MagicMock(token_count=0, cost=0, execution_steps=[])

    async def fake_compaction(*args, **kwargs):
        return session.id

    monkeypatch.setattr("tsugite.daemon.adapters.base.run_agent", fake_run_agent)
    monkeypatch.setattr(adapter, "_run_compaction", fake_compaction)

    with pytest.raises(AgentExecutionError):
        await adapter.handle_message(
            user_id="dave",
            message="post something",
            channel_context=_channel_context("dave", conv_id_override=session.id),
        )

    assert call_count["n"] == 1, "side-effect guard must still skip retry for pinned sessions"


@pytest.mark.asyncio
async def test_proactive_compaction_runs_for_override_session(stub_adapter_internals, monkeypatch):
    """When a pinned session is over the 80% token threshold, sending a new
    message must trigger proactive compaction before invoking the agent. This
    path is what prevents 'Prompt is too long' from ever being raised on a
    healthy turn — without it, the override session grows unbounded.
    """
    adapter = stub_adapter_internals
    session = adapter.session_store.get_or_create_interactive("erin", "test-agent")
    # Small context limit so seeded cumulative_tokens cross the 80% threshold.
    adapter.session_store.update_context_limit("test-agent", 1000)
    adapter.session_store.set_cumulative_tokens(session.id, 900)
    assert adapter.session_store.needs_compaction(session.id), "fixture precondition"

    compact_calls: list[str] = []

    async def fake_compaction(user_id, conv_id, custom_logger=None, reason=None):
        compact_calls.append(conv_id)
        return conv_id

    def fake_run_agent(*args, **kwargs):
        return MagicMock(token_count=0, cost=0, execution_steps=[], __str__=lambda self: "ok")

    monkeypatch.setattr("tsugite.daemon.adapters.base.run_agent", fake_run_agent)
    monkeypatch.setattr(adapter, "_run_compaction", fake_compaction)

    await adapter.handle_message(
        user_id="erin",
        message="anything",
        channel_context=_channel_context("erin", conv_id_override=session.id),
    )

    assert compact_calls == [session.id], "proactive compaction must run for override sessions over threshold"
