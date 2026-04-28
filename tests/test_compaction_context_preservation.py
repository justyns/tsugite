"""Tests for post-compaction context_window display.

Compaction uses a smaller model (e.g. claude-3-haiku, 200k) than the agent's
main model (e.g. sonnet 1m, 1M). Two regression tests:

1. `_llm_complete` must not leak the compact-model's context_window into a
   shared provider instance's internal state (root cause).
2. `_compact_session` must end with the agent's `context_limit` unchanged,
   even if some path mutates it during summarization (display defense).
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from tsugite.daemon.adapters.base import BaseAdapter
from tsugite.daemon.config import AgentConfig
from tsugite.daemon.memory import _llm_complete
from tsugite.daemon.session_store import SessionStore
from tsugite.history.models import Event


class _StubAdapter(BaseAdapter):
    async def start(self):
        pass

    async def stop(self):
        pass


class _FakeProvider:
    """Mimics Claude Code provider's `_context_window` mutation pattern."""

    def __init__(self, mutate_to: int | None = None):
        self._context_window: int | None = None
        self._mutate_to = mutate_to

    async def acompletion(self, messages, model):
        # Simulate Claude Code: response stream sets _context_window mid-call.
        if self._mutate_to is not None:
            self._context_window = self._mutate_to
        return SimpleNamespace(content="summary text")

    def count_tokens(self, text, model):
        return max(1, len(text) // 4)


@pytest.mark.asyncio
async def test_llm_complete_does_not_pollute_provider_context_window():
    """Provider's `_context_window` must be unchanged after `_llm_complete`.

    Without the fix, summarization with a smaller model leaks its context window
    into the shared provider instance, so the next agent turn reads stale state
    and reports the wrong context limit.
    """
    provider = _FakeProvider(mutate_to=200_000)
    provider._context_window = 1_000_000  # Agent's main model already set this.

    with patch(
        "tsugite.models.get_provider_and_model",
        return_value=("anthropic", provider, "claude-3-haiku-20240307"),
    ):
        await _llm_complete("system", "user content", "anthropic:claude-3-haiku-20240307")

    assert provider._context_window == 1_000_000, (
        "summarization call leaked compact-model context_window into provider state"
    )


@pytest.mark.asyncio
async def test_llm_complete_preserves_unset_context_window():
    """Providers without a prior `_context_window` value must still come out clean."""
    provider = _FakeProvider(mutate_to=128_000)
    # _context_window starts as None.

    with patch(
        "tsugite.models.get_provider_and_model",
        return_value=("openai", provider, "gpt-4o-mini"),
    ):
        await _llm_complete("system", "user content", "openai:gpt-4o-mini")

    assert provider._context_window is None


@pytest.mark.asyncio
async def test_llm_complete_no_op_when_provider_lacks_context_window():
    """Providers without a `_context_window` attribute (most non-Claude-Code) work fine."""

    class _PlainProvider:
        async def acompletion(self, messages, model):
            return SimpleNamespace(content="ok")

        def count_tokens(self, text, model):
            return max(1, len(text) // 4)

    provider = _PlainProvider()
    with patch(
        "tsugite.models.get_provider_and_model",
        return_value=("openai", provider, "gpt-4o-mini"),
    ):
        result = await _llm_complete("system", "user content", "openai:gpt-4o-mini")

    assert result == "ok"
    assert not hasattr(provider, "_context_window")


@pytest.fixture
def workspace_dir(tmp_path):
    return tmp_path / "workspace"


@pytest.mark.asyncio
async def test_compact_session_preserves_context_limit(workspace_dir, tmp_path):
    """After `_compact_session`, both `session_store` and `agent_config`
    context limits should equal their pre-compaction values, even if
    summarization's path mutates them mid-call.
    """
    from tsugite.history import SessionStorage

    history_dir = tmp_path / "history"
    history_dir.mkdir()

    INITIAL_LIMIT = 1_000_000  # sonnet 1m

    store = SessionStore(tmp_path / "session_store.json", context_limits={"test-agent": INITIAL_LIMIT})
    session = store.get_or_create_interactive("test-user", "test-agent")
    conv_id = session.id

    session_path = history_dir / f"{conv_id}.jsonl"
    storage = SessionStorage.create(
        agent_name="test-agent",
        model="anthropic:claude-sonnet-4-5",
        session_path=session_path,
    )
    for i in range(6):
        storage.record("user_input", text=f"message {i}")
        storage.record("model_response", raw_content=f"reply {i}")

    agent_config = AgentConfig(workspace_dir=workspace_dir, agent_file="default")
    agent_config.context_limit = INITIAL_LIMIT
    adapter = _StubAdapter("test-agent", agent_config, store)

    old_events = [
        Event(type="user_input", ts=datetime.now(timezone.utc), data={"text": f"old {i}"}) for i in range(4)
    ]
    recent_events = [
        Event(type="user_input", ts=datetime.now(timezone.utc), data={"text": f"recent {i}"}) for i in range(2)
    ]

    async def fake_summarize_with_pollution(messages, model=None, max_context_tokens=None, progress_callback=None):
        # Simulate the bug: something during summarization writes the smaller
        # compact-model context_window onto the agent's tracked limits.
        store.update_context_limit("test-agent", 200_000)
        agent_config.context_limit = 200_000
        return "Summary"

    with (
        patch("tsugite.daemon.memory.get_context_limit", return_value=200_000),
        patch("tsugite.daemon.memory.infer_compaction_model", return_value="anthropic:claude-3-haiku-20240307"),
        patch(
            "tsugite.daemon.memory.split_events_for_compaction",
            return_value=(old_events, recent_events),
        ),
        patch("tsugite.daemon.memory.summarize_session", new=fake_summarize_with_pollution),
        patch("tsugite.history.get_history_dir", return_value=history_dir),
        patch("tsugite.history.storage.get_history_dir", return_value=history_dir),
        patch("tsugite.history.storage.get_machine_name", return_value="test"),
        patch("tsugite.hooks.fire_compact_hooks", new_callable=AsyncMock, return_value=[]),
    ):
        await adapter._compact_session(conv_id)

    assert store.get_context_limit("test-agent") == INITIAL_LIMIT, (
        "session_store context_limit was polluted by compaction"
    )
    assert agent_config.context_limit == INITIAL_LIMIT, (
        "agent_config context_limit was polluted by compaction"
    )


@pytest.mark.asyncio
async def test_compact_session_restores_even_on_summarize_failure(workspace_dir, tmp_path):
    """If summarization raises, context_limit must still be restored."""
    from tsugite.history import SessionStorage

    history_dir = tmp_path / "history"
    history_dir.mkdir()

    INITIAL_LIMIT = 1_000_000

    store = SessionStore(tmp_path / "session_store.json", context_limits={"test-agent": INITIAL_LIMIT})
    session = store.get_or_create_interactive("test-user", "test-agent")
    conv_id = session.id

    session_path = history_dir / f"{conv_id}.jsonl"
    storage = SessionStorage.create(
        agent_name="test-agent",
        model="anthropic:claude-sonnet-4-5",
        session_path=session_path,
    )
    for i in range(4):
        storage.record("user_input", text=f"message {i}")
        storage.record("model_response", raw_content=f"reply {i}")

    agent_config = AgentConfig(workspace_dir=workspace_dir, agent_file="default")
    agent_config.context_limit = INITIAL_LIMIT
    adapter = _StubAdapter("test-agent", agent_config, store)

    old_events = [
        Event(type="user_input", ts=datetime.now(timezone.utc), data={"text": f"old {i}"}) for i in range(2)
    ]
    recent_events = [
        Event(type="user_input", ts=datetime.now(timezone.utc), data={"text": f"recent {i}"}) for i in range(2)
    ]

    async def failing_summarize(messages, model=None, max_context_tokens=None, progress_callback=None):
        store.update_context_limit("test-agent", 200_000)
        agent_config.context_limit = 200_000
        raise RuntimeError("simulated summarization failure")

    with (
        patch("tsugite.daemon.memory.get_context_limit", return_value=200_000),
        patch("tsugite.daemon.memory.infer_compaction_model", return_value="anthropic:claude-3-haiku-20240307"),
        patch(
            "tsugite.daemon.memory.split_events_for_compaction",
            return_value=(old_events, recent_events),
        ),
        patch("tsugite.daemon.memory.summarize_session", new=failing_summarize),
        patch("tsugite.history.get_history_dir", return_value=history_dir),
        patch("tsugite.history.storage.get_history_dir", return_value=history_dir),
        patch("tsugite.history.storage.get_machine_name", return_value="test"),
        patch("tsugite.hooks.fire_compact_hooks", new_callable=AsyncMock, return_value=[]),
    ):
        with pytest.raises(RuntimeError, match="simulated summarization failure"):
            await adapter._compact_session(conv_id)

    assert store.get_context_limit("test-agent") == INITIAL_LIMIT
    assert agent_config.context_limit == INITIAL_LIMIT
