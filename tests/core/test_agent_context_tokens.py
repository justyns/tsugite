"""Tests for context token tracking (last_input_tokens)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from tsugite.agent_runner.models import AgentExecutionResult
from tsugite.core.agent import AgentResult, TsugiteAgent
from tsugite.daemon.session_store import SessionStore
from tsugite.providers.base import CompletionResponse, Usage


def _mock_response(content: str, usage: Usage = None) -> CompletionResponse:
    return CompletionResponse(
        content=content,
        usage=usage or Usage(total_tokens=100, prompt_tokens=80, completion_tokens=20),
        cost=0.001,
    )


def _patch_provider(agent, side_effect=None, return_value=None):
    mock = AsyncMock(side_effect=side_effect, return_value=return_value)
    agent._provider = MagicMock()
    agent._provider.acompletion = mock
    agent._provider.stop = AsyncMock()
    agent._provider.get_state = MagicMock(return_value=None)
    agent._provider.set_context = MagicMock()
    return mock


def _make_agent(**kwargs):
    defaults = dict(model_string="openai:gpt-4o-mini", tools=[], instructions="", max_turns=5)
    defaults.update(kwargs)
    return TsugiteAgent(**defaults)


@pytest.mark.asyncio
async def test_last_input_tokens_from_prompt_tokens():
    """last_input_tokens should reflect prompt_tokens from the last API call."""
    agent = _make_agent()

    call_count = 0

    async def mock_completion(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _mock_response(
                "Thought: step 1\n\n```python\nx = 1\nprint(x)\n```",
                Usage(prompt_tokens=5000, completion_tokens=200, total_tokens=5200),
            )
        return _mock_response(
            "Thought: done\n\n```python\nfinal_answer(42)\n```",
            Usage(prompt_tokens=6000, completion_tokens=150, total_tokens=6150),
        )

    _patch_provider(agent, side_effect=mock_completion)
    result = await agent.run("test", return_full_result=True)

    assert isinstance(result, AgentResult)
    # Should be the last turn's prompt_tokens, not cumulative
    assert agent.last_input_tokens == 6000
    assert result.last_input_tokens == 6000


@pytest.mark.asyncio
async def test_last_input_tokens_includes_cache_tokens():
    """last_input_tokens should include cache_creation + cache_read tokens."""
    agent = _make_agent()

    _patch_provider(
        agent,
        side_effect=[
            _mock_response(
                "Thought: step\n\n```python\nprint(1)\n```",
                Usage(
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=5150,
                    cache_creation_input_tokens=2000,
                    cache_read_input_tokens=3000,
                ),
            ),
            _mock_response(
                "Thought: done\n\n```python\nfinal_answer(1)\n```",
                Usage(
                    prompt_tokens=200,
                    completion_tokens=50,
                    total_tokens=5250,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=5000,
                ),
            ),
        ],
    )

    result = await agent.run("test", return_full_result=True)

    # 200 (prompt) + 0 (cache_creation) + 5000 (cache_read) = 5200
    assert agent.last_input_tokens == 5200
    assert result.last_input_tokens == 5200


@pytest.mark.asyncio
async def test_last_input_tokens_zero_prompt_with_cache():
    """When prompt_tokens=0 but cache tokens exist (claude_code pattern), should still report context size."""
    agent = _make_agent()

    _patch_provider(
        agent,
        return_value=_mock_response(
            "Thought: done\n\n```python\nfinal_answer(1)\n```",
            Usage(
                prompt_tokens=0,
                completion_tokens=100,
                total_tokens=8100,
                cache_creation_input_tokens=3000,
                cache_read_input_tokens=5000,
            ),
        ),
    )

    result = await agent.run("test", return_full_result=True)

    # 0 + 3000 + 5000 = 8000
    assert agent.last_input_tokens == 8000
    assert result.last_input_tokens == 8000


@pytest.mark.asyncio
async def test_last_input_tokens_no_cache():
    """When no cache tokens, last_input_tokens equals prompt_tokens."""
    agent = _make_agent()

    _patch_provider(
        agent,
        return_value=_mock_response(
            "Thought: done\n\n```python\nfinal_answer(1)\n```",
            Usage(prompt_tokens=15000, completion_tokens=500, total_tokens=15500),
        ),
    )

    result = await agent.run("test", return_full_result=True)

    assert agent.last_input_tokens == 15000
    assert result.last_input_tokens == 15000


def test_session_store_update_token_count_replaces(tmp_path):
    """update_token_count should replace cumulative_tokens (represents current context size)."""
    store = SessionStore(tmp_path / "store.json", context_limits={"agent": 200000})
    session = store.get_or_create_interactive("user", "agent")

    store.update_token_count(session.id, 10000)
    assert store.get_session(session.id).cumulative_tokens == 10000

    # Second call replaces, not accumulates
    store.update_token_count(session.id, 15000)
    assert store.get_session(session.id).cumulative_tokens == 15000


def test_context_tokens_prefers_last_input_tokens():
    """The adapter should prefer last_input_tokens over token_count for context tracking."""
    result = AgentExecutionResult(
        response="test",
        token_count=50000,
        last_input_tokens=17000,
    )

    # Simulate the adapter's logic
    last_input = getattr(result, "last_input_tokens", None)
    context_tokens = last_input if isinstance(last_input, int) and last_input > 0 else (result.token_count or 0)

    assert context_tokens == 17000


def test_context_tokens_falls_back_to_token_count():
    """When last_input_tokens is None, fall back to token_count."""
    result = AgentExecutionResult(
        response="test",
        token_count=50000,
        last_input_tokens=None,
    )

    last_input = getattr(result, "last_input_tokens", None)
    context_tokens = last_input if isinstance(last_input, int) and last_input > 0 else (result.token_count or 0)

    assert context_tokens == 50000
