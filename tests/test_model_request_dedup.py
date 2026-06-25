"""model_request events store a message count, not the full sent array.

Storing the entire reconstructed `messages` array on every turn made an N-turn
session re-store the whole conversation N times (O(n^2) on disk). Reconstruction
never reads `model_request` (events_to_messages only handles user_input /
model_response / code_execution / format_error), so the array is pure bloat.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from tsugite.core.agent import TsugiteAgent
from tsugite.history import SessionStorage
from tsugite.providers.base import CompletionResponse, Usage


def _resp(content: str) -> CompletionResponse:
    return CompletionResponse(content=content, usage=Usage(total_tokens=10), cost=0.0)


def _patch(agent, *, return_value=None):
    mock = AsyncMock(return_value=return_value)
    agent._provider = MagicMock()
    agent._provider.acompletion = mock
    agent._provider.stop = AsyncMock()
    agent._provider.get_state = MagicMock(return_value=None)
    agent._provider.set_context = MagicMock()
    return mock


@pytest.mark.asyncio
async def test_model_request_stores_hash_not_messages(tmp_path: Path):
    storage = SessionStorage.create(agent_name="t", model="openai:gpt-4o-mini", session_path=tmp_path / "s.jsonl")
    agent = TsugiteAgent(model_string="openai:gpt-4o-mini", tools=[], instructions="", max_turns=5, storage=storage)
    mock = _patch(agent, return_value=_resp("just text, no code"))

    await agent.run("hello")

    reqs = [e for e in storage.iter_events() if e.type == "model_request"]
    assert len(reqs) == 1, f"expected exactly one model_request, got {len(reqs)}"
    data = reqs[0].data

    # The full messages array must NOT be persisted (that was the O(n^2) bloat).
    assert "messages" not in data

    # A count is what we keep instead, plus the tool names already stored.
    sent_messages = mock.await_args.kwargs["messages"]
    assert data.get("message_count") == len(sent_messages)
    assert "tool_names" in data
