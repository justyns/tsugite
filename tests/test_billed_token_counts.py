"""The displayed context size is the provider-billed count, not the len//4 estimate."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from tsugite_daemon.adapters.base import build_session_event_persister
from tsugite_daemon.session_store import SessionStore

from tsugite.core.agent import TsugiteAgent
from tsugite.events import EventBus, PromptSnapshotEvent
from tsugite.history import get_history_backend
from tsugite.providers.base import CompletionResponse, Usage


def _resp(content: str, prompt: int, read: int, creation: int) -> CompletionResponse:
    usage = Usage(
        prompt_tokens=prompt,
        completion_tokens=20,
        total_tokens=prompt + 20,
        cache_read_input_tokens=read,
        cache_creation_input_tokens=creation,
    )
    return CompletionResponse(content=content, usage=usage, cost=0.0)


def _billed(prompt: int, read: int, creation: int) -> int:
    return prompt + read + creation


async def _run(tmp_path, responses):
    """Run the agent with every prompt_snapshot fed to the daemon's persister.

    Returns (cumulative_tokens after each snapshot, session, recorded breakdowns, agent).
    """
    store = SessionStore(tmp_path / "store.json")
    session = store.get_or_create_interactive("alice")
    persist = build_session_event_persister(store, session.id)
    shown = []

    def on_event(event):
        if isinstance(event, PromptSnapshotEvent):
            persist({"type": "prompt_snapshot", "token_breakdown": event.token_breakdown})
            shown.append(store.get_session(session.id).cumulative_tokens)

    bus = EventBus()
    bus.subscribe(on_event)
    storage = get_history_backend().create(agent_name="t", model="openai:gpt-4o-mini")
    agent = TsugiteAgent(
        model_string="openai:gpt-4o-mini",
        tools=[],
        instructions="be a helpful assistant",
        max_turns=3,
        storage=storage,
        event_bus=bus,
    )
    agent._provider = MagicMock()
    agent._provider.acompletion = AsyncMock(side_effect=responses)
    agent._provider.stop = AsyncMock()
    agent._provider.get_state = MagicMock(return_value=None)
    agent._provider.set_context = MagicMock()

    await agent.run("do the task")
    snaps = [e.data["token_breakdown"] for e in storage.iter_events() if e.type == "prompt_snapshot"]
    return shown, store.get_session(session.id), snaps, agent


@pytest.mark.asyncio
async def test_single_call_shows_billed_count(tmp_path):
    shown, session, snaps, agent = await _run(tmp_path, [_resp("done", 1000, 5000, 300)])
    expected = _billed(1000, 5000, 300)

    assert shown[-1] == expected
    assert session.cumulative_tokens == expected
    assert snaps[-1]["total"] == expected
    assert snaps[-1]["cache_read"] == 5000
    assert snaps[-1]["cache_creation"] == 300
    assert snaps[-1]["unaccounted"] == expected - sum(c["tokens"] for c in snaps[-1]["categories"])
    assert agent.last_input_tokens == expected


@pytest.mark.asyncio
async def test_multi_step_turn_shows_each_calls_billed_count(tmp_path):
    first = _billed(1000, 0, 4000)
    last = _billed(1200, 4000, 100)
    shown, session, snaps, _ = await _run(
        tmp_path,
        [_resp('```python-exec\nprint("working")\n```', 1000, 0, 4000), _resp("done", 1200, 4000, 100)],
    )

    # Before the first call the total is an estimate and says so.
    assert snaps[0]["estimated"] is True
    # After call 1 and before call 2 the badge holds call 1's billed count; after call 2 it holds call 2's.
    assert first in shown
    assert shown[-1] == last
    assert session.cumulative_tokens == last
    assert snaps[-1]["total"] == last
    assert "estimated" not in snaps[-1]
