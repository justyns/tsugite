"""A reasoning block reaches the UI handler once, from the live emit."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tsugite.agent_runner.runner import run_agent_async
from tsugite.events import ReasoningContentEvent


@pytest.mark.asyncio
async def test_reasoning_block_is_delivered_once(tmp_path):
    agent_file = tmp_path / "agent.md"
    agent_file.write_text("---\nname: t\nmodel: openai:gpt-4o-mini\n---\ndo the thing")

    received = []
    handler = SimpleNamespace(handle_event=received.append)

    def make_agent(**kwargs):
        bus = kwargs["event_bus"]
        agent = MagicMock()
        agent.memory.reasoning_history = ["thinking hard"]

        async def run(prompt, return_full_result=False, stream=False):
            bus.emit(ReasoningContentEvent(content="thinking hard", step=1))
            return "done"

        agent.run = run
        return agent

    with patch("tsugite.agent_runner.runner.TsugiteAgent", side_effect=make_agent):
        await run_agent_async(agent_file, "go", custom_logger=SimpleNamespace(ui_handler=handler))

    reasoning = [e for e in received if isinstance(e, ReasoningContentEvent)]
    assert [e.content for e in reasoning] == ["thinking hard"]
