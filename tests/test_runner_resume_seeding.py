"""When resuming a session-owning provider, the runner must still load serialized
history so the provider can fall back to it if the resume replay is rejected
(poisoned Claude Code sidecar transcript -> 400 on every send)."""

from unittest.mock import MagicMock, patch

import pytest

from tsugite.agent_runner.history_integration import ResumableSessionState
from tsugite.agent_runner.runner import run_agent_async


@pytest.mark.asyncio
async def test_previous_messages_loaded_even_when_resuming(tmp_path):
    agent_file = tmp_path / "agent.md"
    agent_file.write_text("---\nname: t\nmodel: openai:gpt-4o-mini\n---\ndo the thing")

    captured = {}

    def capture_agent(**kwargs):
        captured.update(kwargs)
        mock = MagicMock()

        async def mock_run(prompt, return_full_result=False, stream=False):
            return "done"

        mock.run = MagicMock(side_effect=mock_run)
        return mock

    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "reply"}]
    with (
        patch("tsugite.agent_runner.runner.TsugiteAgent", side_effect=capture_agent),
        patch(
            "tsugite.agent_runner.history_integration.get_resumable_session_state",
            return_value=ResumableSessionState(session_id="prov-123"),
        ),
        patch("tsugite.agent_runner.history_integration.load_and_apply_history", return_value=history),
    ):
        await run_agent_async(agent_file, "new prompt", continue_conversation_id="conv-1")

    assert captured.get("resume_session") == "prov-123"
    assert captured.get("previous_messages") == history
