"""A turn's recorded cost is that turn's own cost, not the session running total.

`claude --print --resume <session>` reports `total_cost_usd` cumulative for the whole
resumed session. `ClaudeCodeProvider` opts out of the provider cache and is built fresh
every turn, so its delta baseline has to come from the prior turn's ending total, which
travels on the same `state_delta` channel as `session_id` and `compacted`.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tsugite.agent_runner.history_integration import ResumableSessionState, get_resumable_session_state
from tsugite.agent_runner.runner import run_agent_async


def _mock_process(turn_costs):
    """A fake ClaudeCodeProcess yielding one cumulative cost_usd per call."""
    process = AsyncMock()
    process.session_id = "test-session"
    process.compacted = False
    calls = iter(turn_costs)

    async def send(*args, **kwargs):
        cost = next(calls)
        yield {
            "type": "result",
            "text": "answer",
            "cost_usd": cost,
            "session_id": "test-session",
            "input_tokens": 500,
            "output_tokens": 50,
        }

    process.send_message = send
    process.start = AsyncMock()
    process.stop = AsyncMock()
    return process


@pytest.mark.asyncio
async def test_second_turn_cost_is_not_the_running_total():
    """Two daemon turns, each its own fresh ClaudeCodeProvider resuming the same
    CLI session. The CLI's second result (2.30) is cumulative for the whole
    session. The turn's own cost is 2.30 - 1.80 = 0.50."""
    from tsugite_claude_code.provider import ClaudeCodeProvider

    turn1 = ClaudeCodeProvider()
    with patch("tsugite_claude_code.process.ClaudeCodeProcess", return_value=_mock_process([1.80])):
        resp1 = await turn1.acompletion(messages=[{"role": "user", "content": "q1"}], model="sonnet", stream=False)
    assert resp1.cost == pytest.approx(1.80)

    turn2 = ClaudeCodeProvider()
    turn2.set_context(resume_session="test-session", resume_cost_baseline=turn1.get_state()["cumulative_cost"])
    with patch("tsugite_claude_code.process.ClaudeCodeProcess", return_value=_mock_process([2.30])):
        resp2 = await turn2.acompletion(messages=[{"role": "user", "content": "q2"}], model="sonnet", stream=False)

    assert resp2.cost == pytest.approx(0.50), (
        f"expected the turn's own delta (0.50), got {resp2.cost} (the running total)"
    )


@pytest.mark.asyncio
async def test_runner_forwards_the_resumed_sessions_cumulative_cost(tmp_path):
    """The runner must read cumulative_cost off the resumable session state and
    pass it to TsugiteAgent as resume_cost_baseline, the same way it already
    forwards resume_session."""
    agent_file = tmp_path / "agent.md"
    agent_file.write_text("---\nname: t\nmodel: claude_code:sonnet\n---\ndo the thing")

    captured = {}

    def capture_agent(**kwargs):
        captured.update(kwargs)
        mock = MagicMock()

        async def mock_run(prompt, return_full_result=False, stream=False):
            return "done"

        mock.run = MagicMock(side_effect=mock_run)
        return mock

    with (
        patch("tsugite.agent_runner.runner.TsugiteAgent", side_effect=capture_agent),
        patch(
            "tsugite.agent_runner.history_integration.get_resumable_session_state",
            return_value=ResumableSessionState(session_id="test-session", cumulative_cost=1.80),
        ),
        patch("tsugite.agent_runner.history_integration.load_and_apply_history", return_value=[]),
    ):
        await run_agent_async(agent_file, "new prompt", continue_conversation_id="conv-1")

    assert captured.get("resume_cost_baseline") == 1.80


def test_get_resumable_session_state_carries_cumulative_cost():
    """The resume-state reader must surface the prior turn's ending cumulative
    cost alongside session_id/compacted."""
    from tsugite.history import get_history_backend

    backend = get_history_backend()
    session = backend.create(agent_name="bot", model="claude_code:sonnet")
    session.record("user_input", text="hi")
    session.record(
        "model_response",
        raw_content="ok",
        state_delta={"session_id": "test-session", "compacted": False, "cumulative_cost": 1.80},
    )

    state = get_resumable_session_state(session.session_id)

    assert isinstance(state, ResumableSessionState)
    assert state.cumulative_cost == pytest.approx(1.80)
