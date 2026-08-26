"""Multi-step agents driven through the daemon adapter.

Multi-step runs are exercised through the CLI and the daemon is exercised with
single-step agents, so the intersection has no coverage: every bug found in it
so far was invisible to either half alone. This drives a real multi-step agent
through `handle_message` with the daemon's own `return_token_usage=True`, with
only the model call faked.
"""

from pathlib import Path

import pytest
from tsugite_daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite_daemon.config import RuntimeDefaults
from tsugite_daemon.session_store import SessionStore

TWO_STEP_AGENT = """---
name: two_step
model: ollama:qwen2.5-coder:7b
extends: none
tools: []
---
Preamble.

<!-- tsu:step name="gather" assign="findings" -->
Gather things.

<!-- tsu:step name="write" -->
Write up {{ findings }}.
"""


class _StubAdapter(BaseAdapter):
    def get_platform_name(self) -> str:
        return "test"

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    ws.mkdir()
    agent_file = ws / "agent.md"
    agent_file.write_text(TWO_STEP_AGENT)

    adapter = _StubAdapter(
        runtime=RuntimeDefaults(workspace_dir=ws, agent_file=str(agent_file)),
        session_store=SessionStore(tmp_path / "store.json"),
    )
    monkeypatch.setattr(adapter, "_resolve_agent_path", lambda: Path(adapter.runtime.agent_file))
    monkeypatch.setattr(adapter, "_build_message_context", lambda msg, *a, **kw: msg)
    monkeypatch.setattr(adapter, "_build_agent_context", lambda *a, **kw: {})
    monkeypatch.setattr(adapter, "_save_history", lambda **kw: None)
    monkeypatch.setattr(adapter, "_update_skill_ttl", lambda *a, **kw: None)
    return adapter


@pytest.fixture
def fake_model(monkeypatch):
    async def fake_agent_run(self, task, return_full_result=False, stream=False):
        self.total_tokens = 100
        self.total_cost = 0.25
        self.cost_reported = True
        self.last_input_tokens = 80
        self.cache_read_tokens = 10
        return "STEP_OUTPUT"

    monkeypatch.setattr("tsugite.core.agent.TsugiteAgent.run", fake_agent_run)


def _channel_context(user_id: str) -> ChannelContext:
    return ChannelContext(
        source="http",
        channel_id=None,
        user_id=user_id,
        reply_to=f"http:{user_id}",
        metadata={},
    )


@pytest.mark.asyncio
async def test_multistep_turn_completes_and_records_usage(adapter, fake_model):
    """The turn's work is done by the time the adapter reads the result, so a
    shape it cannot read throws away a finished run.
    """
    session = adapter.session_store.get_or_create_interactive("alice")

    reply = await adapter.handle_message(
        user_id="alice",
        message="do the thing",
        channel_context=_channel_context("alice"),
    )

    assert reply == "STEP_OUTPUT"
    assert adapter.session_store.get_session(session.id).cumulative_tokens == 80
