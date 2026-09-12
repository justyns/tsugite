"""One turn, one usage row, credited to the agent that ran.

These drive `handle_message` against a real UsageStore with only the model call
faked, so a second writer anywhere on the path shows up as an extra row.
"""

from pathlib import Path

import pytest
from tsugite_daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite_daemon.config import RuntimeDefaults
from tsugite_daemon.session_store import SessionStore

from tsugite.providers.base import Usage

AGENT = """---
name: sched_agent
model: ollama:qwen2.5-coder:7b
extends: none
tools: []
---
Do it.
"""


class _StubAdapter(BaseAdapter):
    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    ws.mkdir()
    agent_file = ws / "agent.md"
    agent_file.write_text(AGENT)

    adapter = _StubAdapter(
        runtime=RuntimeDefaults(workspace_dir=ws, agent_file=str(agent_file)),
        session_store=SessionStore(tmp_path / "store.json"),
    )
    monkeypatch.setattr(adapter, "_resolve_agent_path", lambda *a: Path(adapter.runtime.agent_file))
    monkeypatch.setattr(adapter, "_build_message_context", lambda msg, *a, **kw: msg)
    monkeypatch.setattr(adapter, "_build_agent_context", lambda *a, **kw: {})
    monkeypatch.setattr(adapter, "_save_history", lambda **kw: None)
    monkeypatch.setattr(adapter, "_update_skill_ttl", lambda *a, **kw: None)
    return adapter


@pytest.fixture
def fake_model(monkeypatch):
    """Faked model call that still reports usage through `_accumulate_usage`."""

    async def fake_agent_run(self, task, return_full_result=False, stream=False):
        self._accumulate_usage(
            Usage(prompt_tokens=80, completion_tokens=20, total_tokens=100, cache_read_input_tokens=10),
            cost=0.25,
        )
        return "OUT"

    monkeypatch.setattr("tsugite.core.agent.TsugiteAgent.run", fake_agent_run)


@pytest.fixture
def store(tmp_path, monkeypatch):
    from tsugite.usage.store import UsageStore

    s = UsageStore(tmp_path / "usage.db")
    monkeypatch.setattr("tsugite.usage.get_usage_store", lambda: s)
    return s


def _channel_context(source: str, metadata: dict | None = None) -> ChannelContext:
    return ChannelContext(
        source=source,
        channel_id=None,
        user_id="alice",
        reply_to=f"{source}:alice",
        metadata=metadata,
    )


@pytest.mark.asyncio
async def test_scheduled_turn_writes_one_row_credited_to_the_agent(adapter, fake_model, store):
    session = adapter.session_store.get_or_create_interactive("alice")
    await adapter.handle_message(
        user_id="alice",
        message="go",
        channel_context=_channel_context("scheduler", {"schedule_id": "nightly"}),
    )

    rows = store.query(limit=50)
    assert len(rows) == 1
    row = rows[0]
    assert row["agent"] == "sched_agent"
    assert row["source"] == "scheduler"
    assert row["schedule_name"] == "nightly"
    assert row["session_id"] == session.id
    assert row["total_tokens"] == 100
    assert row["input_tokens"] == 80
    assert row["output_tokens"] == 20
    assert row["cache_read_tokens"] == 10
    assert row["cost_usd"] == 0.25

    schedules = store.by_schedule()
    assert [(s["schedule_name"], s["runs"], s["total_tokens"]) for s in schedules] == [("nightly", 1, 100)]


@pytest.mark.asyncio
async def test_http_turn_writes_one_row_on_the_channel_source(adapter, fake_model, store):
    adapter.session_store.get_or_create_interactive("alice")
    await adapter.handle_message(
        user_id="alice",
        message="go",
        channel_context=_channel_context("http"),
    )

    rows = store.query(limit=50)
    assert len(rows) == 1
    assert rows[0]["source"] == "http"
    assert rows[0]["schedule_name"] is None


@pytest.mark.asyncio
async def test_cli_run_writes_one_row_sourced_cli(tmp_path, fake_model, store, monkeypatch):
    from tsugite.agent_runner import run_agent_async
    from tsugite.options import ExecutionOptions

    agent_file = tmp_path / "agent.md"
    agent_file.write_text(AGENT)

    await run_agent_async(
        agent_path=agent_file,
        prompt="go",
        exec_options=ExecutionOptions(return_token_usage=True),
    )

    rows = store.query(limit=50)
    assert len(rows) == 1
    assert rows[0]["source"] == "cli"
    assert rows[0]["agent"] == "sched_agent"
    assert rows[0]["input_tokens"] == 80
    assert rows[0]["output_tokens"] == 20
