"""Auto-titling uses the configured compaction model."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from tsugite_daemon.adapters.base import BaseAdapter
from tsugite_daemon.config import RuntimeDefaults
from tsugite_daemon.memory import SHORT_TITLE_THRESHOLD, compute_session_title
from tsugite_daemon.session_runner import SessionRunner

LONG_MESSAGE = "x" * (SHORT_TITLE_THRESHOLD + 1)


@pytest.fixture
def captured_models(monkeypatch):
    models: list[str] = []

    async def fake_llm_complete(system_prompt, user_content, model):
        models.append(model)
        return "A Generated Title"

    monkeypatch.setattr("tsugite_daemon.memory._llm_complete", fake_llm_complete)
    return models


def _runtime(compaction_model):
    return RuntimeDefaults(workspace_dir=Path("/tmp"), agent_file="agent.md", compaction_model=compaction_model)


@pytest.mark.asyncio
async def test_configured_compaction_model_is_used(captured_models):
    title = await compute_session_title(
        LONG_MESSAGE, "response", "codex_cli:gpt-5", compaction_model="openai:gpt-4o-mini"
    )
    assert title == "A Generated Title"
    assert captured_models == ["openai:gpt-4o-mini"]


@pytest.mark.asyncio
async def test_falls_back_to_inferred_model(captured_models):
    await compute_session_title(LONG_MESSAGE, "response", "openai:gpt-4o")
    assert captured_models == ["openai:gpt-4o-mini"]


@pytest.mark.asyncio
async def test_short_message_skips_the_llm(captured_models):
    title = await compute_session_title(
        "short prompt", "response", "codex_cli:gpt-5", compaction_model="openai:gpt-4o-mini"
    )
    assert title == "short prompt"
    assert captured_models == []


@pytest.mark.asyncio
async def test_adapter_passes_configured_compaction_model(captured_models):
    updates = {}
    adapter = SimpleNamespace(
        runtime=_runtime("openai:gpt-4o-mini"),
        resolve_model=lambda: "codex_cli:gpt-5",
        event_bus=None,
        session_store=SimpleNamespace(update_session=lambda session_id, **kw: updates.update(kw)),
    )

    await BaseAdapter._auto_title_session(adapter, "s-1", LONG_MESSAGE, "response")

    assert captured_models == ["openai:gpt-4o-mini"]
    assert updates == {"title": "A Generated Title"}


@pytest.mark.asyncio
async def test_session_runner_passes_configured_compaction_model(captured_models):
    renamed = []
    adapter = SimpleNamespace(runtime=_runtime("openai:gpt-4o-mini"), resolve_model=lambda: "codex_cli:gpt-5")
    runner = SimpleNamespace(rename_session=lambda session_id, title: renamed.append((session_id, title)))
    session = SimpleNamespace(id="s-2", prompt=LONG_MESSAGE)

    await SessionRunner._auto_title_background_session(runner, session, "response", adapter)

    assert captured_models == ["openai:gpt-4o-mini"]
    assert renamed == [("s-2", "A Generated Title")]
