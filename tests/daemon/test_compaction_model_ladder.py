"""Walking an ordered list of compaction models."""

import logging
from contextlib import ExitStack

import pytest
from tsugite_daemon.config import RuntimeDefaults
from tsugite_daemon.session_store import SessionStore

from tests.history_helpers import load_history_session
from tests.test_post_compaction_counters import _patches, _seed_session_events, _StubAdapter

LADDER = ["codex_cli:gpt-5-mini", "claude_code:haiku"]


def _summarizer(calls: list, fail: dict[str, str]):
    async def fake(messages, model=None, max_context_tokens=None, progress_callback=None):
        calls.append(model)
        if isinstance(model, str) and model in fail:
            raise RuntimeError(f"LLM call failed ({model}): {fail[model]}")
        return f"Summary from {model}"

    return fake


async def _compact(tmp_path, history_dir, compaction_model, fake_summarize) -> str:
    store = SessionStore(tmp_path / "session_store.json", default_context_limit=128_000)
    session = store.get_or_create_interactive("test-user")
    _seed_session_events(session.id)
    runtime = RuntimeDefaults(
        workspace_dir=tmp_path / "workspace", agent_file="default", compaction_model=compaction_model
    )
    adapter = _StubAdapter(runtime, store)
    with ExitStack() as stack:
        for p in _patches(history_dir, summarize=fake_summarize):
            stack.enter_context(p)
        return await adapter._run_compaction("test-user", session.id, reason="manual")


@pytest.mark.asyncio
async def test_list_config_falls_through_when_first_model_is_unavailable(tmp_path, history_dir):
    calls = []
    fake = _summarizer(calls, fail={LADDER[0]: "The 'gpt-5-mini' model is not supported"})

    new_id = await _compact(tmp_path, history_dir, LADDER, fake)

    assert calls == LADDER
    events = load_history_session(new_id).load_events()
    assert next(e.data["summary"] for e in events if e.type == "compaction") == "Summary from claude_code:haiku"


@pytest.mark.asyncio
async def test_string_config_tries_one_model_and_raises_its_error(tmp_path, history_dir):
    calls = []
    fake = _summarizer(calls, fail={"openai:gpt-4o-mini": "The model is not supported"})

    with pytest.raises(RuntimeError, match="LLM call failed \\(openai:gpt-4o-mini\\)"):
        await _compact(tmp_path, history_dir, "openai:gpt-4o-mini", fake)

    assert calls == ["openai:gpt-4o-mini"]


@pytest.mark.asyncio
async def test_context_overflow_does_not_advance_the_ladder(tmp_path, history_dir):
    calls = []
    fake = _summarizer(calls, fail={LADDER[0]: "prompt is too long"})

    with pytest.raises(RuntimeError, match="prompt is too long"):
        await _compact(tmp_path, history_dir, LADDER, fake)

    assert calls == [LADDER[0]]


@pytest.mark.asyncio
async def test_exhausted_ladder_raises_and_logs_each_abandoned_model(tmp_path, history_dir, caplog):
    calls = []
    fake = _summarizer(calls, fail={LADDER[0]: "model not supported", LADDER[1]: "401 unauthorized"})

    with caplog.at_level(logging.WARNING, logger="tsugite_daemon.memory"):
        with pytest.raises(RuntimeError, match="401 unauthorized"):
            await _compact(tmp_path, history_dir, LADDER, fake)

    assert calls == LADDER
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(LADDER[0] in w and "model not supported" in w for w in warnings)
