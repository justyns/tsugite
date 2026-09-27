"""The user_input dedup guard after turns closed by boot recovery or copied by compaction."""

from contextlib import ExitStack
from datetime import datetime, timezone

import pytest
from tsugite_daemon.config import RuntimeDefaults
from tsugite_daemon.session_store import SessionStore

from tests.test_post_compaction_counters import _patches, _seed_session_events, _StubAdapter
from tsugite.agent_runner.history_integration import record_user_input
from tsugite.history import get_history_backend
from tsugite.history.models import Event


def _user_input_texts(session_id):
    return [e.data["text"] for e in get_history_backend().load(session_id).iter_events() if e.type == "user_input"]


def test_message_after_daemon_interrupted_turn_is_recorded(tmp_path, history_dir):
    store = SessionStore(tmp_path / "session_store.json")
    session = store.get_or_create_interactive("test-user")
    get_history_backend().ensure_session(session.id).record("user_input", text="interrupted question")

    store._finalize_interrupted_turn(session.id)
    record_user_input(get_history_backend().load(session.id), "next question")

    assert _user_input_texts(session.id) == ["interrupted question", "next question"]


async def _compact_with_retained(tmp_path, history_dir, recent_events):
    store = SessionStore(tmp_path / "session_store.json")
    session = store.get_or_create_interactive("test-user")
    _seed_session_events(session.id)
    adapter = _StubAdapter(RuntimeDefaults(workspace_dir=tmp_path / "workspace", agent_file="default"), store)
    with ExitStack() as stack:
        for p in _patches(history_dir, recent_events=recent_events):
            stack.enter_context(p)
        return await adapter._compact_session(session.id, reason="manual")


@pytest.mark.asyncio
async def test_message_after_retained_failed_turn_is_recorded(tmp_path, history_dir):
    now = datetime.now(timezone.utc)
    successor = await _compact_with_retained(
        tmp_path,
        history_dir,
        [
            Event(type="user_input", ts=now, data={"text": "a message that errored"}),
            Event(type="model_response", ts=now, data={"raw_content": "[Error: boom]"}),
            Event(type="session_end", ts=now, data={"status": "error", "error_message": "boom"}),
        ],
    )

    record_user_input(get_history_backend().load(successor.id), "the next real message")

    assert _user_input_texts(successor.id) == ["a message that errored", "the next real message"]


@pytest.mark.asyncio
async def test_retry_after_prompt_too_long_compaction_is_not_recorded_twice(tmp_path, history_dir):
    """The in-flight turn's user_input is retained into the successor, and the retry records it again."""
    now = datetime.now(timezone.utc)
    successor = await _compact_with_retained(
        tmp_path, history_dir, [Event(type="user_input", ts=now, data={"text": "too long question"})]
    )

    record_user_input(get_history_backend().load(successor.id), "too long question")

    assert _user_input_texts(successor.id) == ["too long question"]
