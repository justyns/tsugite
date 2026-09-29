"""Events a compaction re-records into the successor are marked and excluded from counts."""

from contextlib import ExitStack

import pytest
from tsugite_daemon.config import RuntimeDefaults
from tsugite_daemon.session_store import SessionStore

from tests.history_helpers import load_history_session, seed_history_session
from tests.test_post_compaction_counters import _patches, _StubAdapter
from tsugite.history import events_to_messages
from tsugite.history.models import COMPACTION_COPY_KEY, SessionSummary

TURNS = 6
RETAINED_TURNS = 2
COST = 0.5


def _seed_turns(session_id):
    storage = seed_history_session(session_id, agent="test-agent", model="anthropic:claude-sonnet-4-5")
    for i in range(TURNS):
        storage.record("user_input", text=f"message {i}")
        storage.record("model_response", raw_content=f"reply {i}", cost=COST, usage={"total_tokens": 10})
        storage.record("code_execution", code="x", duration_ms=100)


async def _compact(session_id, store, workspace_dir, history_dir):
    """Compact ``session_id``, retaining its last RETAINED_TURNS turns (real rows, real ids)."""
    events = load_history_session(session_id).load_events()
    first_retained = [i for i, e in enumerate(events) if e.type == "user_input"][-RETAINED_TURNS]
    agent_config = RuntimeDefaults(workspace_dir=workspace_dir, agent_file="default")
    agent_config.context_limit = 1_000_000
    adapter = _StubAdapter(agent_config, store)
    with ExitStack() as stack:
        for p in _patches(history_dir, recent_events=events[first_retained:]):
            stack.enter_context(p)
        return await adapter._compact_session(session_id, reason="manual")


@pytest.fixture
def workspace_dir(tmp_path):
    return tmp_path / "workspace"


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "session_store.json", default_context_limit=1_000_000)


@pytest.mark.asyncio
async def test_retained_events_are_marked_as_copies_of_their_source(workspace_dir, history_dir, store):
    conv_id = store.get_or_create_interactive("test-user").id
    _seed_turns(conv_id)
    original = load_history_session(conv_id).load_events()

    new_session = await _compact(conv_id, store, workspace_dir, history_dir)

    new_events = load_history_session(new_session.id).load_events()
    copies = [e for e in new_events if COMPACTION_COPY_KEY in e.data]
    first_retained = [i for i, e in enumerate(original) if e.type == "user_input"][-RETAINED_TURNS]
    retained = original[first_retained:]
    assert len(copies) == RETAINED_TURNS * 3, f"expected every retained row marked, got {len(copies)}"
    assert [e.data[COMPACTION_COPY_KEY] for e in copies] == [e.id for e in retained]
    assert all(COMPACTION_COPY_KEY not in e.data for e in new_events if e.type in ("session_start", "compaction"))

    texts = "\n".join(str(m) for m in events_to_messages(new_events))
    assert "message 4" in texts and "reply 5" in texts, "successor lost the retained turns from its context"


@pytest.mark.asyncio
async def test_recompacting_points_copies_at_the_original_event(workspace_dir, history_dir, store):
    conv_id = store.get_or_create_interactive("test-user").id
    _seed_turns(conv_id)
    original_ids = {e.id for e in load_history_session(conv_id).load_events()}

    first = await _compact(conv_id, store, workspace_dir, history_dir)
    second = await _compact(first.id, store, workspace_dir, history_dir)

    second_events = load_history_session(second.id).load_events()
    copies = [e for e in second_events if COMPACTION_COPY_KEY in e.data]
    assert len(copies) == RETAINED_TURNS * 3
    assert next(e for e in second_events if e.type == "compaction").data["retained_count"] == RETAINED_TURNS
    assert {e.data[COMPACTION_COPY_KEY] for e in copies} <= original_ids


@pytest.mark.asyncio
async def test_counting_across_a_compacted_chain_does_not_double_count(workspace_dir, history_dir, store):
    conv_id = store.get_or_create_interactive("test-user").id
    _seed_turns(conv_id)
    backend_session = load_history_session(conv_id)
    new_session = await _compact(conv_id, store, workspace_dir, history_dir)
    chain = [backend_session, load_history_session(new_session.id)]

    assert sum(s.summary().turn_count for s in chain) == TURNS
    assert sum(s.summary().total_cost for s in chain) == pytest.approx(TURNS * COST)
    assert sum(s.summary().total_duration_ms for s in chain) == TURNS * 100
    assert sum(SessionSummary.from_events(s.load_events()).turn_count for s in chain) == TURNS
