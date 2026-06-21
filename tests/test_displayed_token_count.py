"""The displayed "context tokens used" badge in the web UI should reflect the
most recent prompt-snapshot total, not the last LLM-reported `input_tokens`
from a single completed call.

Why this matters: `Session.cumulative_tokens` is updated via REPLACE (not
accumulate) at the end of each LLM call to whatever the provider reported as
that call's `input_tokens`. That has two surprising failure modes:

1. **Stale-high after a one-shot big turn**: a turn that included a large
   one-off attachment can leave `cumulative_tokens` pinned at the big value
   indefinitely, even though the actual prompt that *would* be sent on the
   next turn is much smaller (the attachment isn't part of the persistent
   conversation).
2. **Inconsistent with the prompt inspector**: the inspector reports the
   per-category breakdown from the latest `prompt_snapshot` event, whose
   `total` is the tsugite-side prompt assembly. The badge shows a different
   (and often much larger) number.

Fix: every time the daemon receives a `prompt_snapshot` event with a non-zero
`token_breakdown.total`, push that total into `session.cumulative_tokens`.
The badge and inspector then agree, and the value tracks current prompt
pressure instead of the LLM's report for one historical call.
"""

import pytest
from tsugite_daemon.adapters.http import build_session_event_persister
from tsugite_daemon.session_store import SessionStore


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "session_store.json", context_limits={"agent": 200_000})


@pytest.fixture
def session(store):
    return store.get_or_create_interactive("test-user", "agent")


def test_prompt_snapshot_total_replaces_stale_cumulative_tokens(store, session):
    store.update_token_count(session.id, 200_000)
    assert store.get_session(session.id).cumulative_tokens == 200_000

    persist = build_session_event_persister(store, session.id)
    persist({"type": "prompt_snapshot", "token_breakdown": {"total": 10_000, "categories": []}})

    assert store.get_session(session.id).cumulative_tokens == 10_000


def test_prompt_snapshot_without_total_leaves_cumulative_tokens_unchanged(store, session):
    store.update_token_count(session.id, 50_000)

    persist = build_session_event_persister(store, session.id)
    persist({"type": "prompt_snapshot", "token_breakdown": {}})
    persist({"type": "prompt_snapshot", "token_breakdown": {"total": 0}})
    persist({"type": "prompt_snapshot"})

    assert store.get_session(session.id).cumulative_tokens == 50_000


def test_non_snapshot_events_do_not_touch_cumulative_tokens(store, session):
    store.update_token_count(session.id, 7_500)

    persist = build_session_event_persister(store, session.id)
    persist({"type": "reaction", "token_breakdown": {"total": 99_999}})
    persist({"type": "final_result"})

    assert store.get_session(session.id).cumulative_tokens == 7_500


def test_message_count_not_bumped_by_snapshot_update(store, session):
    # prompt_snapshot fires multiple times per turn; bumping message_count on
    # each would spiral the counter.
    store.update_token_count(session.id, 1_000)
    msg_count_before = store.get_session(session.id).message_count

    persist = build_session_event_persister(store, session.id)
    persist({"type": "prompt_snapshot", "token_breakdown": {"total": 5_000}})
    persist({"type": "prompt_snapshot", "token_breakdown": {"total": 6_000}})

    refreshed = store.get_session(session.id)
    assert refreshed.cumulative_tokens == 6_000
    assert refreshed.message_count == msg_count_before


def test_event_still_persisted_to_jsonl(store, session):
    persist = build_session_event_persister(store, session.id)
    persist({"type": "prompt_snapshot", "token_breakdown": {"total": 4_200}})

    content = store._history_path(session.id).read_text()
    assert '"type":"prompt_snapshot"' in content
    assert '"total":4200' in content
