"""The raw-messages endpoint's event source.

The endpoint rebuilds per-turn raw messages on demand (no durable wire payload).
The reconstruction itself is core (`tsugite.history.reconstruct_raw_turns`, tested
in `tests/test_reconstruct_raw_turns.py`); these tests cover the daemon's event
loader, which must feed that core function the SAME ``Event`` objects resume reads
(via ``get_history_backend``, not ``session_store``'s dict events).
"""

from __future__ import annotations

from pathlib import Path

from tsugite_daemon.adapters.http.agents import _load_session_events

from tests.history_helpers import seed_history_session
from tsugite.history import reconstruct_raw_turns
from tsugite.history.models import Event


def test_load_session_events_yields_event_objects_for_reconstruction(history_dir: Path):
    """The endpoint's event source must yield Event objects (not dicts) so the
    reconstruction matches what resume rebuilds."""
    storage = seed_history_session("sess", agent="t", model="m")
    storage.record("user_input", text="hi")
    storage.record("model_request", turn=0, provider="anthropic", model="claude-x")
    storage.record("model_response", turn=0, provider="anthropic", model="claude-x", raw_content="done")

    events = _load_session_events("sess")
    assert all(isinstance(e, Event) for e in events)

    turns = reconstruct_raw_turns(events)
    assert len(turns) == 1
    assert turns[0]["turn"] == 0
    assert turns[0]["response"] == {"raw_content": "done"}
    assert turns[0]["request"][-1]["content"].endswith("hi")


def test_missing_session_loads_no_events(history_dir: Path):
    assert _load_session_events("nope") == []
