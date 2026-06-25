"""event_to_ui_dict maps a stored Event to the flat dict the daemon/UI consume.

The daemon reads top-level keys (type, timestamp, and data fields like name/turn),
not Event's nested {type, ts, data}. This pins the mapping against the real consumer
(_progress_status_text / _apply_event_to_progress) so storage can round-trip while the
UI keeps working.
"""

from datetime import datetime, timezone

from tsugite.history.models import Event
from tsugite.history.ui_events import event_to_ui_dict

TS = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_flattens_data_renames_ts_includes_id():
    e = Event(id=7, type="tool_invocation", ts=TS, data={"name": "grep", "duration_ms": 5})
    d = event_to_ui_dict(e)
    assert d["type"] == "tool_invocation"
    assert d["name"] == "grep"  # data flattened to top level
    assert d["duration_ms"] == 5
    assert d["timestamp"].startswith("2026-01-01T12:00:00")
    assert d["id"] == 7
    assert "ts" not in d and "data" not in d


def test_authoritative_keys_win_over_data_collision():
    e = Event(type="real", ts=TS, data={"type": "evil", "timestamp": "evil", "id": "evil"})
    d = event_to_ui_dict(e)
    assert d["type"] == "real"
    assert d["timestamp"].startswith("2026-01-01")


def test_daemon_progress_consumer_reads_the_dict():
    from tsugite_daemon.session_store import _apply_event_to_progress, _progress_status_text

    assert _progress_status_text(event_to_ui_dict(Event(type="model_request", ts=TS, data={"turn": 2}))) == (
        "Waiting on LLM..."
    )
    assert _progress_status_text(event_to_ui_dict(Event(type="tool_invocation", ts=TS, data={"name": "grep"}))) == (
        "Tool: grep"
    )
    # The progress fold reads timestamp + counts a real tool event without error.
    progress = {"turn_count": 0, "tool_count": 0, "status_text": "", "last_event_time": None}
    _apply_event_to_progress(progress, event_to_ui_dict(Event(type="tool_invocation", ts=TS, data={"name": "grep"})))
    assert progress["tool_count"] == 1
    assert progress["last_event_time"].startswith("2026-01-01")
