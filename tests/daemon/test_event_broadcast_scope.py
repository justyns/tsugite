"""Narrow the cross-session broadcaster to events other surfaces actually need.

The sidebar progress cache reads `turn_start`, `thought`, `reasoning_content`,
`tool_result`, `hook_status`, and session-end events (see conversations.js
_updateProgressCache and event_types.js). Broadcasting every event (including
chunk-level streaming output and prompt snapshots) is pure noise for that
purpose, and — combined with the existing per-chat SSE stream — causes the
active client to render progress twice.
"""

from unittest.mock import MagicMock

from tsugite_daemon.adapters.http import SSEProgressHandler


def _broadcast_event_types(handler: SSEProgressHandler, event_types: list[str]) -> list[dict]:
    broadcaster = MagicMock()
    handler.set_broadcaster(broadcaster)
    handler.set_session_id("sess-1")
    for et in event_types:
        handler._emit(et, {})
    return [call.args[1] for call in broadcaster.emit.call_args_list]


def test_noisy_events_not_broadcast():
    handler = SSEProgressHandler()
    noisy = ["stream_chunk", "stream_complete", "prompt_snapshot"]
    assert _broadcast_event_types(handler, noisy) == []


def test_turn_end_events_not_broadcast():
    """final_result/error/cancelled are delivered to the active tab via the
    per-chat streaming SSE response. Broadcasting them on the cross-session bus
    races with the streaming reader's clear of the session's `sending` flag in
    the web UI and produces duplicate bubbles. history_update + loadHistory
    keep other tabs in sync."""
    handler = SSEProgressHandler()
    turn_end = ["final_result", "error", "cancelled"]
    assert _broadcast_event_types(handler, turn_end) == []


def test_sidebar_relevant_events_still_broadcast():
    handler = SSEProgressHandler()
    needed = [
        "turn_start",
        "thought",
        "reasoning_content",
        "tool_result",
        "hook_status",
        "session_complete",
        "session_error",
        "session_cancelled",
    ]
    forwarded = _broadcast_event_types(handler, needed)
    assert len(forwarded) == len(needed)
    assert all(p["event_type"] in needed for p in forwarded)


def test_no_broadcast_without_session_id():
    handler = SSEProgressHandler()
    broadcaster = MagicMock()
    handler.set_broadcaster(broadcaster)
    handler._emit("turn_start", {})
    broadcaster.emit.assert_not_called()
