"""Tests covering the final_answer delivery path through SSEProgressHandler.

H1 hypothesis: final_result SSE events are not persisted to the session event
log, so if the SSE connection drops between emission and client receipt (daemon
restart, browser backgrounding, proxy timeout), the final answer is silently
lost with no way for the UI to recover it.
"""

from unittest.mock import MagicMock

from tsugite.daemon.adapters.http import SSEProgressHandler
from tsugite.events import ErrorEvent, FinalAnswerEvent, InfoEvent, ReactionEvent


class TestFinalResultPersistence:
    """Unit tests for SSEProgressHandler persistence allowlist."""

    def test_final_result_is_persisted(self):
        """A final_result emission MUST be written to the event log so the UI
        can recover it after an SSE disconnect."""
        handler = SSEProgressHandler()
        persister = MagicMock()
        handler.set_event_persister(persister)

        handler.handle_event(FinalAnswerEvent(answer="hello world", turns=1))

        assert persister.called, (
            "final_result was emitted but _persist_event was not called — "
            "if the SSE stream drops, the final answer is lost with no recovery path"
        )
        payload = persister.call_args[0][0]
        assert payload["type"] == "final_result"
        assert payload["result"] == "hello world"

    def test_error_is_persisted(self):
        """Errors must also be persisted so the UI can show why a run failed
        after a disconnect."""
        handler = SSEProgressHandler()
        persister = MagicMock()
        handler.set_event_persister(persister)

        handler.handle_event(ErrorEvent(error="boom", step=1))

        assert persister.called, (
            "error was emitted but _persist_event was not called — UI has no way to show the failure after a reconnect"
        )
        payload = persister.call_args[0][0]
        assert payload["type"] == "error"
        assert payload["error"] == "boom"

    def test_reaction_is_still_persisted(self):
        """Regression guard: reaction persistence (already working) must stay
        working after the allowlist is widened."""
        handler = SSEProgressHandler()
        persister = MagicMock()
        handler.set_event_persister(persister)

        handler.handle_event(ReactionEvent(emoji=":+1:"))

        assert persister.called
        payload = persister.call_args[0][0]
        assert payload["type"] == "reaction"

    def test_transient_events_are_not_persisted(self):
        """Progress-only events (thought, code, tool_result, info) are high-volume
        and should stay out of the event log."""
        handler = SSEProgressHandler()
        persister = MagicMock()
        handler.set_event_persister(persister)

        handler.handle_event(InfoEvent(message="working on it"))

        assert not persister.called, (
            "transient info events must not be persisted — they would bloat the event log with progress chatter"
        )
