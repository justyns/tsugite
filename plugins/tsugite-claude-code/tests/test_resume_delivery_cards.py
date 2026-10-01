"""Delivery cards recorded between turns must reach a resumed claude_code session."""

from datetime import datetime, timezone

from tsugite_claude_code.provider import ClaudeCodeProvider

from tsugite.history.models import Event
from tsugite.history.reconstruction import events_to_messages

_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _delivery(kind: str, message: str) -> Event:
    return Event(
        type="delivery",
        ts=_TS,
        data={"source": "schedule", "kind": kind, "message": message, "delivery_id": "dlv-1"},
    )


def _user(text: str) -> Event:
    return Event(type="user_input", ts=_TS, data={"text": text})


def _reply(text: str) -> Event:
    return Event(type="model_response", ts=_TS, data={"raw_content": text})


def _first_message(history: list[dict], resume: str | None = "sess-1") -> str:
    provider = ClaudeCodeProvider()
    provider.set_context(resume_session=resume, previous_messages=history)
    return provider._build_first_message([{"role": "user", "content": "next task"}])


def test_resumed_session_includes_fyi_card_since_last_reply():
    history = events_to_messages(
        [_user("hi"), _reply("hello"), _delivery("fyi", "nightly backup done"), _user("next task")]
    )
    text = _first_message(history)
    assert "nightly backup done" in text
    assert text.endswith("next task")


def test_resumed_session_skips_card_the_model_already_answered():
    history = events_to_messages([_user("hi"), _delivery("fyi", "old card"), _reply("saw it"), _user("next task")])
    assert "old card" not in _first_message(history)


def test_resumed_session_leaves_needs_ack_card_to_pending_deliveries():
    history = events_to_messages(
        [_user("hi"), _reply("hello"), _delivery("needs_ack", "approve me"), _user("next task")]
    )
    assert "approve me" not in _first_message(history)


def test_fresh_session_does_not_duplicate_cards_beside_history():
    history = events_to_messages([_user("hi"), _reply("hello"), _delivery("fyi", "card"), _user("next task")])
    text = _first_message(history, resume=None)
    assert text.count("<tsugite_delivery") == 1
