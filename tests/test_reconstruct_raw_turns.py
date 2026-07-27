"""Core reconstruction of per-turn raw messages from a session event log.

``reconstruct_raw_turns`` rebuilds, per turn, the request messages the model saw
(the log prefix up to that turn's ``model_request``) and its verbatim response.
It is a pure function over ``Event`` objects, so these tests need no storage or
HTTP - the daemon endpoint is a thin wrapper over this.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tsugite.history import reconstruct_raw_turns
from tsugite.history.models import Event
from tsugite.history.reconstruction import events_to_messages

BASE_TS = datetime(2026, 5, 4, 10, 30, tzinfo=timezone.utc)


def _ev(type_: str, offset: int, **data) -> Event:
    # Distinct, ordered timestamps so reconstruction's per-message ts prefixes
    # stay stable and the log order is unambiguous.
    return Event(type=type_, ts=BASE_TS + timedelta(seconds=offset), data=data)


def _two_turn_log() -> list[Event]:
    """user_input, model_request(0), model_response(0), code_execution,
    user_input, model_request(1), model_response(1)."""
    return [
        _ev("user_input", 0, text="hello"),
        _ev("model_request", 1, turn=0, provider="anthropic", model="claude-x", message_count=1),
        _ev(
            "model_response",
            2,
            turn=0,
            provider="anthropic",
            model="claude-x",
            raw_content="r0\n```python-exec\nx=1\n```",
        ),
        _ev("code_execution", 3, code="x=1", output="ok"),
        _ev("user_input", 4, text="again"),
        _ev("model_request", 5, turn=1, provider="anthropic", model="claude-x", message_count=4),
        _ev("model_response", 6, turn=1, provider="anthropic", model="claude-x", raw_content="r1"),
    ]


def test_one_turn_entry_per_model_request_with_carried_fields():
    turns = reconstruct_raw_turns(_two_turn_log())
    assert [t["turn"] for t in turns] == [0, 1]
    for t in turns:
        assert t["provider"] == "anthropic"
        assert t["model"] == "claude-x"


def test_turn0_request_is_prefix_up_to_its_model_request():
    events = _two_turn_log()
    turns = reconstruct_raw_turns(events)
    # Prefix ends at the turn-0 model_request (index 1): just the turn-0 user
    # message, not its assistant response.
    assert turns[0]["request"] == events_to_messages(events[:1])
    assert turns[0]["request"][-1]["role"] == "user"
    assert all(m["role"] != "assistant" for m in turns[0]["request"])


def test_turn1_request_includes_prior_turn_and_ends_on_new_user_message():
    events = _two_turn_log()
    turns = reconstruct_raw_turns(events)
    # Prefix ends at the turn-1 model_request (index 5): turn-0's user +
    # assistant + execution, then the turn-1 user message last.
    assert turns[1]["request"] == events_to_messages(events[:5])
    roles = [m["role"] for m in turns[1]["request"]]
    assert roles == ["user", "assistant", "user", "user"]
    assert any("<tsugite_execution_result" in str(m["content"]) for m in turns[1]["request"])
    assert turns[1]["request"][-1]["content"].endswith("again")


def test_response_carries_that_turns_verbatim_raw_content():
    turns = reconstruct_raw_turns(_two_turn_log())
    # Verbatim: the response shows the model's own output unpromoted, unlike the
    # replayed assistant message in the next turn's request.
    assert turns[0]["response"] == {"raw_content": "r0\n```python-exec\nx=1\n```"}
    assert turns[1]["response"] == {"raw_content": "r1"}


def test_model_request_without_a_response_yields_null_response():
    events = [
        _ev("user_input", 0, text="hello"),
        _ev("model_request", 1, turn=0, provider="anthropic", model="claude-x"),
    ]
    turns = reconstruct_raw_turns(events)
    assert len(turns) == 1
    assert turns[0]["response"] is None


def test_no_model_requests_yields_no_turns():
    events = [_ev("user_input", 0, text="hello")]
    assert reconstruct_raw_turns(events) == []


def _multistep_log() -> list[Event]:
    """A realistic session: message "first" runs two steps (the agent writes code,
    then answers), message "second" runs one. ``turn`` is a per-run step counter,
    so it reads 0, 1, then 0 again - the second message's first step reuses turn 0.
    """
    return [
        _ev("user_input", 0, text="first"),
        _ev("model_request", 1, turn=0, provider="anthropic", model="claude-x"),
        _ev("model_response", 2, turn=0, raw_content="r0\n```python-exec\nx=1\n```"),
        _ev("code_execution", 3, code="x=1", output="ok"),
        _ev("model_request", 4, turn=1, provider="anthropic", model="claude-x"),
        _ev("model_response", 5, turn=1, raw_content="answer to first"),
        _ev("user_input", 6, text="second"),
        _ev("model_request", 7, turn=0, provider="anthropic", model="claude-x"),
        _ev("model_response", 8, turn=0, raw_content="answer to second"),
    ]


def test_index_is_monotonic_and_turn_is_the_repeating_step_counter():
    turns = reconstruct_raw_turns(_multistep_log())
    assert [t["index"] for t in turns] == [1, 2, 3]
    # turn repeats (0, 1, 0) because it resets every user message; index does not.
    assert [t["turn"] for t in turns] == [0, 1, 0]


def test_response_pairs_by_log_position_not_the_colliding_turn():
    # The regression this guards: pairing by ``turn`` collides on the two turn-0
    # calls, so call 1 would wrongly show call 3's response. Positional pairing
    # gives each call its own following model_response.
    turns = reconstruct_raw_turns(_multistep_log())
    assert [t["response"]["raw_content"] for t in turns] == [
        "r0\n```python-exec\nx=1\n```",
        "answer to first",
        "answer to second",
    ]


def test_new_messages_is_the_delta_the_call_added():
    events = _multistep_log()
    turns = reconstruct_raw_turns(events)
    # First call: nothing prior, so the delta is the whole (tiny) prompt, and it
    # is not a reset (no earlier context was dropped).
    assert turns[0]["reset_before"] is False
    assert turns[0]["new_messages"] == turns[0]["request"]
    assert [m["role"] for m in turns[0]["new_messages"]] == ["user"]
    # Second call adds only the assistant's code turn and its execution result.
    assert turns[1]["reset_before"] is False
    assert [m["role"] for m in turns[1]["new_messages"]] == ["assistant", "user"]
    assert "<tsugite_execution_result" in str(turns[1]["new_messages"][1]["content"])
    # Third call adds the answer to "first" and the new "second" user message.
    assert [m["role"] for m in turns[2]["new_messages"]] == ["assistant", "user"]
    assert turns[2]["new_messages"][-1]["content"].endswith("second")
    # The full request stays the whole prefix; the delta is a suffix of it.
    for t in turns:
        assert t["new_messages"] == t["request"][len(t["request"]) - len(t["new_messages"]) :]


def test_compaction_resets_the_delta_and_flags_it():
    events = [
        _ev("user_input", 0, text="first"),
        _ev("model_request", 1, turn=0, provider="anthropic", model="claude-x"),
        _ev("model_response", 2, turn=0, raw_content="r0"),
        _ev("compaction", 3, summary="EARLIER SUMMARY"),
        _ev("user_input", 4, text="second"),
        _ev("model_request", 5, turn=0, provider="anthropic", model="claude-x"),
        _ev("model_response", 6, turn=0, raw_content="r1"),
    ]
    turns = reconstruct_raw_turns(events)
    # The post-compaction call's prompt resets to the summary and shares no prefix
    # with the pre-compaction call, so it is flagged and the whole prompt is "new".
    assert turns[0]["reset_before"] is False
    assert turns[1]["reset_before"] is True
    assert turns[1]["new_messages"] == turns[1]["request"]
    assert "EARLIER SUMMARY" in str(turns[1]["request"][0]["content"])
