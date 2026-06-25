"""Tests for the events -> LLM messages reconstruction function."""

from datetime import datetime, timezone

from tsugite.history.models import Event
from tsugite.history.reconstruction import events_to_messages

FIXED_TS = datetime(2026, 5, 4, 10, 30, tzinfo=timezone.utc)
FIXED_TS_PREFIX = "[2026-05-04 10:30 UTC]"


def _ev(type_: str, ts: datetime | None = None, **data) -> Event:
    return Event(type=type_, ts=ts or datetime.now(timezone.utc), data=data)


class TestEventsToMessagesStateless:
    """Provider-agnostic (OpenAI/Anthropic-style) reconstruction."""

    def test_empty_events_returns_empty(self):
        assert events_to_messages([]) == []

    def test_only_session_start_returns_empty(self):
        events = [_ev("session_start", agent="a", model="m")]
        assert events_to_messages(events) == []

    def test_user_input_then_model_response(self):
        events = [
            _ev("session_start", agent="a", model="m"),
            _ev("user_input", ts=FIXED_TS, text="hi"),
            _ev("model_response", raw_content="hello"),
        ]
        msgs = events_to_messages(events)
        assert msgs == [
            {"role": "user", "content": f"{FIXED_TS_PREFIX} hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_assistant_uses_raw_content_not_reparse(self):
        # The raw content may include arbitrary prose plus a code fence; we must
        # send it back verbatim, not re-render from a parsed shape.
        raw = "Thinking out loud.\n\n```python\nprint(1)\n```\n\nDone."
        events = [
            _ev("user_input", ts=FIXED_TS, text="task"),
            _ev("model_response", raw_content=raw),
            _ev("code_execution", output="1\n", duration_ms=5),
        ]
        msgs = events_to_messages(events)
        assert msgs[0] == {"role": "user", "content": f"{FIXED_TS_PREFIX} task"}
        assert msgs[1] == {"role": "assistant", "content": raw}
        # Observation goes back as a user message in XML envelope form
        assert msgs[2]["role"] == "user"
        assert "<tsugite_execution_result" in msgs[2]["content"]
        assert "1\n" in msgs[2]["content"]

    def test_multi_turn_conversation(self):
        events = [
            _ev("user_input", ts=FIXED_TS, text="t1"),
            _ev("model_response", raw_content="r1\n```python\nx=1\n```"),
            _ev("code_execution", output="ok"),
            _ev("model_response", raw_content="r2"),
            _ev("user_input", ts=FIXED_TS, text="t2"),
            _ev("model_response", raw_content="r3"),
        ]
        msgs = events_to_messages(events)
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant", "user", "assistant", "user", "assistant"]
        assert msgs[0]["content"] == f"{FIXED_TS_PREFIX} t1"
        assert msgs[1]["content"].startswith("r1")
        assert "<tsugite_execution_result" in msgs[2]["content"]
        assert msgs[3]["content"] == "r2"
        assert msgs[4]["content"] == f"{FIXED_TS_PREFIX} t2"
        assert msgs[5]["content"] == "r3"

    def test_error_observation_includes_error_tag(self):
        events = [
            _ev("user_input", text="t"),
            _ev("model_response", raw_content="```python\nbad\n```"),
            _ev("code_execution", code="bad", output="", error="NameError: bad"),
        ]
        msgs = events_to_messages(events)
        assert "<error>" in msgs[2]["content"]
        assert "NameError" in msgs[2]["content"]

    def test_compaction_event_replaces_prior_history(self):
        events = [
            _ev("user_input", text="old"),
            _ev("model_response", raw_content="old reply"),
            _ev("compaction", summary="we talked about cats", retained_count=0),
            _ev("user_input", ts=FIXED_TS, text="new"),
            _ev("model_response", raw_content="new reply"),
        ]
        msgs = events_to_messages(events)
        # Pre-compaction events drop out; a synthetic summary user/assistant pair
        # appears, then post-compaction events.
        assert len(msgs) == 4
        assert msgs[0]["role"] == "user"
        assert "we talked about cats" in msgs[0]["content"]
        assert msgs[1]["role"] == "assistant"
        assert msgs[2] == {"role": "user", "content": f"{FIXED_TS_PREFIX} new"}
        assert msgs[3] == {"role": "assistant", "content": "new reply"}

    def test_format_error_observation(self):
        events = [
            _ev("user_input", text="t"),
            _ev("model_response", raw_content="```python\na\n```\n```python\nb\n```"),
            _ev("format_error", reason="2 python blocks", rejected_content="..."),
        ]
        msgs = events_to_messages(events)
        # Format error becomes a user-side observation back to the model
        assert msgs[-1]["role"] == "user"
        assert "Format Error" in msgs[-1]["content"] or "format_error" in msgs[-1]["content"]

    def test_unknown_event_types_ignored(self):
        events = [
            _ev("user_input", ts=FIXED_TS, text="t"),
            _ev("custom_telemetry", whatever=1),
            _ev("model_response", raw_content="r"),
        ]
        msgs = events_to_messages(events)
        assert msgs == [
            {"role": "user", "content": f"{FIXED_TS_PREFIX} t"},
            {"role": "assistant", "content": "r"},
        ]


class TestTimestampPrefixing:
    """Each user_input gets a `[ts tz]` prefix so the agent can anchor recalled
    facts in time. Code execution envelopes get a `ts="..."` attribute. Model
    responses are passed through verbatim so the model sees its own past
    output unchanged.
    """

    def test_user_input_prefixed_with_absolute_ts(self):
        events = [_ev("user_input", ts=FIXED_TS, text="hello")]
        msgs = events_to_messages(events)
        assert msgs[0]["content"] == f"{FIXED_TS_PREFIX} hello"

    def test_model_response_is_unchanged(self):
        # Modifying assistant content would confuse the model about its own
        # past output and break prompt cache hits.
        events = [
            _ev("user_input", ts=FIXED_TS, text="hi"),
            _ev("model_response", ts=FIXED_TS, raw_content="raw assistant text"),
        ]
        msgs = events_to_messages(events)
        assert msgs[1] == {"role": "assistant", "content": "raw assistant text"}

    def test_code_execution_envelope_has_ts_attribute(self):
        events = [
            _ev("user_input", text="t"),
            _ev("model_response", raw_content="```python\nx=1\n```"),
            _ev("code_execution", ts=FIXED_TS, output="ok", duration_ms=5),
        ]
        msgs = events_to_messages(events)
        content = msgs[2]["content"]
        assert 'ts="2026-05-04 10:30 UTC"' in content
        assert 'status="success"' in content
        assert 'duration_ms="5"' in content

    def test_format_error_unchanged(self):
        # Format errors are corrective/ephemeral - no timestamp signal needed.
        events = [
            _ev("user_input", text="t"),
            _ev("model_response", raw_content="bad"),
            _ev("format_error", ts=FIXED_TS, reason="2 blocks", rejected_content="..."),
        ]
        msgs = events_to_messages(events)
        assert "ts=" not in msgs[2]["content"]

    def test_replay_is_byte_stable(self):
        # The same events rendered twice must produce identical bytes so the
        # prompt cache stays warm across turns.
        events = [
            _ev("user_input", ts=FIXED_TS, text="t1"),
            _ev("model_response", raw_content="r1"),
            _ev("code_execution", ts=FIXED_TS, output="ok"),
            _ev("user_input", ts=FIXED_TS, text="t2"),
        ]
        first = events_to_messages(events)
        second = events_to_messages(events)
        assert first == second
