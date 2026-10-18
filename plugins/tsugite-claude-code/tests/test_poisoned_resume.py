"""Recovery from an unresumable Claude Code resume transcript.

A resumed sidecar transcript containing an empty text content block makes the
Anthropic API reject every request with `400 ... text content blocks must be
non-empty`. A deleted transcript makes the CLI fail with `No conversation found
with session ID`. Both wedge the conversation, since the resume UUID is
re-selected on every send. The provider falls back to a fresh session seeded
from tsugite's serialized history, and never sends an empty message that could
poison a transcript.
"""

from unittest.mock import patch

import pytest
from tsugite_claude_code.provider import ClaudeCodeProvider

from tsugite.exceptions import AgentExecutionError

POISON_ERROR = "API Error: 400 messages: text content blocks must be non-empty"


class FakeProcess:
    """Stands in for ClaudeCodeProcess; behavior keyed off resume_session."""

    instances: list = []
    session_id = None
    compacted = False

    def __init__(self):
        self.start_kwargs = None
        self.sent = []
        self.stopped = False
        FakeProcess.instances.append(self)

    async def start(self, **kwargs):
        self.start_kwargs = kwargs

    async def stop(self):
        self.stopped = True

    def send_message(self, content):
        self.sent.append(content)
        if self.start_kwargs.get("resume_session"):
            return self._events_error()
        return self._events_ok()

    async def _events_error(self):
        yield {"type": "result", "text": POISON_ERROR, "is_error": True, "subtype": "success"}

    async def _events_ok(self):
        yield {"type": "text_delta", "text": "recovered"}
        yield {
            "type": "result",
            "text": "recovered",
            "is_error": False,
            "subtype": "success",
            "session_id": "new-session",
            "cost_usd": 0.01,
            "input_tokens": 10,
            "output_tokens": 5,
        }


class AlwaysError(FakeProcess):
    def send_message(self, content):
        self.sent.append(content)
        return self._events_error()


@pytest.fixture(autouse=True)
def _fake_process():
    FakeProcess.instances = []
    with patch("tsugite_claude_code.process.ClaudeCodeProcess", FakeProcess):
        yield


def _resumed_provider():
    provider = ClaudeCodeProvider()
    provider.set_context(
        resume_session="9e6bd114-poisoned",
        previous_messages=[
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": ""},
        ],
    )
    return provider


@pytest.mark.asyncio
async def test_poisoned_resume_falls_back_to_fresh_session():
    provider = _resumed_provider()

    response = await provider.acompletion(
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "new prompt"}],
        model="opus",
    )

    assert response.content == "recovered"
    assert len(FakeProcess.instances) == 2
    poisoned, fresh = FakeProcess.instances
    assert poisoned.start_kwargs["resume_session"] == "9e6bd114-poisoned"
    assert poisoned.stopped
    assert fresh.start_kwargs["resume_session"] is None
    # The fresh session is seeded from serialized history; the serializer
    # renders "Role: content" lines so the empty assistant turn is non-empty.
    first_message = fresh.sent[0]
    assert "<conversation_history>" in first_message
    assert "Assistant:" in first_message
    assert "new prompt" in first_message


@pytest.mark.asyncio
async def test_poisoned_resume_fallback_streaming():
    provider = _resumed_provider()

    stream = await provider.acompletion(
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "new prompt"}],
        model="opus",
        stream=True,
    )
    content = ""
    async for chunk in stream:
        content += chunk.content

    assert content == "recovered"
    assert len(FakeProcess.instances) == 2
    assert FakeProcess.instances[1].start_kwargs["resume_session"] is None


@pytest.mark.asyncio
async def test_400_without_resume_still_raises():
    """A malformed-history 400 on a fresh session has no resume to sever — no retry loop."""

    with patch("tsugite_claude_code.process.ClaudeCodeProcess", AlwaysError):
        provider = ClaudeCodeProvider()
        with pytest.raises(AgentExecutionError):
            await provider.acompletion(messages=[{"role": "user", "content": "hi"}], model="opus")
    assert len(FakeProcess.instances) == 1


@pytest.mark.asyncio
async def test_400_on_later_turn_still_raises():
    """The fallback only covers the resume replay (first send); later turns surface errors."""
    provider = ClaudeCodeProvider()

    await provider.acompletion(messages=[{"role": "user", "content": "hi"}], model="opus")

    poison = FakeProcess.instances[0]
    poison.start_kwargs["resume_session"] = None

    def _err(content):
        poison.sent.append(content)
        return poison._events_error()

    poison.send_message = _err
    with pytest.raises(AgentExecutionError):
        await provider.acompletion(messages=[{"role": "user", "content": "again"}], model="opus")
    assert len(FakeProcess.instances) == 1


@pytest.mark.asyncio
async def test_non_400_resume_error_still_raises():
    """Prompt-too-long and friends must keep surfacing (the daemon retry path owns them)."""

    class TooLong(FakeProcess):
        async def _events_error(self):
            yield {"type": "result", "text": "Prompt is too long", "is_error": True, "subtype": "error"}

    with patch("tsugite_claude_code.process.ClaudeCodeProcess", TooLong):
        provider = _resumed_provider()
        with pytest.raises(AgentExecutionError, match="[Pp]rompt is too long"):
            await provider.acompletion(messages=[{"role": "user", "content": "hi"}], model="opus")
    assert len(FakeProcess.instances) == 1


@pytest.mark.asyncio
async def test_empty_observation_not_sent_verbatim():
    """Prevention: an empty message must never reach the CLI (it would persist an
    empty text block in the sidecar transcript and poison the next resume)."""
    provider = ClaudeCodeProvider()

    await provider.acompletion(messages=[{"role": "user", "content": "hi"}], model="opus")
    proc = FakeProcess.instances[0]
    proc.start_kwargs["resume_session"] = None

    await provider.acompletion(messages=[{"role": "user", "content": ""}], model="opus")

    assert proc.sent[-1].strip(), "empty user content was sent verbatim to the CLI"


@pytest.mark.asyncio
async def test_fresh_session_costs_start_from_zero_again():
    """A fresh CLI session counts its own cost from 0, so the resumed session's
    baseline does not apply to it."""
    provider = ClaudeCodeProvider()
    provider.set_context(
        resume_session="9e6bd114-poisoned",
        resume_cost_baseline=5.0,
        previous_messages=[{"role": "user", "content": "earlier question"}],
    )

    response = await provider.acompletion(
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "new prompt"}],
        model="opus",
    )

    assert response.cost == pytest.approx(0.01)


MISSING_TRANSCRIPT = "No conversation found with session ID: c3dc320d-ee27-45bc-b245-89d85ef9cbec"


class MissingTranscriptProcess(FakeProcess):
    """Result event as the real CLI emits it for a deleted transcript: empty text, message in `errors`."""

    async def _events_error(self):
        yield {
            "type": "result",
            "text": "",
            "errors": [MISSING_TRANSCRIPT],
            "is_error": True,
            "subtype": "error_during_execution",
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_missing_transcript_falls_back_to_fresh_session(stream):
    provider = _resumed_provider()

    with patch("tsugite_claude_code.process.ClaudeCodeProcess", MissingTranscriptProcess):
        result = await provider.acompletion(
            messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "new prompt"}],
            model="opus",
            stream=stream,
        )
        content = "".join([chunk.content async for chunk in result]) if stream else result.content

    assert content == "recovered"
    dead, fresh = FakeProcess.instances
    assert dead.stopped
    assert fresh.start_kwargs["resume_session"] is None
    assert "<conversation_history>" in fresh.sent[0]
    # The next turn records the new session id, not the dead one.
    assert provider.get_state()["session_id"] == "new-session"


@pytest.mark.asyncio
async def test_error_result_surfaces_cli_errors_when_text_is_empty():
    class AlwaysMissingError(AlwaysError, MissingTranscriptProcess):
        pass

    provider = ClaudeCodeProvider()
    with (
        patch("tsugite_claude_code.process.ClaudeCodeProcess", AlwaysMissingError),
        pytest.raises(AgentExecutionError, match="No conversation found with session ID") as exc,
    ):
        await provider.acompletion(messages=[{"role": "user", "content": "hi"}], model="opus")
    assert "subtype=error_during_execution" in str(exc.value)
