"""Provider behaviour against a fake cursor_sdk module."""

from __future__ import annotations

import types
from dataclasses import dataclass, field

import pytest
import tsugite_cursor.provider as provider_module
from tsugite_cursor import create_provider

from tsugite.exceptions import AgentExecutionError

pytestmark = pytest.mark.asyncio

MODEL = "composer-2.5"
SYSTEM = {"role": "system", "content": "You are the tsugite test agent."}
TASK = {"role": "user", "content": "Say ok"}


class CursorSDKError(Exception):
    pass


class AuthenticationError(CursorSDKError):
    pass


class AgentNotFoundError(CursorSDKError):
    pass


@dataclass(frozen=True)
class TextDeltaUpdate:
    text: str


@dataclass(frozen=True)
class ThinkingDeltaUpdate:
    text: str


@dataclass(frozen=True)
class ToolCallStartedUpdate:
    tool_call: dict


class PartialToolCallUpdate(ToolCallStartedUpdate):
    pass


@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    reasoning_tokens: int | None = None


@dataclass
class Turn:
    updates: list
    status: str = "finished"
    error_message: str | None = None
    usage: TokenUsage = field(default_factory=lambda: TokenUsage(1200, 40, 500))
    error: Exception | None = None
    cancel_error: Exception | None = None


class FakeRun:
    def __init__(self, turn: Turn, sdk, streams: bool):
        self.turn, self.sdk, self.streams, self.status = turn, sdk, streams, "running"

    async def events(self):
        for update in self.turn.updates if self.streams else ():
            yield types.SimpleNamespace(interaction_update=update)
        if self.turn.error:
            raise self.turn.error

    async def wait(self):
        self.status = self.turn.status
        model = types.SimpleNamespace(id=MODEL)
        return types.SimpleNamespace(
            status=self.turn.status, error_message=self.turn.error_message, usage=self.turn.usage, model=model
        )

    async def cancel(self):
        self.sdk.calls.append(("cancel",))
        if self.turn.cancel_error:
            raise self.turn.cancel_error


class FakeAgent:
    def __init__(self, sdk, agent_id):
        self.sdk, self.agent_id = sdk, agent_id

    async def send(self, text, options=None):
        self.sdk.calls.append(("send", text))
        return FakeRun(self.sdk.turns.pop(0), self.sdk, streams=getattr(options, "on_delta", None) is not None)


class FakeClient:
    def __init__(self, sdk):
        self.sdk = sdk

    async def aclose(self):
        self.sdk.calls.append(("close",))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()


CATALOG = [
    types.SimpleNamespace(
        id=MODEL,
        variants=[
            types.SimpleNamespace(is_default=True, params=[types.SimpleNamespace(id="fast", value="false")]),
        ],
    )
]


def make_sdk(turns, missing_agent=False, create_error=None):
    sdk = types.SimpleNamespace(
        calls=[],
        turns=list(turns),
        CursorSDKError=CursorSDKError,
        AuthenticationError=AuthenticationError,
        AgentNotFoundError=AgentNotFoundError,
        TextDeltaUpdate=TextDeltaUpdate,
        ThinkingDeltaUpdate=ThinkingDeltaUpdate,
        ToolCallStartedUpdate=ToolCallStartedUpdate,
        PartialToolCallUpdate=PartialToolCallUpdate,
        SendOptions=lambda on_delta: types.SimpleNamespace(on_delta=on_delta),
    )

    async def launch_bridge(workspace):
        sdk.calls.append(("launch", workspace))
        return FakeClient(sdk)

    async def create(options, client):
        sdk.calls.append(("create", options))
        if create_error:
            raise create_error
        return FakeAgent(sdk, "agent-new")

    async def resume(agent_id, options, client):
        sdk.calls.append(("resume", agent_id, options))
        if missing_agent:
            raise AgentNotFoundError(f"{agent_id} not found")
        return FakeAgent(sdk, agent_id)

    async def list_models(client, api_key):
        return CATALOG

    sdk.AsyncCursorClient = types.SimpleNamespace(launch_bridge=launch_bridge)
    sdk.AsyncAgent = types.SimpleNamespace(create=create, resume=resume)
    sdk.AsyncCursor = types.SimpleNamespace(models=types.SimpleNamespace(list=list_models))
    return sdk


@pytest.fixture
def cursor(monkeypatch, tmp_path):
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    monkeypatch.setattr(provider_module, "session_cwd", lambda: tmp_path)

    def install(*turns, **kwargs):
        sdk = make_sdk(turns, **kwargs)
        monkeypatch.setattr(provider_module, "_sdk", lambda: sdk)
        return sdk

    return install


def ok(text="ok"):
    return Turn([ThinkingDeltaUpdate("Saying ok."), TextDeltaUpdate(text)])


def calls(sdk, name):
    return [c for c in sdk.calls if c[0] == name]


async def test_new_session_runs_as_a_plain_model(cursor, tmp_path):
    sdk = cursor(ok())
    provider = create_provider()
    response = await provider.acompletion([SYSTEM, TASK], MODEL)

    ((_, options),) = calls(sdk, "create")
    assert options == {
        "api_key": "test-key",
        "name": "tsugite",
        "model": {"id": MODEL, "params": [{"id": "fast", "value": "false"}]},
        "tools": [],
        "local": {"cwd": str(tmp_path), "setting_sources": []},
    }
    assert calls(sdk, "send") == [
        ("send", f"<tsugite_system_prompt>\n{SYSTEM['content']}\n</tsugite_system_prompt>\n\nSay ok")
    ]
    assert (response.content, response.reasoning_content) == ("ok", "Saying ok.")
    usage = response.usage
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens, usage.cached_tokens) == (
        1200,
        40,
        1240,
        500,
    )
    assert provider.get_state() == {"session_id": "agent-new", "context_window": 200_000}


@pytest.mark.parametrize(
    "model, expected",
    [
        ("auto", {"id": "default"}),
        ("grok-4.7[effort=high]", {"id": "grok-4.7", "params": [{"id": "effort", "value": "high"}]}),
    ],
)
async def test_model_selection(cursor, model, expected):
    sdk = cursor(ok())
    await create_provider().acompletion([TASK], model)
    assert calls(sdk, "create")[0][1]["model"] == expected


async def test_later_turns_send_only_the_new_message_to_the_same_agent(cursor):
    sdk = cursor(ok("first"), ok("second"))
    provider = create_provider()
    await provider.acompletion([SYSTEM, TASK], MODEL)
    second = await provider.acompletion([SYSTEM, TASK, {"role": "user", "content": "observation"}], MODEL)

    assert second.content == "second"
    assert len(calls(sdk, "create")) == 1
    assert calls(sdk, "send")[1] == ("send", "observation")


async def test_stream_yields_reasoning_then_text_then_usage(cursor):
    cursor(ok())
    chunks = [c async for c in await create_provider().acompletion([TASK], MODEL, stream=True)]
    assert chunks[0].reasoning_content == "Saying ok."
    assert chunks[1].content == "ok"
    assert chunks[-1].done and chunks[-1].usage.prompt_tokens == 1200


async def test_resume_reopens_the_agent_without_the_preface(cursor):
    sdk = cursor(ok())
    provider = create_provider()
    provider.set_context(resume_session="agent-old", previous_messages=[{"role": "user", "content": "earlier"}])
    await provider.acompletion([SYSTEM, TASK], MODEL)

    assert calls(sdk, "resume")[0][1] == "agent-old"
    assert calls(sdk, "create") == []
    assert calls(sdk, "send") == [("send", "Say ok")]
    assert provider.get_state()["session_id"] == "agent-old"


async def test_missing_agent_on_resume_starts_a_fresh_one_with_history(cursor):
    sdk = cursor(ok(), missing_agent=True)
    provider = create_provider()
    provider.set_context(resume_session="agent-gone", previous_messages=[{"role": "user", "content": "earlier"}])
    await provider.acompletion([TASK], MODEL)

    assert calls(sdk, "create")
    assert "<conversation_history>" in calls(sdk, "send")[0][1]
    assert provider.get_state()["session_id"] == "agent-new"


async def test_a_tool_call_fails_the_run_and_cancels_it(cursor):
    sdk = cursor(Turn([ToolCallStartedUpdate({"type": "read"})]))
    with pytest.raises(AgentExecutionError, match=r"started a tool call \(read\) although tools are disabled"):
        await create_provider().acompletion([TASK], MODEL)
    assert calls(sdk, "cancel")


async def test_failed_run_reports_cursors_message(cursor):
    cursor(Turn([], status="error", error_message="Cloud Agent is not available for free users."))
    with pytest.raises(AgentExecutionError, match="Cursor run error: Cloud Agent is not available"):
        await create_provider().acompletion([TASK], MODEL)


async def test_sdk_errors_become_agent_errors_and_close_the_bridge(cursor):
    sdk = cursor(create_error=AuthenticationError("Invalid User API Key"))
    with pytest.raises(AgentExecutionError, match="Cursor rejected the API key: Invalid User API Key"):
        await create_provider().acompletion([TASK], MODEL)
    assert calls(sdk, "close")


async def test_a_failed_cancel_keeps_the_original_error(cursor):
    turn = Turn(
        [TextDeltaUpdate("par")], error=CursorSDKError("stream lost"), cancel_error=CursorSDKError("bridge gone")
    )
    sdk = cursor(turn)
    with pytest.raises(AgentExecutionError, match="Cursor error: stream lost"):
        await create_provider().acompletion([TASK], MODEL)
    assert calls(sdk, "cancel")


async def test_missing_api_key_fails_before_starting_the_bridge(cursor, monkeypatch):
    sdk = cursor(ok())
    monkeypatch.setattr(provider_module, "resolve_api_key", lambda name: None)
    with pytest.raises(AgentExecutionError, match="No Cursor API key found"):
        await create_provider().acompletion([TASK], MODEL)
    assert calls(sdk, "launch") == []


async def test_abandoning_a_stream_cancels_the_run(cursor):
    sdk = cursor(ok())
    provider = create_provider()
    stream = await provider.acompletion([TASK], MODEL, stream=True)
    await stream.__anext__()
    await stream.aclose()
    await provider.stop()

    assert calls(sdk, "cancel") and calls(sdk, "close")


async def test_list_models_reads_the_account_catalog(cursor):
    sdk = cursor()
    assert await create_provider().list_models() == [MODEL]
    assert calls(sdk, "close")
