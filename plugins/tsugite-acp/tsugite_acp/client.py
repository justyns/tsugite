"""ACPClientSession: handshake + prompt-turn loop bridging an ACP agent to tsugite.

The Client implementation captures session_update notifications onto an asyncio.Queue.
ACPClientSession.prompt() runs the agent's prompt() as a background task while draining
the queue, so the caller sees text/thought chunks as an async iterator and a final
"done" event with the stop reason and usage.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from collections import deque
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, AsyncIterator, Literal, get_args

from acp import PROTOCOL_VERSION, connect_to_agent
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    AgentThoughtChunk,
    AllowedOutcome,
    ClientCapabilities,
    FileSystemCapabilities,
    Implementation,
    RequestPermissionResponse,
    StopReason,
    ToolCallProgress,
    ToolCallStart,
)

from tsugite.exceptions import AgentExecutionError
from tsugite_acp.policy import PermissionPolicy

logger = logging.getLogger(__name__)

_STDERR_BUFFER_LINES = 200
# Stop reasons other than a clean end_turn / user cancel. The agent still did work
# under these (hit a token / turn-request cap, or refused), so we preserve the turn's
# content and note the truncation instead of raising and discarding everything.
_INCOMPLETE_STOP_REASONS = frozenset(get_args(StopReason)) - {"end_turn", "cancelled"}
# ToolCallProgress statuses worth surfacing; intermediate ticks (pending/in_progress)
# are dropped as noise.
_TERMINAL_TOOL_STATUS = frozenset({"completed", "failed"})


def _plugin_version() -> str:
    try:
        return version("tsugite-acp")
    except PackageNotFoundError:
        return "0.0.0"


@dataclass
class ACPEvent:
    """Yielded from ACPClientSession.prompt(). One of text/thought/tool/done."""

    kind: Literal["text", "thought", "tool", "done"]
    text: str = ""
    stop_reason: str | None = None
    usage: dict | None = None


class ACPClientHandler:
    """Implementation of acp.Client. Bridges agent notifications onto a queue.

    Capabilities advertised by ACPClientSession declare fs/terminal as unsupported,
    so the corresponding callbacks should never fire - they raise NotImplementedError
    if they do.
    """

    def __init__(self, policy: PermissionPolicy | None = None) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()
        self._policy = policy or PermissionPolicy()

    async def session_update(self, session_id: str, update, **_: Any) -> None:
        await self._queue.put(update)

    async def request_permission(self, options, session_id: str, tool_call, **_: Any) -> RequestPermissionResponse:
        if not options:
            raise AgentExecutionError("agent requested permission with no options")

        tool_name = self._tool_name_from_call(tool_call)
        params = self._tool_params_from_call(tool_call)
        action = self._policy.evaluate(tool_name, params)

        if action == "deny":
            chosen = self._first_option_of_kinds(options, ("reject_once", "reject_always"))
            if chosen is None:
                logger.warning("policy denied %s but no reject option offered; passing first option", tool_name)
                chosen = options[0]
            else:
                logger.warning("policy denied %s(%s)", tool_name, params)
        else:
            chosen = self._first_option_of_kinds(options, ("allow_once", "allow_always")) or options[0]

        return RequestPermissionResponse(outcome=AllowedOutcome(option_id=chosen.option_id, outcome="selected"))

    @staticmethod
    def _first_option_of_kinds(options, kinds: tuple[str, ...]):
        for opt in options:
            if opt.kind in kinds:
                return opt
        return None

    @staticmethod
    def _tool_name_from_call(tool_call) -> str:
        # ToolCallUpdate doesn't carry a tool name; the title is the closest proxy
        # used by claude-agent-acp (e.g. "Read foo", "Bash git status").
        title = getattr(tool_call, "title", "") or ""
        return title.split(" ", 1)[0] if title else ""

    @staticmethod
    def _tool_params_from_call(tool_call) -> dict:
        raw = getattr(tool_call, "raw_input", None)
        return dict(raw) if isinstance(raw, dict) else {}

    async def write_text_file(self, *_a: Any, **_kw: Any):  # pragma: no cover - unsupported
        raise NotImplementedError("write_text_file capability is unsupported")

    async def read_text_file(self, *_a: Any, **_kw: Any):  # pragma: no cover - unsupported
        raise NotImplementedError("read_text_file capability is unsupported")

    async def create_terminal(self, *_a: Any, **_kw: Any):  # pragma: no cover - unsupported
        raise NotImplementedError("terminal capability is unsupported")

    async def terminal_output(self, *_a: Any, **_kw: Any):  # pragma: no cover - unsupported
        raise NotImplementedError("terminal capability is unsupported")

    async def release_terminal(self, *_a: Any, **_kw: Any):  # pragma: no cover - unsupported
        raise NotImplementedError("terminal capability is unsupported")

    async def wait_for_terminal_exit(self, *_a: Any, **_kw: Any):  # pragma: no cover - unsupported
        raise NotImplementedError("terminal capability is unsupported")

    async def kill_terminal(self, *_a: Any, **_kw: Any):  # pragma: no cover - unsupported
        raise NotImplementedError("terminal capability is unsupported")

    async def ext_method(self, method: str, params: dict) -> dict:
        return {}

    async def ext_notification(self, method: str, params: dict) -> None:
        return None

    def on_connect(self, conn) -> None:  # noqa: ARG002
        return None


class ACPClientSession:
    """Lifecycle wrapper around an ACP ClientSideConnection.

    Constructor takes the connection + handler so tests can inject mocks. Production
    code uses a factory (see :func:`spawn_acp_session`) that wraps acp.spawn_agent_process.
    """

    def __init__(self, handler: ACPClientHandler, conn: Any) -> None:
        self._handler = handler
        self._conn = conn
        self._session_id: str | None = None
        self.agent_capabilities: AgentCapabilities | None = None
        self._process: asyncio.subprocess.Process | None = None  # set by spawn_acp_session
        self._stderr_task: asyncio.Task | None = None
        self._stderr_lines: deque[str] = deque(maxlen=_STDERR_BUFFER_LINES)

    @property
    def session_id(self) -> str | None:
        return self._session_id

    async def start(
        self,
        *,
        cwd: str,
        resume_session_id: str | None = None,
        mcp_servers: list | None = None,
    ) -> str:
        init_resp = await self._conn.initialize(
            protocol_version=PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapabilities(read_text_file=False, write_text_file=False),
                terminal=False,
            ),
            client_info=Implementation(name="tsugite-acp", version=_plugin_version()),
        )
        self.agent_capabilities = init_resp.agent_capabilities

        if resume_session_id:
            await self._conn.load_session(
                cwd=cwd,
                session_id=resume_session_id,
                mcp_servers=mcp_servers,
            )
            self._session_id = resume_session_id
        else:
            new_resp = await self._conn.new_session(cwd=cwd, mcp_servers=mcp_servers)
            self._session_id = new_resp.session_id
        return self._session_id

    async def prompt(self, blocks: list) -> AsyncIterator[ACPEvent]:
        if self._session_id is None:
            raise RuntimeError("ACPClientSession.start() must be called before prompt()")

        sentinel = object()
        prompt_task = asyncio.create_task(self._conn.prompt(prompt=blocks, session_id=self._session_id))

        async def signal_done() -> None:
            try:
                await prompt_task
            except BaseException:  # noqa: BLE001 - propagate via prompt_task.exception()
                pass
            finally:
                await self._handler._queue.put(sentinel)

        signal_task = asyncio.create_task(signal_done())

        try:
            while True:
                item = await self._handler._queue.get()
                if item is sentinel:
                    break
                event = self._convert_update(item)
                if event is not None:
                    yield event
        except BaseException:
            prompt_task.cancel()
            signal_task.cancel()
            raise
        finally:
            try:
                await signal_task
            except (asyncio.CancelledError, Exception):
                pass

        exc = prompt_task.exception()
        if exc is not None:
            raise exc

        resp = prompt_task.result()
        stop = resp.stop_reason
        if stop in _INCOMPLETE_STOP_REASONS:
            # The agent stopped without a clean end_turn (token/turn-request cap, or
            # refusal). It still did work, so surface a truncation note as content and
            # complete the turn rather than raising - otherwise the accumulated
            # text/tool activity is discarded and the turn renders empty in history.
            yield ACPEvent(kind="text", text=f"\n\n[ACP turn stopped: {stop}]")

        yield ACPEvent(
            kind="done",
            stop_reason=stop,
            usage=resp.usage.model_dump() if getattr(resp, "usage", None) else None,
        )

    @staticmethod
    def _convert_update(update) -> ACPEvent | None:
        if isinstance(update, AgentMessageChunk):
            return ACPEvent(kind="text", text=update.content.text)
        if isinstance(update, AgentThoughtChunk):
            return ACPEvent(kind="thought", text=update.content.text)
        if isinstance(update, (ToolCallStart, ToolCallProgress)):
            text = ACPClientSession._format_tool_update(update)
            return ACPEvent(kind="tool", text=text) if text else None
        return None

    @staticmethod
    def _format_tool_update(update) -> str | None:
        """Render a tool-call notification as a compact transcript line so the agent's
        executed tool/code activity reaches history. ToolCallStart (the invocation)
        always surfaces via its action-shaped title; ToolCallProgress surfaces only its
        terminal status so in-progress ticks don't spam the turn."""
        title = (getattr(update, "title", None) or "").strip()
        if isinstance(update, ToolCallStart):
            return f"\n\n[tool] {title}" if title else None
        status = getattr(update, "status", None)
        if status in _TERMINAL_TOOL_STATUS:
            return f"\n\n[tool] {title or 'tool'}: {status}"
        return None

    async def cancel(self) -> None:
        if self._session_id is not None:
            await self._conn.cancel(session_id=self._session_id)

    async def close(self) -> None:
        session_caps = getattr(self.agent_capabilities, "session_capabilities", None)
        if self._session_id is not None and getattr(session_caps, "close", None) is not None:
            try:
                await self._conn.close_session(session_id=self._session_id)
            except Exception as e:  # pragma: no cover - best effort shutdown
                logger.debug("close_session failed (continuing): %s", e)
        try:
            await self._conn.close()
        except Exception as e:  # pragma: no cover
            logger.debug("conn.close failed (continuing): %s", e)

        if self._stderr_task is not None:
            self._stderr_task.cancel()
            self._stderr_task = None

        if self._process is not None:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=2.0)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    self._process.kill()
                except ProcessLookupError:
                    pass
            self._process = None


async def spawn_acp_session(
    *,
    command: str,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    policy: PermissionPolicy | None = None,
) -> ACPClientSession:
    """Spawn an ACP agent subprocess and wire up an ACPClientSession.

    Caller is responsible for awaiting :meth:`ACPClientSession.start` to perform the
    initialize/new_session handshake, and :meth:`ACPClientSession.close` to terminate.
    """
    if shutil.which(command) is None and "/" not in command:
        raise RuntimeError(
            f"ACP agent command {command!r} not found on PATH. "
            "Install Node.js + npm and verify `npx --version`, or set TSUGITE_ACP_COMMAND."
        )

    process = await asyncio.create_subprocess_exec(
        command,
        *(args or []),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=cwd,
    )

    handler = ACPClientHandler(policy=policy)
    conn = connect_to_agent(handler, process.stdin, process.stdout)
    session = ACPClientSession(handler=handler, conn=conn)
    session._process = process
    session._stderr_task = asyncio.create_task(_drain_stream(process.stderr, session._stderr_lines))
    return session


async def _drain_stream(stream, sink: deque[str]) -> None:
    """Read lines from a stream into a bounded sink to keep the pipe from blocking."""
    try:
        while True:
            line = await stream.readline()
            if not line:
                break
            sink.append(line.decode(errors="replace").rstrip())
    except Exception:
        pass
