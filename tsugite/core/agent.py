"""Core agent implementation"""

import ast
import asyncio
import contextlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Cap the persisted return_value repr so a huge structured return doesn't bloat the event.
RETURN_VALUE_REPR_MAX = 2048

from tsugite.attachments.base import Attachment, AttachmentContentType, format_attachment_open_tag  # noqa: E402
from tsugite.cancellation import is_cancelled  # noqa: E402
from tsugite.events import (  # noqa: E402
    CodeExecutionEvent,
    ContentBlockEvent,
    CostSummaryEvent,
    EventBus,
    FinalAnswerEvent,
    LLMMessageEvent,
    LLMWaitProgressEvent,
    ObservationEvent,
    PromptSnapshotEvent,
    ReasoningContentEvent,
    ReasoningTokensEvent,
    StepStartEvent,
    StreamChunkEvent,
    StreamCompleteEvent,
    TaskStartEvent,
    WarningEvent,
)
from tsugite.providers.base import CompletionResponse as ProviderResponse  # noqa: E402
from tsugite.skill_discovery import Skill  # noqa: E402

from .content_blocks import extract_content_blocks  # noqa: E402
from .executor import Executor  # noqa: E402
from .executor_registry import get_executor_class  # noqa: E402
from .memory import AgentMemory, StepResult  # noqa: E402
from .tools import Tool  # noqa: E402

# Agent execution constants
DEFAULT_MAX_TURNS = 10  # Default maximum reasoning iterations before timeout

_LLM_WAIT_HEARTBEAT_INTERVAL = 10.0


def _safe_json(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return None
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return None


def _attachment_char_limit(name: str) -> int | None:
    """Return max chars for an attachment, or None for no limit.

    Currently returns None (no limit) for all attachments.
    Kept as a hook for future per-attachment size policies.
    """
    return None


CONTEXT_ACK = "Context loaded."


def estimate_content_tokens(content) -> int:
    """Rough token estimate for message content (string or multipart blocks)."""
    if isinstance(content, str):
        return len(content) // 4
    if isinstance(content, list):
        return sum(len(b.get("text", "")) // 4 if isinstance(b, dict) else 25 for b in content)
    return 100


def _trim_messages_to_token_budget(messages: List[Dict], budget_tokens: int) -> List[Dict]:
    """Keep the most recent messages that fit within a token budget.

    Walks from newest to oldest. Returns messages in original order.
    """
    if not messages:
        return messages

    kept_indices = []
    used = 0
    for i in range(len(messages) - 1, -1, -1):
        est_tokens = estimate_content_tokens(messages[i].get("content", ""))
        if used + est_tokens > budget_tokens and kept_indices:
            break
        used += est_tokens
        kept_indices.append(i)

    kept_indices.reverse()
    return [messages[i] for i in kept_indices]


def build_system_prompt(tools: List[Tool], instructions: str = "") -> str:
    """Build system prompt for LLM with tools and instructions.

    This is shared between TsugiteAgent and the render command to ensure
    consistency between what's shown and what's sent to the LLM.

    Args:
        tools: List of Tool objects available to the agent
        instructions: Additional instructions from agent config

    Returns:
        Complete system prompt string
    """
    tools_section = build_tools_section(tools)
    has_tools = bool(tools)
    return build_standard_mode_prompt(tools_section, instructions, has_tools)


# Only ```python-exec blocks are executed. A bare ```python block is treated as
# illustration (shown, not run) so an agent can quote Python in prose without the
# runtime executing it. See justyns/tsugite#479.
_EXEC_FENCE = "```python-exec"
_CLOSE_FENCE = "\n```"

# Start-of-line ```python fence whose info string is exactly "python" (bare) — i.e.
# NOT ```python-exec. Used to nudge the model toward the exec fence when it emits a
# bare block. The negative lookahead keeps ```python-exec from matching.
_BARE_PYTHON_FENCE = re.compile(r"(?:^|\n)```python(?!-exec)[ \t]*\r?\n")


def _find_parseable_close_fence(cleaned: str, code_start: int) -> Optional[int]:
    """Return the offset of the close fence whose preceding slice parses as Python, else None."""
    pos = code_start
    while True:
        close_at = cleaned.find(_CLOSE_FENCE, pos)
        if close_at == -1:
            return None
        try:
            ast.parse(cleaned[code_start:close_at].strip())
            return close_at
        except SyntaxError:
            pos = close_at + len(_CLOSE_FENCE)


def _find_python_blocks(cleaned: str) -> List[tuple[int, int]]:
    """Return (code_start, close_at) spans for every ``` ```python-exec ``` block on its own line."""
    blocks: List[tuple[int, int]] = []
    search_pos = 0
    while True:
        open_at = cleaned.find(_EXEC_FENCE, search_pos)
        if open_at == -1:
            return blocks
        if open_at != 0 and cleaned[open_at - 1] != "\n":
            search_pos = open_at + len(_EXEC_FENCE)
            continue
        code_start = open_at + len(_EXEC_FENCE)
        close_at = _find_parseable_close_fence(cleaned, code_start)
        if close_at is None:
            return blocks
        blocks.append((code_start, close_at))
        search_pos = close_at + len(_CLOSE_FENCE)


def _has_bare_python_fence(cleaned: str) -> bool:
    """True if the text contains a start-of-line bare ```python block (info string
    exactly "python", not ```python-exec)."""
    return _BARE_PYTHON_FENCE.search(cleaned) is not None


# Tags the runtime injects into the model's NEXT user message after executing
# its code (the execution result, multi-block warning, and budget hints). A
# well-behaved model never writes these itself; one that does is impersonating
# the runtime - usually hallucinating the execution loop and replaying stale
# data from a prior session, occasionally just quoting the protocol in prose.
_RUNTIME_TAG_NAMES = ("tsugite_execution_result", "tsugite_multi_block_warning", "tsugite_budget")


def escape_runtime_injection_tags(content: str) -> tuple[str, bool]:
    """Neutralize any runtime-only tags a model emitted, returning (escaped, found).

    Escaping (not dropping) the angle bracket keeps the response verbatim and
    non-destructive - a legitimate explanation of the protocol survives intact -
    while making the tag inert so it can't be mistaken for a real injection by:
      - the compaction eliding pass (its regex matches `<tsugite_execution_result>`),
      - the web UI's history renderer (a fabricated result would otherwise show
        as its own prose bubble - the post-reload double-render), or
      - the model itself, which would otherwise re-read its own fabricated
        results as fact on every subsequent turn.
    """
    if not content:
        return content, False
    found = False
    for name in _RUNTIME_TAG_NAMES:
        for raw, esc in ((f"<{name}", f"&lt;{name}"), (f"</{name}", f"&lt;/{name}")):
            if raw in content:
                found = True
                content = content.replace(raw, esc)
    return content, found


def _build_spoofed_runtime_tag_warning() -> str:
    """Model-visible note when a response contained a runtime-only tag. Tells the
    model those tags are runtime-injected (not its to write) so a hallucinated
    execution loop doesn't compound across turns."""
    return (
        "\n<tsugite_runtime_tag_notice>"
        "Your previous response wrote one or more runtime-only tags "
        "(tsugite_execution_result / tsugite_multi_block_warning / tsugite_budget). "
        "The runtime injects those AFTER it runs your code - do not write them yourself. "
        "They were neutralized; only the real execution result below is authoritative. "
        "Reply with prose or exactly one ```python-exec block."
        "</tsugite_runtime_tag_notice>"
    )


def _build_multi_block_warning_xml(count: int) -> str:
    """Model-visible note appended to a turn's observation when the agent
    received N>1 ```python blocks in one response.

    The parser only ever runs the first block; without this signal the model
    sees its full N-block response in raw_content but only one execution
    result and (reasonably) assumes the rest also ran. See justyns/tsugite#212.
    """
    return (
        f'\n<tsugite_multi_block_warning dropped="{count - 1}" total="{count}">'
        f"Your response contained {count} ```python-exec blocks. "
        f"Only block 1 was executed; blocks 2..{count} were dropped silently. "
        "If those blocks contained work that still needs to happen, re-emit them "
        "in your next response — exactly one ```python-exec block per turn. "
        "Do not assume the dropped blocks ran."
        "</tsugite_multi_block_warning>"
    )


def _build_bare_python_notice_xml() -> str:
    """Model-visible nudge when a response carried a bare ```python block but no
    executable ```python-exec block. A bare block is illustration (shown, not
    run); this tells the model which fence actually executes so a habit miss
    doesn't leave intended work silently unexecuted. See justyns/tsugite#479."""
    return (
        "\n<tsugite_bare_python_notice>"
        "Your response contained a ```python block. Bare ```python blocks are shown but "
        "NOT executed. If you meant to run that code, re-emit it as a ```python-exec block."
        "</tsugite_bare_python_notice>"
    )


@dataclass
class ParsedResponse:
    """Result from parsing an LLM response."""

    thought: str
    code: str
    content_blocks: Dict[str, str] = field(default_factory=dict)
    num_code_blocks: int = 0
    # True when the response carried a bare ```python block (not ```python-exec).
    # Drives a corrective nudge so the model learns which fence executes.
    has_bare_python: bool = False


@dataclass
class TurnResult:
    """Result from a single agent turn (LLM call + parsing)."""

    thought: str
    code: str
    step_cost: float
    content_blocks: Dict[str, str] = field(default_factory=dict)
    response: Optional[Any] = None
    num_code_blocks: int = 0
    # True when the model emitted a runtime-only tag (escaped before storage);
    # drives the model-facing notice so a hallucinated loop doesn't compound.
    spoofed_runtime_tag: bool = False
    # True when the response carried a bare ```python block (not ```python-exec).
    has_bare_python: bool = False


@dataclass
class AgentResult:
    """Result from agent execution."""

    output: Any
    token_usage: Optional[int] = None
    cost: Optional[float] = None
    steps: Optional[List[StepResult]] = None
    error: Optional[str] = None
    provider_state: Optional[Dict] = None
    last_input_tokens: Optional[int] = None

    def __str__(self) -> str:
        return self.output if self.output else self.error if self.error else ""


class TsugiteAgent:
    """Custom agent that uses Thought/Code/Observation loop.

    Supports reasoning models, custom parameters, and full control over
    the execution loop via pluggable provider backends.

    Example:
        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=[tool1, tool2],
            instructions="You are a helpful assistant",
            max_turns=10
        )

        result = await agent.run("Calculate 5 + 3")
        print(result)  # "8"
    """

    def __init__(
        self,
        model_string: str,
        tools: List[Tool],
        instructions: str = "",
        max_turns: int = DEFAULT_MAX_TURNS,
        executor: Optional[Executor] = None,
        model_kwargs: dict = None,
        event_bus: EventBus = None,
        model_name: str = None,
        attachments: List[Attachment] = None,
        skills: List[Skill] = None,
        expiring_skills: Optional[Dict[str, int]] = None,
        previous_messages: List[Dict] = None,
        resume_session: Optional[str] = None,
        resume_after_compaction: bool = False,
        hook_vars: Optional[Dict[str, str]] = None,
        storage: Optional[Any] = None,
        pre_llm_call: Optional[Callable] = None,
    ):
        """Initialize the agent.

        Args:
            model_string: Model identifier like "openai:gpt-4o-mini"
            tools: List of Tool objects the agent can use
            instructions: Additional instructions to append to system prompt
            max_turns: Maximum number of reasoning turns (think-act cycles) before giving up
            executor: Code executor (microsandbox or local). If None, uses LocalExecutor
            model_kwargs: Extra parameters for the provider (reasoning_effort, response_format, etc.)
            event_bus: Optional EventBus for broadcasting events
            model_name: Optional display name for the model (for UI)
            attachments: List of Attachment objects for multi-modal inputs
            skills: List of Skill objects for loaded skills
            previous_messages: List of previous conversation messages (user/assistant pairs)
            hook_vars: Dict of pre_message hook captured outputs (e.g. rag_context)
        """
        from tsugite.models import get_model_kwargs, get_provider_and_model

        self.model_string = model_string
        self.tools = tools
        self.instructions = instructions
        self.max_turns = max_turns
        self.executor = executor or get_executor_class()()
        self.memory = AgentMemory()
        self.event_bus = event_bus
        self.model_name = model_name or model_string
        self.attachments = attachments or []
        self.skills = skills or []
        # Map of skill name -> turns_remaining. Surfaced as <skill_expiring> blocks
        # in the context turn so the agent knows the skill will auto-unload if
        # unreferenced.
        self.expiring_skills: Dict[str, int] = dict(expiring_skills or {})
        self.previous_messages = previous_messages or []
        self.hook_vars = hook_vars or {}
        self._pre_llm_call = pre_llm_call
        self._resume_session = resume_session
        self._resume_after_compaction = resume_after_compaction

        self.total_cost = 0.0
        # Distinguishes "provider reported $0" (subscription models) from "no
        # cost data at all" (interrupted turn) - only the latter records NULL.
        self.cost_reported = False
        self.total_tokens = 0
        self.last_input_tokens = 0
        self.cache_creation_tokens = 0
        self.cache_read_tokens = 0
        self._previous_turn_had_error = False
        self.storage = storage
        self._user_input_recorded = False  # caller may pre-record before run()

        self.tool_map = {tool.name: tool for tool in tools}

        self._inject_tools_into_executor()

        self._provider_name, self._provider, self._model_id = get_provider_and_model(model_string)
        self._model_kwargs = get_model_kwargs(model_string, **(model_kwargs or {}))

        self._provider.set_context(
            resume_session=resume_session,
            resume_after_compaction=resume_after_compaction,
            attachments=self.attachments,
            skills=self.skills,
            previous_messages=self.previous_messages,
        )

    def _inject_tools_into_executor(self):
        """Register tools with the executor; each backend handles dispatch its own way."""
        self.executor.set_tools(self.tools, event_bus=self.event_bus)

    async def run(self, task: str, return_full_result: bool = False, stream: bool = False):
        """Run the agent on a task.

        Args:
            task: The task to solve
            return_full_result: If True, return AgentResult with metadata
            stream: If True, stream the response chunks in real-time

        Returns:
            str: The final answer from the agent
            or AgentResult: Full result with token usage and turns

        Raises:
            RuntimeError: If agent reaches max_turns without finishing
        """
        start_time = time.time()
        self.memory.add_task(task)
        if self.event_bus:
            self.event_bus.emit(TaskStartEvent(task=task, model=self.model_name))

        self._record_user_input_if_needed(task)

        unset = object()
        final_value: Any = unset
        last_response_text: str = ""
        turn_num = 0
        cancelled = False

        try:
            for turn_num in range(self.max_turns):
                # Cooperative cancel checkpoint (between turns): the daemon runs this
                # loop in a worker thread that can't be preempted, so a user Stop is
                # honored here rather than killing the thread. See tsugite/cancellation.py.
                if is_cancelled():
                    cancelled = True
                    break

                if self.event_bus:
                    self.event_bus.emit(
                        StepStartEvent(
                            step=turn_num + 1,
                            max_turns=self.max_turns,
                            recovering_from_error=self._previous_turn_had_error,
                        )
                    )

                messages = self._build_messages()
                logger.debug("Turn %d sending %d messages", turn_num + 1, len(messages))

                if self.event_bus:
                    self.event_bus.emit(
                        PromptSnapshotEvent(
                            messages=messages,
                            token_breakdown=self._compute_token_breakdown(messages),
                        )
                    )

                turn = await self._provider_turn(messages, turn_num, stream)
                thought, code = turn.thought, turn.code
                last_response_text = turn.response.content if turn.response else (thought or "")

                if self.event_bus and (thought or code):
                    self.event_bus.emit(
                        PromptSnapshotEvent(messages=messages + [{"role": "assistant", "content": last_response_text}])
                    )

                if turn.content_blocks:
                    if self.event_bus:
                        for name, content in turn.content_blocks.items():
                            self.event_bus.emit(ContentBlockEvent(name=name, content=content))
                    await self.executor.inject_content_blocks(turn.content_blocks)

                # Multiple python blocks: the parser already took just the first
                # one as `code`. Surface the drop on two channels — a UI warning
                # for the human, and an in-conversation observation for the
                # model. Without the latter the model sees its full N-block
                # response in raw_content but only one execution result, and
                # (reasonably) assumes the rest also ran. See #212.
                multi_block_count = turn.num_code_blocks if turn.num_code_blocks > 1 else 0
                if multi_block_count and self.event_bus:
                    self.event_bus.emit(
                        WarningEvent(
                            message=(
                                f"Response contained {multi_block_count} ```python-exec blocks; "
                                "only the first was executed, the rest were dropped."
                            ),
                            category="multi_code_block",
                            step=turn_num + 1,
                        )
                    )

                # No code = the model is done. Its raw text is the answer.
                if not code or not code.strip():
                    final_value = last_response_text
                    trailing_notice = ""
                    if multi_block_count:
                        trailing_notice += _build_multi_block_warning_xml(multi_block_count)
                    if turn.spoofed_runtime_tag:
                        trailing_notice += _build_spoofed_runtime_tag_warning()
                    # The model wrote a bare ```python block instead of ```python-exec;
                    # it wasn't executed. Nudge it toward the exec fence for next turn.
                    if turn.has_bare_python:
                        trailing_notice += _build_bare_python_notice_xml()
                    self.memory.add_step(
                        thought=thought,
                        code="",
                        output="",
                        tools_called=[],
                        content_blocks=turn.content_blocks,
                        raw_content=last_response_text,
                        xml_observation=trailing_notice or None,
                    )
                    break

                # Cooperative cancel checkpoint (before running a tool/code block):
                # honor a Stop that landed after the model responded but before its
                # code executes, so no further side effects run.
                if is_cancelled():
                    cancelled = True
                    break

                if self.event_bus:
                    self.event_bus.emit(CodeExecutionEvent(code=code))

                exec_start = time.perf_counter()
                # `code` is the raw LLM string; never substitute an escaped observation (that's model input, not exec input).
                exec_result = await self.executor.execute(code)
                exec_duration_ms = int((time.perf_counter() - exec_start) * 1000)

                self._record_code_execution(
                    code=code,
                    exec_result=exec_result,
                    duration_ms=exec_duration_ms,
                )

                xml_observation = exec_result.to_xml(duration_ms=exec_duration_ms)

                if self.event_bus:
                    from tsugite.secrets.registry import get_registry

                    masked = get_registry().mask(exec_result.output)
                    if exec_result.error:
                        self._previous_turn_had_error = True
                        preview = exec_result.error[:100] + "..." if len(exec_result.error) > 100 else exec_result.error
                        self.event_bus.emit(
                            WarningEvent(message=f"Tool failed, will retry: {preview}", step=turn_num + 1)
                        )
                    else:
                        self._previous_turn_had_error = False
                        self.event_bus.emit(ObservationEvent(observation=masked))

                # Multi-block extras were dropped earlier; tell the model so it
                # knows to re-emit them rather than assume they ran.
                if multi_block_count:
                    xml_observation += _build_multi_block_warning_xml(multi_block_count)

                # The model wrote a runtime-only tag (now escaped). Tell it not to,
                # so a hallucinated execution loop doesn't compound across turns.
                if turn.spoofed_runtime_tag:
                    xml_observation += _build_spoofed_runtime_tag_warning()

                budget_tag = self._build_budget_tag(turn_num)
                xml_observation += budget_tag

                self.memory.add_step(
                    thought=thought,
                    code=code,
                    output=exec_result.output + budget_tag,
                    error=exec_result.error,
                    tools_called=exec_result.tools_called,
                    loaded_skills=exec_result.loaded_skills,
                    unloaded_skills=exec_result.unloaded_skills,
                    xml_observation=xml_observation,
                    content_blocks=turn.content_blocks,
                    raw_content=last_response_text,
                )

                self._absorb_skill_changes(exec_result)

                if exec_result.return_value is not None:
                    final_value = exec_result.return_value
                    self.memory.add_final_answer(final_value)
                    break

            # Cancelled at a checkpoint: keep whatever the model last produced as the
            # partial answer and record the run as cancelled so partial work persists.
            if cancelled:
                final_value = last_response_text
                status = "cancelled"
                error_message = "Cancelled by user"
                if self.event_bus:
                    self.event_bus.emit(WarningEvent(message=error_message, step=turn_num + 1))
            # If we never broke out, max_turns hit. Use the last response text as
            # the answer and record the run as interrupted.
            elif final_value is unset:
                final_value = last_response_text
                status = "interrupted"
                error_message = f"max_turns ({self.max_turns}) reached"
                if self.event_bus:
                    self.event_bus.emit(WarningEvent(message=error_message, step=turn_num + 1))
            else:
                status = "success"
                error_message = None

            total_tokens = self.total_tokens if self.total_tokens > 0 else None
            answer_text = str(final_value) if final_value is not None else ""
            answer_data = _safe_json(final_value)
            response_context = {
                "answer": answer_text[:500],
                "turns": turn_num + 1,
                "tokens": total_tokens,
                "cost": self.reported_cost,
            }

            from tsugite.hooks import fire_hooks_background

            fire_hooks_background("pre_response", response_context)

            if self.event_bus:
                self.event_bus.emit(
                    FinalAnswerEvent(
                        answer=answer_text,
                        answer_data=answer_data,
                        turns=turn_num + 1,
                        tokens=total_tokens,
                        cost=self.reported_cost,
                    )
                )
                self.event_bus.emit(
                    CostSummaryEvent(
                        tokens=total_tokens,
                        cost=self.reported_cost,
                        model=self.model_name,
                        duration_seconds=time.time() - start_time,
                        cache_creation_input_tokens=self.cache_creation_tokens or None,
                        cache_read_input_tokens=self.cache_read_tokens or None,
                    )
                )

            fire_hooks_background("post_response", response_context)

            if self.storage:
                from tsugite.agent_runner.history_integration import record_final_result, record_session_end

                # Durable answer record for run paths with no live SSE persist
                # (scheduled, subprocess, CLI) — the conversation view renders
                # the answer bubble from this event.
                record_final_result(
                    self.storage,
                    result=answer_text,
                    result_data=answer_data,
                    turns=turn_num + 1,
                    tokens=total_tokens,
                    cost=self.reported_cost,
                )
                record_session_end(self.storage, status=status, error_message=error_message)

            if return_full_result:
                return AgentResult(
                    output=final_value,
                    token_usage=total_tokens,
                    cost=self.reported_cost,
                    steps=self.memory.steps,
                    error=error_message,
                    provider_state=self._provider.get_state(),
                    last_input_tokens=self.last_input_tokens if self.last_input_tokens > 0 else None,
                )
            return final_value
        finally:
            await self._provider.stop()

    def _record_user_input_if_needed(self, task: str) -> None:
        if self.storage and not self._user_input_recorded:
            self.storage.record("user_input", text=task)
            self._user_input_recorded = True

    def _record_code_execution(self, code: str, exec_result, duration_ms: int) -> None:
        if not self.storage:
            return
        # Mask secrets BEFORE persisting: the live observation (to_xml) masks, but
        # the stored event must too - otherwise the raw value sits on disk and is
        # replayed verbatim into the model on continuation (reconstruction only
        # escapes, never masks).
        from tsugite.secrets.registry import get_registry

        mask = get_registry().mask
        # Persist what the runtime already knows about the executed block so replay is
        # deterministic instead of regex-scraping raw_content. Store the return value as
        # a masked repr string (never json.dumps - it may be an arbitrary,
        # non-serializable object).
        rv = exec_result.return_value
        return_value_repr = mask(repr(rv))[:RETURN_VALUE_REPR_MAX] if rv is not None else None
        return_value_type = type(rv).__name__ if rv is not None else None
        state_keys = list(exec_result.state_keys) if exec_result.state_keys else None
        self.storage.record(
            "code_execution",
            code=code,
            output=mask(exec_result.output) if exec_result.output else exec_result.output,
            error=mask(exec_result.error) if exec_result.error else exec_result.error,
            duration_ms=duration_ms,
            tools_called=list(exec_result.tools_called) if exec_result.tools_called else None,
            last_statement_type=exec_result.last_statement_type,
            return_value_repr=return_value_repr,
            return_value_type=return_value_type,
            state_keys=state_keys,
        )

    def _absorb_skill_changes(self, exec_result) -> None:
        if exec_result.loaded_skills:
            existing = {s.name for s in self.skills}
            for name, content in exec_result.loaded_skills.items():
                if name not in existing:
                    self.skills.append(Skill(name=name, content=content))
                    existing.add(name)
                    if self.storage:
                        self.storage.record("skill_added", name=name)
        if exec_result.unloaded_skills:
            drop = set(exec_result.unloaded_skills)
            self.skills = [s for s in self.skills if s.name not in drop]
            for name in drop:
                self.expiring_skills.pop(name, None)
                if self.storage:
                    self.storage.record("skill_removed", name=name)

    @contextlib.asynccontextmanager
    async def _llm_wait_heartbeat(self):
        """No-op when there's no event_bus."""
        if not self.event_bus:
            yield
            return
        started_at = time.monotonic()

        async def emit_loop():
            while True:
                await asyncio.sleep(_LLM_WAIT_HEARTBEAT_INTERVAL)
                self.event_bus.emit(LLMWaitProgressEvent(elapsed_seconds=int(time.monotonic() - started_at)))

        task = asyncio.create_task(emit_loop())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _provider_turn(self, messages, turn_num, stream) -> TurnResult:
        """Execute one turn via the provider system."""
        if self._pre_llm_call is not None:
            await self._pre_llm_call(messages, self._model_id)

        self._record_model_request(messages, turn_num)

        async with self._llm_wait_heartbeat():
            if stream:
                return await self._provider_turn_streaming(messages, turn_num)
            return await self._provider_turn_blocking(messages, turn_num)

    async def _provider_turn_streaming(self, messages, turn_num) -> TurnResult:
        accumulated_content = ""
        accumulated_reasoning = ""
        step_cost = 0.0
        final_chunk = None
        result = await self._provider.acompletion(
            messages=messages, model=self._model_id, stream=True, **self._model_kwargs
        )

        async for chunk in result:
            if chunk.content:
                accumulated_content += chunk.content
                if self.event_bus:
                    self.event_bus.emit(StreamChunkEvent(chunk=chunk.content))
            if getattr(chunk, "reasoning_content", ""):
                accumulated_reasoning += chunk.reasoning_content
                if self.event_bus:
                    self.event_bus.emit(ReasoningContentEvent(content=chunk.reasoning_content, step=turn_num + 1))
            if chunk.done:
                final_chunk = chunk

        if self.event_bus:
            self.event_bus.emit(StreamCompleteEvent())

        if final_chunk and final_chunk.usage:
            step_cost = self._accumulate_usage(final_chunk.usage, final_chunk.cost)

        accumulated_content, spoofed = escape_runtime_injection_tags(accumulated_content)
        parsed = self._parse_response_from_text(accumulated_content)
        if accumulated_reasoning:
            self.memory.add_reasoning(accumulated_reasoning)
        self._record_model_response(
            turn_num,
            raw_content=accumulated_content,
            usage=final_chunk.usage if final_chunk else None,
            cost=final_chunk.cost if final_chunk else None,
            response=None,
        )

        synthetic = ProviderResponse(content=accumulated_content)
        return TurnResult(
            thought=parsed.thought,
            code=parsed.code,
            step_cost=step_cost,
            content_blocks=parsed.content_blocks,
            response=synthetic,
            num_code_blocks=parsed.num_code_blocks,
            spoofed_runtime_tag=spoofed,
            has_bare_python=parsed.has_bare_python,
        )

    async def _provider_turn_blocking(self, messages, turn_num) -> TurnResult:
        response: ProviderResponse = await self._provider.acompletion(
            messages=messages, model=self._model_id, stream=False, **self._model_kwargs
        )
        response.content, spoofed = escape_runtime_injection_tags(response.content)
        parsed = self._parse_response_from_text(response.content)

        step_cost = response.cost or 0.0
        if response.usage:
            self._accumulate_usage(response.usage, response.cost)
        else:
            if response.cost is not None:
                self.cost_reported = True
            self.total_cost += step_cost

        if response.reasoning_content:
            self.memory.add_reasoning(response.reasoning_content)
            if self.event_bus:
                self.event_bus.emit(ReasoningContentEvent(content=response.reasoning_content, step=turn_num + 1))

        if self.event_bus and response.usage and response.usage.reasoning_tokens:
            self.event_bus.emit(ReasoningTokensEvent(tokens=response.usage.reasoning_tokens, step=turn_num + 1))

        # Only emit thought prose. Falling back to response.content would include the
        # raw ```python-exec fence, causing the UI to render the code block twice (once
        # inside the thought markdown, once as a separate code-execution event).
        if self.event_bus and parsed.thought and parsed.thought.strip():
            self.event_bus.emit(
                LLMMessageEvent(content=parsed.thought, title=f"Turn {turn_num + 1} Reasoning", step=turn_num + 1)
            )

        self._record_model_response(
            turn_num,
            raw_content=response.content,
            usage=response.usage,
            cost=response.cost,
            response=response,
        )

        return TurnResult(
            thought=parsed.thought,
            code=parsed.code,
            step_cost=step_cost,
            content_blocks=parsed.content_blocks,
            response=response,
            num_code_blocks=parsed.num_code_blocks,
            spoofed_runtime_tag=spoofed,
            has_bare_python=parsed.has_bare_python,
        )

    def _record_model_request(self, messages, turn_num: int) -> None:
        if not self.storage:
            return
        # Store a hash of the sent messages, not the array itself: reconstruction rebuilds
        # the messages from the other events on demand, so persisting the full array every
        # turn just re-stored the whole conversation N times.
        from tsugite.history.models import dedup_model_request_data

        data = dedup_model_request_data(
            {
                "messages": messages,
                "turn": turn_num,
                "provider": self._provider_name,
                "model": self._model_id,
                "tool_names": [t.name for t in self.tools],
            }
        )
        self.storage.record("model_request", **data)

    def _record_model_response(self, turn_num: int, *, raw_content: str, usage, cost, response) -> None:
        if not self.storage:
            return
        usage_dump = usage.model_dump(exclude_none=True) if usage and hasattr(usage, "model_dump") else None
        state_delta = self._provider.get_state() if self._provider else None
        raw = getattr(response, "raw", None) if response is not None else None
        stop_reason = raw.get("stop_reason") if isinstance(raw, dict) else None
        self.storage.record(
            "model_response",
            turn=turn_num,
            provider=self._provider_name,
            model=self._model_id,
            raw_content=raw_content,
            usage=usage_dump,
            cost=cost,
            stop_reason=stop_reason,
            state_delta=state_delta,
        )

    def _format_attachment(self, attachment: Attachment) -> Optional[Dict]:
        """Format an attachment for the provider based on its content type.

        Args:
            attachment: Attachment object to format

        Returns:
            Formatted content block for the provider, or None if invalid
        """
        if attachment.content_type == AttachmentContentType.TEXT:
            # Text attachment - wrap in XML tags
            return {
                "type": "text",
                "text": f"<Attachment: {attachment.name}>\n{attachment.content}\n</Attachment: {attachment.name}>",
            }

        elif attachment.content_type == AttachmentContentType.IMAGE:
            # Image attachment
            if attachment.source_url:
                # URL reference
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": attachment.source_url,
                        "format": attachment.mime_type,
                    },
                }
            elif attachment.content:
                # Base64 encoded image
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{attachment.mime_type};base64,{attachment.content}",
                    },
                }

        elif attachment.content_type == AttachmentContentType.AUDIO:
            # Audio attachment
            if attachment.source_url:
                # URL reference
                # Note: Some models may not support audio URLs directly
                return {
                    "type": "input_audio",
                    "input_audio": {
                        "data": attachment.source_url,
                        "format": attachment.mime_type.split("/")[-1] if "/" in attachment.mime_type else "wav",
                    },
                }
            elif attachment.content:
                # Base64 encoded audio
                audio_format = attachment.mime_type.split("/")[-1] if "/" in attachment.mime_type else "wav"
                return {
                    "type": "input_audio",
                    "input_audio": {
                        "data": attachment.content,
                        "format": audio_format,
                    },
                }

        elif attachment.content_type == AttachmentContentType.DOCUMENT:
            # Document attachment (PDF, etc.)
            if attachment.source_url:
                # URL reference
                return {
                    "type": "file",
                    "file": {
                        "file_id": attachment.source_url,
                        "format": attachment.mime_type,
                    },
                }
            elif attachment.content:
                # Base64 encoded document
                return {
                    "type": "file",
                    "file": {
                        "file_data": f"data:{attachment.mime_type};base64,{attachment.content}",
                    },
                }

        return None

    def _build_context_turn(self) -> list | None:
        """Build context turn content with attachments and auto-loaded skills.

        Context is injected as a user/assistant pair for prompt cache stability.
        System message stays constant, context turn gets cached after first call.

        Returns:
            List of content blocks, or None if no context
        """
        if not self.attachments and not self.skills:
            return None

        blocks = []
        text_parts = ["<context>"]

        model_info = self._provider.get_model_info(self._model_id)
        model_supports_vision = model_info.supports_vision if model_info else True

        for att in self.attachments:
            open_tag = format_attachment_open_tag(att)
            if att.content_type == AttachmentContentType.TEXT:
                text_parts.append(open_tag)
                text_parts.append(att.content)
                text_parts.append("</attachment>")
            elif att.content_type == AttachmentContentType.IMAGE and not model_supports_vision:
                text_parts.append(f"{open_tag}[Image: {att.name}]</attachment>")
            else:
                block = self._format_attachment(att)
                if block:
                    blocks.append(block)

        # Skills wrapped per the agentskills.io client-implementation guidance,
        # so the block is identifiable for compaction-protection and downstream tools.
        for skill in self.skills:
            text_parts.append(f'<skill_content name="{skill.name}">')
            text_parts.append(skill.content)
            text_parts.append("</skill_content>")
            remaining = self.expiring_skills.get(skill.name)
            if remaining is not None:
                text_parts.append(f'<skill_expiring name="{skill.name}" turns_remaining="{remaining}">')
                text_parts.append(
                    f"This skill will auto-unload in {remaining} turn(s) unless referenced. "
                    f'Call load_skill("{skill.name}") to renew, or unload_skill("{skill.name}") to drop now.'
                )
                text_parts.append("</skill_expiring>")

        text_parts.append("</context>")

        return [{"type": "text", "text": "\n".join(text_parts)}] + blocks

    def _build_observation(self, step) -> str:
        """Build the observation string that replays as a user message.

        Dynamically-loaded skill content does not live here; it is promoted into
        `self.skills` after each step so the cached context turn carries it
        forward. That keeps skill content compaction-immune and avoids
        duplicating large skill bodies in every turn's observation replay.

        Args:
            step: StepResult with execution output

        Returns:
            Observation string (tool results, code output, errors).
        """
        return step.xml_observation

    def _build_messages(self) -> List[Dict]:
        """Build message list for LLM from memory.

        Uses a context turn pattern for better prompt cache stability:
        - System message is stable (no attachments/skills)
        - Attachments and auto-loaded skills go in a context turn
        - Dynamically-loaded skills are embedded in observations

        Format:
        [
            {"role": "system", "content": system_prompt},           # STABLE
            {"role": "user", "content": <context>...</context>},    # Cached
            {"role": "assistant", "content": "Context loaded."},    # Cached
            {"role": "user", "content": "previous turn 1"},
            {"role": "assistant", "content": "previous response 1"},
            {"role": "user", "content": task},
            {"role": "assistant", "content": "```python-exec\\n...```"},
            {"role": "user", "content": <loaded_skill>...</loaded_skill>\\n<observation>..."},
            ...
        ]
        """
        messages = []

        # 1. Stable system message (never changes mid-conversation)
        messages.append({"role": "system", "content": self._build_system_prompt()})

        # 2. Context turn (attachments + auto-loaded skills)
        context = self._build_context_turn()
        if context:
            messages.append({"role": "user", "content": context, "cache_control": {"type": "ephemeral"}})
            messages.append({"role": "assistant", "content": CONTEXT_ACK, "cache_control": {"type": "ephemeral"}})

        # 3. Previous conversation messages (if continuing a conversation)
        if self.previous_messages:
            messages.extend(self.previous_messages)

        # 4. Task
        messages.append({"role": "user", "content": self.memory.task})

        # 5. Previous steps. Use the verbatim raw_content so the model sees its
        # own past response unchanged. Fall back to a re-rendered code block for
        # legacy steps that lack raw_content.
        for step in self.memory.steps:
            if step.raw_content:
                assistant_msg = step.raw_content
            elif step.code and step.code.strip():
                assistant_msg = f"```python-exec\n{step.code}\n```"
            else:
                assistant_msg = step.thought if step.thought else "(empty response)"
            messages.append({"role": "assistant", "content": assistant_msg})

            if step.xml_observation:
                messages.append({"role": "user", "content": self._build_observation(step)})

        return messages

    def _compute_token_breakdown(self, messages: List[Dict]) -> Dict:
        """Compute per-category token breakdown with individual item details."""
        est = estimate_content_tokens
        categories = []

        # Instructions (system prompt minus tools)
        instructions_tokens = est(self.instructions) if self.instructions else 0
        categories.append({"name": "instructions", "tokens": instructions_tokens, "items": []})

        # Tools — per-tool breakdown
        tool_items = []
        for tool in self.tools:
            tok = est(tool.to_code_prompt())
            tool_items.append({"name": tool.name, "tokens": tok})
        tool_items.sort(key=lambda x: x["tokens"], reverse=True)
        tools_total = sum(t["tokens"] for t in tool_items)
        categories.append({"name": "tools", "tokens": tools_total, "items": tool_items})

        # Attachments — per-attachment breakdown
        att_items = []
        for att in self.attachments:
            tok = est(att.content) if att.content else 0
            att_items.append({"name": att.name, "tokens": tok})
        att_total = sum(a["tokens"] for a in att_items)
        categories.append({"name": "attachments", "tokens": att_total, "items": att_items})

        # Skills — per-skill breakdown
        skill_items = []
        for skill in self.skills:
            tok = est(skill.content) if skill.content else 0
            skill_items.append({"name": skill.name, "tokens": tok})
        skills_total = sum(s["tokens"] for s in skill_items)
        categories.append({"name": "skills", "tokens": skills_total, "items": skill_items})

        # Hooks (pre_message hook captured output like rag_context)
        hook_items = []
        for name, content in self.hook_vars.items():
            tok = est(content) if content else 0
            hook_items.append({"name": name, "tokens": tok})
        hooks_total = sum(h["tokens"] for h in hook_items)
        categories.append({"name": "hooks", "tokens": hooks_total, "items": hook_items})

        # History — walk messages between context turn and task
        history_tokens = 0
        i = 0
        n = len(messages)
        if i < n and messages[i].get("role") == "system":
            i += 1
        if i + 1 < n and messages[i + 1].get("content") == CONTEXT_ACK:
            i += 2
        task_content = self.memory.task if self.memory else None
        while i < n:
            if messages[i].get("role") == "user" and messages[i].get("content") == task_content:
                break
            content = messages[i].get("content", "")
            text = content if isinstance(content, str) else ""
            if not (text.startswith("<context>") or text.startswith("<context_update>")):
                history_tokens += est(content)
            i += 1
        categories.append({"name": "history", "tokens": history_tokens, "items": []})

        # Task
        task_tokens = est(task_content) if task_content else 0
        categories.append({"name": "task", "tokens": task_tokens, "items": []})

        # Steps
        steps_tokens = 0
        if i < n:
            i += 1  # skip task message
        while i < n:
            steps_tokens += est(messages[i].get("content", ""))
            i += 1
        categories.append({"name": "steps", "tokens": steps_tokens, "items": []})

        total = sum(c["tokens"] for c in categories)
        return {"categories": categories, "total": total}

    def _build_system_prompt(self) -> str:
        """Build system prompt that teaches LLM how to solve tasks."""
        return build_system_prompt(self.tools, self.instructions)

    def _build_budget_tag(self, turn_num: int) -> str:
        """Build XML budget tag showing turn and token usage for the LLM."""
        turn = turn_num + 1
        parts = [f'turn="{turn}"', f'max_turns="{self.max_turns}"']
        if self.total_tokens > 0:
            parts.append(f'tokens_used="{self.total_tokens}"')
        if self.max_turns - turn <= 2:
            parts.append('warning="approaching turn limit, wrap up soon"')
        return f"\n<tsugite_budget {' '.join(parts)} />"

    @property
    def reported_cost(self) -> float | None:
        """Cumulative cost when any provider response carried one, else None."""
        return self.total_cost if (self.cost_reported or self.total_cost > 0) else None

    def _accumulate_usage(self, usage, cost: float | None = None) -> float:
        """Update cumulative token/cost counters from a usage object.

        Returns the step cost for caller convenience.
        """
        self.total_tokens += usage.total_tokens
        self.last_input_tokens = (
            usage.prompt_tokens + (usage.cache_creation_input_tokens or 0) + (usage.cache_read_input_tokens or 0)
        )
        self.cache_creation_tokens += usage.cache_creation_input_tokens or 0
        self.cache_read_tokens += usage.cache_read_input_tokens or 0
        if cost is not None:
            self.cost_reported = True
        self.total_cost += cost or 0.0
        return cost or 0.0

    def _parse_response_from_text(self, content: str) -> ParsedResponse:
        """Parse text content into thought, code, and content blocks."""
        cleaned, content_blocks = extract_content_blocks(content)

        blocks = _find_python_blocks(cleaned)
        num_code_blocks = len(blocks)

        code = ""
        if blocks:
            start, end = blocks[0]
            code = cleaned[start:end].strip()
        else:
            # No block parsed cleanly. If there's still a ```python-exec opener,
            # fall back to the first naive close fence so the LLM gets a
            # SyntaxError back instead of empty code (which would look like
            # "model is done").
            opener = cleaned.find(_EXEC_FENCE)
            if opener != -1:
                code_start = opener + len(_EXEC_FENCE)
                fallback_end = cleaned.find(_CLOSE_FENCE, code_start)
                if fallback_end != -1:
                    code = cleaned[code_start:fallback_end].strip()

        first_open = cleaned.find(_EXEC_FENCE)
        prose_end = first_open if first_open != -1 else len(cleaned)
        thought_start = cleaned.find("Thought:")
        if thought_start != -1:
            thought = cleaned[thought_start + len("Thought:") : prose_end].strip()
        else:
            thought = cleaned[:prose_end].strip()

        return ParsedResponse(
            thought=thought,
            code=code,
            content_blocks=content_blocks,
            num_code_blocks=num_code_blocks,
            has_bare_python=_has_bare_python_fence(cleaned),
        )


def build_tools_section(tools: List[Tool]) -> str:
    """Build the tools section of the system prompt.

    Args:
        tools: List of Tool objects available to the agent

    Returns:
        Formatted tools section or empty string if no tools
    """
    if not tools:
        return ""

    tool_definitions = "\n\n".join([tool.to_code_prompt() for tool in tools])
    return f"""
## Available functions:

You have access to these Python functions:

```python
{tool_definitions}
```
"""


def build_standard_mode_prompt(tools_section: str, instructions: str, has_tools: bool) -> str:
    """Build system prompt for standard mode (code blocks required).

    Args:
        tools_section: Formatted tools section
        instructions: Additional instructions from agent config
        has_tools: Whether tools are available

    Returns:
        Complete system prompt for standard mode
    """
    import os

    tool_rule = (
        "3. Call functions with keyword arguments: result = tool_name(arg1=value1, arg2=value2)"
        if has_tools
        else "3. Use standard Python to solve the task"
    )

    cwd = os.getcwd()

    return f"""You are an expert assistant who solves tasks using Python code.

## How to Respond

Each turn you can either:

1. **Run Python code** to use tools, read files, compute things — wrap it in a single
   ```python-exec code block. You'll see the result and can run more code next turn.

2. **Answer directly with text** — when you're done, just respond with your answer
   in plain text (no code block). That ends the run; the user sees your text.

```python-exec
config = read_file("config.yaml")
print(config)
```

Only ```python-exec blocks are executed. A plain ```python block is treated as
illustration — it is shown to the user but NOT run — so you can quote or explain
Python without executing it.

## Current Working Directory

{cwd}

## Execution Results

After your code block runs, the runtime injects a `tsugite_execution_result`
element into your next user-role message. It carries:

- `status` attribute: "success" or "error"
- `output` child: stdout from your `print()` calls
- `error` + `traceback` children: present only on failure (traceback truncated to last 10 lines)
- `variables_set` child: variables created this turn (discarded at turn end)
- `state` child: values persisted in `state` (carry across turns)
- `return_value` child: the value you passed to `return_value()` — ends the run

**Critical — these tags are runtime output, never your input.** Never write
`tsugite_execution_result`, `tsugite_multi_block_warning`, or `tsugite_budget`
tags inside your own response — not even to predict, illustrate, or reason about
what a result will look like. The runtime injects these; if one appears in your
response, you have hallucinated and any reasoning that follows it is unsound.
Describe expected output in plain prose instead, then run the code and react to
the real result next turn.

## How to write code

- Exactly one ```python-exec code block per response. The parser runs only the first;
  any additional blocks are silently dropped, and the runtime will warn you next
  turn. Never assume dropped blocks ran.
- Use print() to surface anything you'll want to refer to next turn.
- Each turn starts with a fresh namespace. Plain variables are discarded between turns.
- To persist across turns: `state["key"] = value` then `state["key"]` next turn.
  Only JSON-serializable values.
- For a structured (non-string) return: `return_value({{"status": "ok"}})` — ends the run
  and returns the value as-is. For a plain text answer, just stop using code blocks.
{tools_section}
## Rules

1. Only use variables you defined this turn, or values from `state`.
2. Use comments in code for reasoning if needed.
{tool_rule}
4. If you get an error, try a different approach.
5. To carry data across turns use `state['key'] = value`; bare names don't survive.
6. To finish, either respond with plain text, or call `return_value(value)` for structured output.

{instructions}

Now begin!"""
