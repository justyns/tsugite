"""Core agent implementation using LiteLLM directly.

A simpler, more direct implementation that gives us full control over
model parameters and reasoning model support.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from tsugite.attachments.base import Attachment, AttachmentContentType
from tsugite.events import (
    CodeExecutionEvent,
    ContentBlockEvent,
    CostSummaryEvent,
    ErrorEvent,
    EventBus,
    FinalAnswerEvent,
    LLMMessageEvent,
    ObservationEvent,
    ReasoningContentEvent,
    ReasoningTokensEvent,
    StepStartEvent,
    StreamChunkEvent,
    StreamCompleteEvent,
    TaskStartEvent,
    ToolCallEvent,
    ToolResultEvent,
    WarningEvent,
)
from tsugite.skill_discovery import Skill

from .content_blocks import extract_content_blocks
from .executor import LocalExecutor
from .memory import AgentMemory, StepResult
from .tools import Tool

# Agent execution constants
DEFAULT_MAX_TURNS = 10  # Default maximum reasoning iterations before timeout


def _attachment_char_limit(name: str) -> int | None:
    """Return max chars for a Claude Code attachment, or None for no limit.

    Currently returns None (no limit) for all attachments.
    Kept as a hook for future per-attachment size policies.
    """
    return None


def _trim_messages_to_token_budget(messages: List[Dict], budget_tokens: int) -> List[Dict]:
    """Keep the most recent messages that fit within a token budget.

    Walks from newest to oldest, estimating tokens as len(content) // 4.
    Returns messages in original order.
    """
    if not messages:
        return messages

    kept_indices = []
    used = 0
    for i in range(len(messages) - 1, -1, -1):
        content = messages[i].get("content", "")
        est_tokens = len(content) // 4 if isinstance(content, str) else 100
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


@dataclass
class ParsedResponse:
    """Result from parsing an LLM response."""

    thought: str
    code: str
    content_blocks: Dict[str, str] = field(default_factory=dict)


@dataclass
class TurnResult:
    """Result from a single agent turn (LLM call + parsing)."""

    thought: str
    code: str
    step_cost: float
    content_blocks: Dict[str, str] = field(default_factory=dict)
    response: Optional[Any] = None  # LiteLLM response object (not set for claude_code path)


@dataclass
class AgentResult:
    """Result from agent execution."""

    output: Any
    token_usage: Optional[int] = None
    cost: Optional[float] = None
    steps: Optional[List[StepResult]] = None
    error: Optional[str] = None
    claude_code_session_id: Optional[str] = None
    claude_code_compacted: bool = False
    context_window: Optional[int] = None

    def __str__(self) -> str:
        return self.output if self.output else self.error if self.error else ""


class TsugiteAgent:
    """Custom agent that uses Thought/Code/Observation loop.

    Provides direct access to LiteLLM features including reasoning models,
    custom parameters, and full control over the execution loop.

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
        executor: LocalExecutor = None,
        model_kwargs: dict = None,
        event_bus: EventBus = None,
        model_name: str = None,
        attachments: List[Attachment] = None,
        skills: List[Skill] = None,
        previous_messages: List[Dict] = None,
        resume_session: Optional[str] = None,
        resume_after_compaction: bool = False,
    ):
        """Initialize the agent.

        Args:
            model_string: Model identifier like "openai:gpt-4o-mini"
            tools: List of Tool objects the agent can use
            instructions: Additional instructions to append to system prompt
            max_turns: Maximum number of reasoning turns (think-act cycles) before giving up
            executor: Code executor (microsandbox or local). If None, uses LocalExecutor
            model_kwargs: Extra parameters for LiteLLM (reasoning_effort, response_format, etc.)
            event_bus: Optional EventBus for broadcasting events
            model_name: Optional display name for the model (for UI)
            attachments: List of Attachment objects for multi-modal inputs
            skills: List of Skill objects for loaded skills
            previous_messages: List of previous conversation messages (user/assistant pairs)
        """
        from tsugite.models import get_model_params

        self.model_string = model_string
        self.tools = tools
        self.instructions = instructions
        self.max_turns = max_turns
        self.executor = executor or LocalExecutor()
        self.memory = AgentMemory()
        self.event_bus = event_bus
        self.model_name = model_name or model_string
        self.attachments = attachments or []
        self.skills = skills or []
        self.previous_messages = previous_messages or []
        self._resume_session = resume_session
        self._resume_after_compaction = resume_after_compaction

        self.total_cost = 0.0
        self.total_tokens = 0
        self._previous_turn_had_error = False

        self.tool_map = {tool.name: tool for tool in tools}

        self._inject_tools_into_executor()

        self.litellm_params = get_model_params(model_string, **(model_kwargs or {}))

        # Detect claude_code provider
        self._is_claude_code = self.litellm_params.get("_provider") == "claude_code"
        self._claude_code_model = self.litellm_params.get("model") if self._is_claude_code else None
        self._claude_code_session_id: Optional[str] = None
        self._claude_code_last_turn_tokens: int = 0
        self._claude_code_context_tokens: int = 0
        self._claude_code_context_window: Optional[int] = None
        self._claude_code_cache_creation_tokens: int = 0
        self._claude_code_cache_read_tokens: int = 0
        self._claude_code_compacted: bool = False

    def _run_async_in_sync_context(self, coro):
        """Run async coroutine in synchronous context, handling event loop properly.

        This handles the case where we're already inside an async context (the agent's
        run method) but need to run tool coroutines synchronously from user code.
        """
        import concurrent.futures

        try:
            # Check if there's an existing running loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # We're inside an async context - run in a thread with its own event loop
            # to avoid issues with nest_asyncio and coroutine reuse.
            # Copy current context so contextvars (interaction backend, etc.) propagate.
            import contextvars

            ctx = contextvars.copy_context()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:

                def run_coro():
                    # Create a fresh event loop for this thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(coro)
                    finally:
                        new_loop.close()

                future = executor.submit(ctx.run, run_coro)
                return future.result()
        else:
            # No running loop - we can use asyncio.run directly
            return asyncio.run(coro)

    def _convert_positional_to_kwargs(self, tool_obj, args, kwargs):
        """Convert positional arguments to keyword arguments based on function signature."""
        if not args:
            return

        import inspect

        try:
            sig = inspect.signature(tool_obj.function)
            param_names = list(sig.parameters.keys())

            for i, arg in enumerate(args):
                if i < len(param_names):
                    param_name = param_names[i]
                    if param_name not in kwargs:
                        kwargs[param_name] = arg
                else:
                    raise TypeError(
                        f"Tool '{tool_obj.name}' takes at most {len(param_names)} "
                        f"positional arguments but {len(args)} were given"
                    )
        except Exception:
            raise TypeError(
                f"Tool '{tool_obj.name}' must be called with keyword arguments, "
                f"not positional arguments. "
                f"Example: {tool_obj.name}(param1=value1, param2=value2)"
            )

    def _inject_tools_into_executor(self):
        """Inject tools into executor namespace so they can be called from Python code.

        Creates wrapper functions for each tool that call the tool's execute() method.
        The LLM sees tools as Python functions and calls them directly in generated code.

        For SubprocessExecutor, tools are registered via set_tools() instead of
        namespace injection — the child process handles tool dispatch via IPC.

        Note: final_answer is registered as a tool for documentation purposes but is NOT
        injected here - the executor has its own built-in final_answer that properly
        signals completion. We skip it to avoid overriding the executor's version.
        """
        from .subprocess_executor import SubprocessExecutor

        if isinstance(self.executor, SubprocessExecutor):
            self.executor.set_tools(self.tools, event_bus=self.event_bus)
            return
        from tsugite.core.executor import EXECUTOR_BUILTIN_TOOLS

        tool_functions = {}

        for tool in self.tools:
            if tool.name in EXECUTOR_BUILTIN_TOOLS:
                continue

            def make_tool_wrapper(tool_obj):
                def tool_wrapper(*args, **kwargs):
                    if hasattr(self.executor, "_tools_called"):
                        self.executor._tools_called.append(tool_obj.name)

                    self._convert_positional_to_kwargs(tool_obj, args, kwargs)

                    # Emit audit events
                    if self.event_bus:
                        self.event_bus.emit(ToolCallEvent(tool_name=tool_obj.name, arguments=kwargs))

                    t0 = time.perf_counter()
                    try:
                        result = self._run_async_in_sync_context(tool_obj.execute(**kwargs))
                        duration_ms = int((time.perf_counter() - t0) * 1000)
                        if self.event_bus:
                            summary = str(result)[:200] if result is not None else ""
                            self.event_bus.emit(
                                ToolResultEvent(
                                    tool_name=tool_obj.name,
                                    success=True,
                                    result_summary=summary,
                                    duration_ms=duration_ms,
                                )
                            )
                        return result
                    except Exception as exc:
                        duration_ms = int((time.perf_counter() - t0) * 1000)
                        if self.event_bus:
                            self.event_bus.emit(
                                ToolResultEvent(
                                    tool_name=tool_obj.name,
                                    success=False,
                                    result_summary=str(exc)[:200],
                                    duration_ms=duration_ms,
                                )
                            )
                        raise

                tool_wrapper.__name__ = tool_obj.name
                tool_wrapper.__doc__ = tool_obj.description
                if hasattr(tool_obj.function, "__signature__"):
                    tool_wrapper.__signature__ = tool_obj.function.__signature__
                if hasattr(tool_obj.function, "__annotations__"):
                    tool_wrapper.__annotations__ = tool_obj.function.__annotations__

                return tool_wrapper

            tool_functions[tool.name] = make_tool_wrapper(tool)

        if hasattr(self.executor, "namespace"):
            self.executor.namespace.update(tool_functions)

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
        # Track execution time
        start_time = time.time()

        # Add task to memory
        self.memory.add_task(task)

        # Trigger task start event
        if self.event_bus:
            self.event_bus.emit(TaskStartEvent(task=task, model=self.model_name))

        # Start claude_code subprocess if needed
        claude_process = None
        if self._is_claude_code:
            from .claude_code import ClaudeCodeProcess

            claude_process = ClaudeCodeProcess()
            await claude_process.start(
                model=self._claude_code_model,
                system_prompt=self._build_system_prompt(),
                resume_session=self._resume_session,
            )
        # Main agent loop
        try:
            for turn_num in range(self.max_turns):
                # Trigger turn start event
                if self.event_bus:
                    self.event_bus.emit(
                        StepStartEvent(
                            step=turn_num + 1,
                            max_turns=self.max_turns,
                            recovering_from_error=self._previous_turn_had_error,
                        )
                    )

                # Build conversation messages from memory
                messages = self._build_messages()
                last_msg = messages[-1]["content"] if messages else ""
                logger.debug("Turn %d sending %d messages (last: %.200s)", turn_num + 1, len(messages), last_msg)

                if self._is_claude_code:
                    turn = await self._claude_code_turn(claude_process, messages, turn_num, stream)
                else:
                    turn = await self._litellm_turn(messages, turn_num, stream)

                thought, code, response = turn.thought, turn.code, turn.response
                logger.debug(
                    "Turn %d response (cost=%.4f): %.200s", turn_num + 1, turn.step_cost, (thought or "")[:200]
                )

                # Emit and inject content blocks into executor namespace
                if turn.content_blocks:
                    if self.event_bus:
                        for cb_name, cb_content in turn.content_blocks.items():
                            self.event_bus.emit(ContentBlockEvent(name=cb_name, content=cb_content))
                    await self.executor.inject_content_blocks(turn.content_blocks)

                # Only execute code if the LLM actually generated some
                if code and code.strip():
                    # Trigger code execution event
                    if self.event_bus:
                        self.event_bus.emit(CodeExecutionEvent(code=code))

                    # Execute the code and track duration
                    exec_start_time = time.perf_counter()
                    exec_result = await self.executor.execute(code)
                    exec_duration_ms = int((time.perf_counter() - exec_start_time) * 1000)

                    # Generate XML observation
                    xml_observation = exec_result.to_xml(duration_ms=exec_duration_ms)

                    # Trigger observation event
                    if self.event_bus:
                        observation = exec_result.output

                        if exec_result.error:
                            # Mark that this turn had an error (for recovery UX)
                            self._previous_turn_had_error = True

                            # Emit warning instead of error - less alarming and signals retry
                            error_preview = (
                                exec_result.error[:100] + "..." if len(exec_result.error) > 100 else exec_result.error
                            )
                            self.event_bus.emit(
                                WarningEvent(
                                    message=f"Tool failed, will retry: {error_preview}",
                                    step=turn_num + 1,
                                )
                            )
                        else:
                            # Success - reset error flag
                            self._previous_turn_had_error = False
                            self.event_bus.emit(ObservationEvent(observation=observation))
                else:
                    # No code to execute

                    # Claude Code subprocess may respond with plain text instead of a code block,
                    # especially when workspace attachments add conversational context that overrides
                    # the code-block instruction. Treat the text as a final answer rather than looping.
                    if self._is_claude_code:
                        answer = (thought or "").strip() or "(no response)"
                        self.memory.add_final_answer(answer)

                        total_tokens = (
                            self._claude_code_last_turn_tokens if self._claude_code_last_turn_tokens else None
                        )
                        if self.event_bus:
                            self.event_bus.emit(
                                FinalAnswerEvent(
                                    answer=answer,
                                    turns=turn_num + 1,
                                    tokens=total_tokens,
                                    cost=self.total_cost if self.total_cost > 0 else None,
                                )
                            )
                            duration = time.time() - start_time
                            cache_creation_tokens = self._claude_code_cache_creation_tokens or None
                            cache_read_tokens = self._claude_code_cache_read_tokens or None
                            self.event_bus.emit(
                                CostSummaryEvent(
                                    tokens=total_tokens,
                                    cost=self.total_cost if self.total_cost > 0 else None,
                                    model=self.model_name,
                                    duration_seconds=duration,
                                    cached_tokens=None,
                                    cache_creation_input_tokens=cache_creation_tokens,
                                    cache_read_input_tokens=cache_read_tokens,
                                )
                            )
                        return answer

                    from .executor import ExecutionResult

                    exec_result = ExecutionResult(output="", error=None, stdout="", stderr="")
                    xml_observation = None  # No XML for non-code responses

                    if self.event_bus:
                        self.event_bus.emit(
                            ErrorEvent(
                                error="LLM did not generate code. Expected format:\n\n```python\n<code>\n```",
                                error_type="Format Error",
                                step=turn_num + 1,
                            )
                        )

                    correction_msg = (
                        "Format Error: You must respond with a Python code block.\n\n"
                        "```python\n"
                        "# Your code here\n"
                        'final_answer("your answer")\n'
                        "```\n\n"
                        "Even for simple responses, wrap them in a code block with final_answer()."
                    )

                    from xml.sax.saxutils import escape

                    correction_xml = (
                        '<tsugite_execution_result status="error">\n'
                        "<output></output>\n"
                        f"<error>{escape(correction_msg)}</error>\n"
                        "</tsugite_execution_result>"
                    )
                    self.memory.add_step(
                        thought=thought if thought else "(No thought provided)",
                        code="",
                        output=correction_msg,
                        error=None,
                        tools_called=[],
                        xml_observation=correction_xml,
                    )

                    # Continue to next turn - the correction will be in the observation
                    continue

                # Build observation output, adding reminder if no final_answer
                step_output = exec_result.output
                if exec_result.final_answer is None and not exec_result.error:
                    step_output += (
                        "\n\n(Reminder: Call final_answer() when you have the result, "
                        "or ask_user() if you need input from the user.)"
                    )

                # Append turn/token budget tag so the LLM knows its resource limits
                budget_tag = self._build_budget_tag(turn_num)
                step_output += budget_tag
                xml_observation += budget_tag

                # Add this step to memory (only for successful executions or text mode)
                self.memory.add_step(
                    thought=thought,
                    code=code,
                    output=step_output,
                    error=exec_result.error,
                    tools_called=exec_result.tools_called,
                    loaded_skills=exec_result.loaded_skills,
                    xml_observation=xml_observation,
                    content_blocks=turn.content_blocks,
                )

                # Check if final_answer was called during execution
                if exec_result.final_answer is not None:
                    # Agent is done!
                    self.memory.add_final_answer(exec_result.final_answer)

                    # Trigger final answer event
                    if response and response.usage:
                        total_tokens = response.usage.total_tokens
                    elif self._is_claude_code and self._claude_code_last_turn_tokens:
                        total_tokens = self._claude_code_last_turn_tokens
                    else:
                        total_tokens = None

                    if self.event_bus:
                        self.event_bus.emit(
                            FinalAnswerEvent(
                                answer=str(exec_result.final_answer),
                                turns=turn_num + 1,
                                tokens=total_tokens,
                                cost=self.total_cost if self.total_cost > 0 else None,
                            )
                        )

                        duration = time.time() - start_time

                        # Extract cache-related fields (supported by OpenAI, Anthropic, Bedrock, Deepseek)
                        cached_tokens = None
                        cache_creation_tokens = None
                        cache_read_tokens = None
                        if response and response.usage:
                            cached_tokens = getattr(response.usage, "cached_tokens", None)
                            cache_creation_tokens = getattr(response.usage, "cache_creation_input_tokens", None)
                            cache_read_tokens = getattr(response.usage, "cache_read_input_tokens", None)
                        elif self._is_claude_code:
                            cache_creation_tokens = self._claude_code_cache_creation_tokens or None
                            cache_read_tokens = self._claude_code_cache_read_tokens or None

                        self.event_bus.emit(
                            CostSummaryEvent(
                                tokens=total_tokens,
                                cost=self.total_cost if self.total_cost > 0 else None,
                                model=self.model_name,
                                duration_seconds=duration,
                                cached_tokens=cached_tokens,
                                cache_creation_input_tokens=cache_creation_tokens,
                                cache_read_input_tokens=cache_read_tokens,
                            )
                        )

                    if claude_process:
                        self._claude_code_session_id = claude_process.session_id

                    if return_full_result:
                        context_tokens = self._claude_code_context_tokens or total_tokens
                        return AgentResult(
                            output=exec_result.final_answer,
                            token_usage=context_tokens,
                            cost=self.total_cost if self.total_cost > 0 else None,
                            steps=self.memory.steps,
                            claude_code_session_id=self._claude_code_session_id,
                            claude_code_compacted=self._claude_code_compacted,
                            context_window=self._claude_code_context_window,
                        )
                    return exec_result.final_answer

                # Continue loop (LLM will see the observation in next iteration)

            # If we get here, we hit max_turns — give the LLM one last chance to call final_answer()
            last_chance_msg = (
                f"TURN LIMIT REACHED. You have used all {self.max_turns} turns.\n"
                "You MUST call final_answer() NOW with a summary of your progress so far.\n"
                "If the task is not complete, include what remains to be done so the user can ask to continue.\n"
                "Do NOT call any other tools — only final_answer() is available."
            )
            self.memory.add_step(
                thought="(system: turn limit reached)",
                code="",
                output=last_chance_msg,
                error=None,
                tools_called=[],
            )

            saved_tools = self.tools
            self.tools = []
            try:
                messages = self._build_messages()
                if self._is_claude_code:
                    turn = await self._claude_code_turn(claude_process, messages, self.max_turns, stream)
                else:
                    turn = await self._litellm_turn(messages, self.max_turns, stream)

                if turn.code and turn.code.strip():
                    exec_result = await self.executor.execute(turn.code)
                    if exec_result.final_answer is not None:
                        self.memory.add_final_answer(exec_result.final_answer)
                        if claude_process:
                            self._claude_code_session_id = claude_process.session_id
                        if return_full_result:
                            return AgentResult(
                                output=exec_result.final_answer,
                                token_usage=None,
                                cost=self.total_cost if self.total_cost > 0 else None,
                                steps=self.memory.steps,
                                claude_code_session_id=self._claude_code_session_id,
                                context_window=self._claude_code_context_window,
                            )
                        return exec_result.final_answer
            except Exception:
                logger.debug("Last-chance turn failed", exc_info=True)
            finally:
                self.tools = saved_tools

            error_msg = f"Agent reached max_turns ({self.max_turns}) without completing task"
            if self.event_bus:
                self.event_bus.emit(ErrorEvent(error=error_msg, error_type="RuntimeError"))

            partial_output = None
            if self.memory.steps:
                last_step = self.memory.steps[-1]
                partial_output = last_step.thought or last_step.output

            if return_full_result:
                return AgentResult(
                    output=partial_output,
                    token_usage=None,
                    cost=self.total_cost,
                    steps=self.memory.steps,
                    error=error_msg,
                    claude_code_session_id=self._claude_code_session_id,
                    context_window=self._claude_code_context_window,
                )

            raise RuntimeError(error_msg)
        finally:
            if claude_process:
                self._claude_code_session_id = claude_process.session_id
                self._claude_code_compacted = claude_process.compacted
                await claude_process.stop()

    async def _claude_code_turn(self, process, messages, turn_num, stream) -> TurnResult:
        """Execute one turn via Claude Code subprocess."""
        # Build user content for this turn
        if turn_num == 0:
            # First turn: include previous conversation context (for --continue)
            user_content = self._build_claude_code_first_message()
        else:
            # Subsequent turns: subprocess has context, just send the new observation
            user_content = messages[-1]["content"] if messages else ""

        logger.info(
            "Claude Code turn %d: sending ~%d chars (~%d est tokens)",
            turn_num + 1, len(user_content), len(user_content) // 4,
        )

        accumulated = ""
        step_cost = 0.0

        async for event in process.send_message(user_content):
            if event["type"] == "text_delta":
                accumulated += event["text"]
                if stream and self.event_bus:
                    self.event_bus.emit(StreamChunkEvent(chunk=event["text"]))
            elif event["type"] == "result":
                if not accumulated:
                    accumulated = event.get("text", "")
                step_cost = event.get("cost_usd") or 0.0
                self.total_cost += step_cost
                self._claude_code_session_id = event.get("session_id", self._claude_code_session_id)
                input_tokens = event.get("input_tokens") or 0
                cache_creation = event.get("cache_creation_input_tokens") or 0
                cache_read = event.get("cache_read_input_tokens") or 0
                output_tokens = event.get("output_tokens") or 0
                turn_total = input_tokens + cache_creation + cache_read + output_tokens
                self._claude_code_last_turn_tokens = turn_total
                self._claude_code_context_tokens = input_tokens + cache_creation + cache_read
                self._claude_code_cache_creation_tokens += cache_creation
                self._claude_code_cache_read_tokens += cache_read
                self.total_tokens += turn_total
                if event.get("context_window"):
                    self._claude_code_context_window = event["context_window"]

        if stream and self.event_bus:
            self.event_bus.emit(StreamCompleteEvent())

        if accumulated.strip().lower() == "prompt is too long":
            raise RuntimeError(f"Claude Code prompt too long (session={self._claude_code_session_id})")

        parsed = self._parse_response_from_text(accumulated)

        # If no code block was found, preserve the raw text as thought so the
        # caller can use it (e.g. treat plain-text responses as final answers).
        thought = parsed.thought if parsed.thought or parsed.code else accumulated.strip()

        if self.event_bus and not stream and thought.strip():
            self.event_bus.emit(
                LLMMessageEvent(content=thought, title=f"Turn {turn_num + 1} Reasoning", step=turn_num + 1)
            )

        return TurnResult(
            thought=thought,
            code=parsed.code,
            step_cost=step_cost,
            content_blocks=parsed.content_blocks,
        )

    async def _litellm_turn(self, messages, turn_num, stream) -> TurnResult:
        """Execute one turn via LiteLLM."""
        import litellm
        import sniffio

        sniffio.current_async_library_cvar.set("asyncio")

        if stream:
            accumulated_content = ""
            response = None
            stream_params = {**self.litellm_params, "stream": True}
            stream_response = await litellm.acompletion(messages=messages, **stream_params)

            async for chunk in stream_response:
                if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        chunk_text = delta.content
                        accumulated_content += chunk_text
                        if self.event_bus:
                            self.event_bus.emit(StreamChunkEvent(chunk=chunk_text))
                response = chunk

            if self.event_bus:
                self.event_bus.emit(StreamCompleteEvent())

            parsed = self._parse_response_from_text(accumulated_content)
        else:
            response = await litellm.acompletion(messages=messages, **self.litellm_params)
            parsed = self._parse_response(response)

        # Track cost
        step_cost = 0.0
        if hasattr(response, "_hidden_params") and "response_cost" in response._hidden_params:
            step_cost = response._hidden_params["response_cost"]
        if step_cost is not None:
            self.total_cost += step_cost

        # Track cumulative tokens
        if response and response.usage:
            self.total_tokens += getattr(response.usage, "total_tokens", 0) or 0

        # Extract reasoning content
        reasoning_content = self._extract_reasoning_content(response)
        if reasoning_content:
            self.memory.add_reasoning(reasoning_content)
            if self.event_bus:
                self.event_bus.emit(ReasoningContentEvent(content=reasoning_content, step=turn_num + 1))

        if response.usage and hasattr(response.usage, "completion_tokens_details"):
            details = response.usage.completion_tokens_details
            if hasattr(details, "reasoning_tokens") and details.reasoning_tokens:
                if self.event_bus:
                    self.event_bus.emit(ReasoningTokensEvent(tokens=details.reasoning_tokens, step=turn_num + 1))

        if self.event_bus and not stream:
            display_content = (
                parsed.thought if parsed.thought else (response.choices[0].message.content if response.choices else "")
            )
            if display_content and display_content.strip():
                self.event_bus.emit(
                    LLMMessageEvent(content=display_content, title=f"Turn {turn_num + 1} Reasoning", step=turn_num + 1)
                )

        return TurnResult(
            thought=parsed.thought,
            code=parsed.code,
            step_cost=step_cost,
            content_blocks=parsed.content_blocks,
            response=response,
        )

    def _format_attachment(self, attachment: Attachment) -> Optional[Dict]:
        """Format an attachment for LiteLLM based on its content type.

        Args:
            attachment: Attachment object to format

        Returns:
            Formatted content block for LiteLLM, or None if invalid
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
                # URL reference - let LiteLLM fetch it
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
                # URL reference - let LiteLLM fetch it
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
                # URL reference - let LiteLLM fetch it
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

        # Text attachments wrapped in XML
        for att in self.attachments:
            if att.content_type == AttachmentContentType.TEXT:
                text_parts.append(f'<attachment name="{att.name}">')
                text_parts.append(att.content)
                text_parts.append("</attachment>")
            else:
                # Native multi-modal block (image_url, file, etc.)
                block = self._format_attachment(att)
                if block:
                    blocks.append(block)

        # Auto-loaded skills wrapped in XML
        for skill in self.skills:
            text_parts.append(f'<skill name="{skill.name}">')
            text_parts.append(skill.content)
            text_parts.append("</skill>")

        text_parts.append("</context>")

        return [{"type": "text", "text": "\n".join(text_parts)}] + blocks

    def _build_observation(self, step) -> str:
        """Build observation with dynamically-loaded skills embedded.

        Skills loaded mid-conversation via load_skill() are embedded in the
        observation of the turn where they were loaded, keeping them visible
        to the LLM without modifying system messages.

        Args:
            step: StepResult with execution output and loaded_skills

        Returns:
            Observation string with embedded skills
        """
        parts = []

        # Embed skills loaded during this step
        if step.loaded_skills:
            for name, content in step.loaded_skills.items():
                parts.append(f'<loaded_skill name="{name}">')
                parts.append(content)
                parts.append("</loaded_skill>")

        # Regular XML observation
        parts.append(step.xml_observation)

        return "\n".join(parts)

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
            {"role": "assistant", "content": "```python\\n...```"},
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
            messages.append({"role": "assistant", "content": "Context loaded.", "cache_control": {"type": "ephemeral"}})

        # 3. Previous conversation messages (if continuing a conversation)
        if self.previous_messages:
            messages.extend(self.previous_messages)

        # 4. Task
        messages.append({"role": "user", "content": self.memory.task})

        # 5. Previous steps (Code → Observation with embedded skills)
        for step in self.memory.steps:
            # Assistant message is code + any content blocks the LLM defined
            if step.code and step.code.strip():
                assistant_msg = f"```python\n{step.code}\n```"
            else:
                # No code — show what the LLM actually said so it sees its own response
                assistant_msg = step.thought if step.thought else "(empty response)"
            from tsugite.core.content_blocks import serialize_content_blocks
            cb_str = serialize_content_blocks(step.content_blocks)
            if cb_str:
                assistant_msg += f"\n\n{cb_str}"
            messages.append({"role": "assistant", "content": assistant_msg})

            # Observation with loaded skills embedded
            messages.append({"role": "user", "content": self._build_observation(step)})

        return messages

    def _build_claude_code_first_message(self) -> str:
        """Build first message for claude_code subprocess, including context and history.

        Claude Code receives the system prompt via --system-prompt-file, but attachments
        and skills are normally sent as a context turn in _build_messages(). Since that
        turn is skipped for the claude_code path, we inline attachment/skill content here.

        When resuming a Claude Code session, history is skipped (Claude Code already has
        it). Attachments are included on fresh sessions and after Claude Code compaction
        (which may have lost them), but skipped on regular resumes to avoid accumulation.
        """
        parts = []

        # Include attachments on fresh sessions or after compaction (may have been lost)
        include_context = not self._resume_session or self._resume_after_compaction

        if include_context:
            context_parts = []
            for att in self.attachments:
                if att.content_type == AttachmentContentType.TEXT:
                    limit = _attachment_char_limit(att.name)
                    if limit == 0:
                        continue
                    content = att.content
                    if limit is not None and len(content) > limit:
                        content = content[:limit] + "\n... (truncated)"
                    context_parts.append(f'<attachment name="{att.name}">')
                    context_parts.append(content)
                    context_parts.append("</attachment>")
            for skill in self.skills:
                content = skill.content
                if len(content) > 4000:
                    content = content[:4000] + "\n... (truncated)"
                context_parts.append(f'<skill name="{skill.name}">')
                context_parts.append(content)
                context_parts.append("</skill>")
            if context_parts:
                parts.append("<context>\n" + "\n".join(context_parts) + "\n</context>\n")

        # Only serialize history for fresh sessions. Resumed sessions already have
        # the conversation in Claude Code's internal state.
        if self.previous_messages and not self._resume_session:
            trimmed = _trim_messages_to_token_budget(
                self.previous_messages,
                self._get_claude_code_history_budget(),
            )
            dropped = len(self.previous_messages) - len(trimmed)
            history_lines = [
                f"{msg.get('role', 'unknown').capitalize()}: {msg.get('content', '')}" for msg in trimmed
            ]
            header = "<conversation_history"
            if dropped > 0:
                header += f' note="{dropped} older messages omitted for context"'
                logger.info("Trimmed %d of %d previous messages to fit history budget", dropped, len(self.previous_messages))
            header += ">"
            parts.append(header + "\n" + "\n\n".join(history_lines) + "\n</conversation_history>\n")

        # Add the current task
        parts.append(self.memory.task)
        return "\n".join(parts)

    def _build_system_prompt(self) -> str:
        """Build system prompt that teaches LLM how to solve tasks."""
        return build_system_prompt(self.tools, self.instructions)

    def _get_claude_code_history_budget(self) -> int:
        """Get token budget for conversation history in claude_code first message.

        Reserves 50% of context for: Claude Code's own system prompt, tsugite's
        system prompt, attachments/skills, current task, and multi-step headroom.
        """
        from tsugite.daemon.memory import _CLAUDE_CODE_CONTEXT_LIMITS

        if self._claude_code_context_window:
            context_limit = self._claude_code_context_window
        else:
            litellm_model = self.litellm_params.get("_litellm_model", self._claude_code_model or "")
            context_limit = _CLAUDE_CODE_CONTEXT_LIMITS.get(litellm_model, 200_000)

        return context_limit // 2

    def _build_budget_tag(self, turn_num: int) -> str:
        """Build XML budget tag showing turn and token usage for the LLM."""
        turn = turn_num + 1
        parts = [f'turn="{turn}"', f'max_turns="{self.max_turns}"']
        if self.total_tokens > 0:
            parts.append(f'tokens_used="{self.total_tokens}"')
        if self.max_turns - turn <= 2:
            parts.append('warning="approaching turn limit, wrap up soon"')
        return f"\n<tsugite_budget {' '.join(parts)} />"

    def _parse_response(self, response) -> ParsedResponse:
        """Parse LLM response into thought, code, and content blocks."""
        if not response.choices:
            return ParsedResponse(thought="", code="", content_blocks=[])
        content = response.choices[0].message.content or ""
        return self._parse_response_from_text(content)

    def _parse_response_from_text(self, content: str) -> ParsedResponse:
        """Parse text content into thought, code, and content blocks."""
        cleaned, content_blocks = extract_content_blocks(content)

        thought = ""
        code = ""

        # Extract thought (everything before code block)
        thought_start = cleaned.find("Thought:")
        if thought_start != -1:
            thought_start += len("Thought:")
            code_block_start = cleaned.find("```python", thought_start)
            if code_block_start != -1:
                thought = cleaned[thought_start:code_block_start].strip()
            else:
                thought = cleaned[thought_start:].strip()

        # Extract code block
        code_block_start = cleaned.find("```python")
        if code_block_start != -1:
            code_start = code_block_start + len("```python")
            code_end = cleaned.find("```", code_start)
            if code_end != -1:
                code = cleaned[code_start:code_end].strip()

        return ParsedResponse(thought=thought, code=code, content_blocks=content_blocks)

    def _extract_reasoning_content(self, response) -> Optional[str]:
        """Extract reasoning content from response (for o1/o3/Claude thinking).

        Returns:
            str: Reasoning content if present, None otherwise
        """
        try:
            if hasattr(response, "choices") and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice.message, "reasoning_content"):
                    return choice.message.reasoning_content
        except (AttributeError, IndexError):
            pass

        return None


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

Write Python code in a code block. Use comments for reasoning if needed:

```python
# First, read the config to understand the structure
config = read_file("config.yaml")
print(config)  # See the contents
```

## Current Working Directory

{cwd}

## Execution Results

After code runs, you'll see structured XML:

```xml
<tsugite_execution_result status="success" duration_ms="142">
<output>file contents here...</output>
<variables_set>config=str(1234 chars)</variables_set>
</tsugite_execution_result>
```

Fields:
- `status`: "success" or "error"
- `<output>`: Your print() output
- `<error>`: Error message if failed
- `<traceback>`: Python traceback if failed (last 10 lines)
- `<variables_set>`: New variables you created
- `<final_answer>`: Confirms completion when you call final_answer()

## How to write code:

- Write code in triple-backtick code blocks: ```python
- Use print() to output important information
- Variables persist between code blocks
- When you have the final answer, call: final_answer(your_answer)
{tools_section}
## Rules:

1. Only use variables you've defined
2. Use comments in code for reasoning if needed
{tool_rule}
4. If you get an error, try a different approach
5. State persists - variables remain available across code blocks
6. **To complete your turn, you MUST call one of:**
   - `final_answer(result)` - when you have the answer
   - `ask_user(question)` - when you need input from the user

{instructions}

Now begin!"""
