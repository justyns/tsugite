"""Core agent implementation using LiteLLM directly.

A simpler, more direct implementation that gives us full control over
model parameters and reasoning model support.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from tsugite.attachments.base import Attachment, AttachmentContentType
from tsugite.events import (
    CodeExecutionEvent,
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

from .executor import LocalExecutor
from .memory import AgentMemory, StepResult
from .tools import Tool

# Agent execution constants
DEFAULT_MAX_TURNS = 10  # Default maximum reasoning iterations before timeout


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
class AgentResult:
    """Result from agent execution."""

    output: Any
    token_usage: Optional[int] = None
    cost: Optional[float] = None
    steps: Optional[List[StepResult]] = None
    error: Optional[str] = None
    claude_code_session_id: Optional[str] = None


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
        resume_session: str = None,
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
            resume_session: Claude Code session ID to resume (only used with claude_code provider)
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
        tool_functions = {}

        for tool in self.tools:
            # Skip built-in functions - executor has its own versions
            # final_answer: handles completion signaling
            # send_message: needs event_bus access for progress updates
            if tool.name in ("final_answer", "send_message"):
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
                                    tool_name=tool_obj.name, success=True, result_summary=summary, duration_ms=duration_ms
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

            response = None
            if self._is_claude_code:
                # Claude Code subprocess path
                thought, code, step_cost = await self._claude_code_turn(
                    claude_process, messages, turn_num, stream
                )
            else:
                # LiteLLM path
                thought, code, step_cost, response = await self._litellm_turn(
                    messages, turn_num, stream
                )

            logger.debug("Turn %d response (cost=%.4f): %.200s", turn_num + 1, step_cost, (thought or "")[:200])

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
                # No code to execute - create a dummy result
                from .executor import ExecutionResult

                exec_result = ExecutionResult(output="", error=None, stdout="", stderr="")
                xml_observation = None  # No XML for non-code responses

                # Code is required - show a warning that the LLM didn't generate code
                if self.event_bus:
                    self.event_bus.emit(
                        ErrorEvent(
                            error="LLM did not generate code. Expected format:\n\nThought: <explanation>\n```python\n<code>\n```",
                            error_type="Format Error",
                            step=turn_num + 1,
                        )
                    )

                # Add a correction to memory to guide the LLM
                correction_msg = (
                    "Format Error: You must provide your response in a Python code block.\n\n"
                    "Use this format:\n\n"
                    "Thought: <your explanation>\n"
                    "```python\n"
                    "# Your code here\n"
                    'final_answer("your answer")\n'
                    "```\n\n"
                    "Remember to call final_answer() with your result."
                )

                # Add the thought and correction as a step
                # This will show the LLM what it did wrong and how to fix it
                # Use XML format for the correction message
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
            step_output += self._build_budget_tag(turn_num)

            # Add this step to memory (only for successful executions or text mode)
            self.memory.add_step(
                thought=thought,
                code=code,
                output=step_output,
                error=exec_result.error,
                tools_called=exec_result.tools_called,
                loaded_skills=exec_result.loaded_skills,
                xml_observation=xml_observation,
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

                # Stop claude_code process on completion
                if claude_process:
                    self._claude_code_session_id = claude_process.session_id
                    await claude_process.stop()

                if return_full_result:
                    return AgentResult(
                        output=exec_result.final_answer,
                        token_usage=total_tokens,
                        cost=self.total_cost if self.total_cost > 0 else None,
                        steps=self.memory.steps,
                        claude_code_session_id=self._claude_code_session_id,
                    )
                return exec_result.final_answer

            # Continue loop (LLM will see the observation in next iteration)

        # If we get here, we hit max_turns
        if claude_process:
            self._claude_code_session_id = claude_process.session_id
            await claude_process.stop()

        error_msg = f"Agent reached max_turns ({self.max_turns}) without completing task"
        if self.event_bus:
            self.event_bus.emit(ErrorEvent(error=error_msg, error_type="RuntimeError"))

        if return_full_result:
            return AgentResult(
                output=None,
                token_usage=None,
                cost=self.total_cost,
                steps=self.memory.steps,
                error=error_msg,
                claude_code_session_id=self._claude_code_session_id,
            )

        raise RuntimeError(error_msg)

    async def _claude_code_turn(self, process, messages, turn_num, stream) -> tuple:
        """Execute one turn via Claude Code subprocess.

        Returns:
            (thought, code, step_cost)
        """
        # Build user content for this turn
        if turn_num == 0:
            # First turn: include previous conversation context (for --continue)
            user_content = self._build_claude_code_first_message()
        else:
            # Subsequent turns: subprocess has context, just send the new observation
            user_content = messages[-1]["content"] if messages else ""

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
                output_tokens = event.get("output_tokens") or 0
                self._claude_code_last_turn_tokens = input_tokens + output_tokens
                self.total_tokens += input_tokens + output_tokens

        if stream and self.event_bus:
            self.event_bus.emit(StreamCompleteEvent())

        thought, code, _ = self._parse_response_from_text(accumulated)

        if self.event_bus and not stream and thought.strip():
            self.event_bus.emit(
                LLMMessageEvent(content=thought, title=f"Turn {turn_num + 1} Reasoning", step=turn_num + 1)
            )

        return thought, code, step_cost

    async def _litellm_turn(self, messages, turn_num, stream) -> tuple:
        """Execute one turn via LiteLLM.

        Returns:
            (thought, code, step_cost, response)
        """
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

            thought, code, _ = self._parse_response_from_text(accumulated_content)
        else:
            response = await litellm.acompletion(messages=messages, **self.litellm_params)
            thought, code, _ = self._parse_response(response)

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
            display_content = thought if thought else response.choices[0].message.content
            if display_content and display_content.strip():
                self.event_bus.emit(
                    LLMMessageEvent(content=display_content, title=f"Turn {turn_num + 1} Reasoning", step=turn_num + 1)
                )

        return thought, code, step_cost, response

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
            # Assistant message is just code (comments can provide reasoning)
            assistant_msg = f"```python\n{step.code}\n```"
            messages.append({"role": "assistant", "content": assistant_msg})

            # Observation with loaded skills embedded
            messages.append({"role": "user", "content": self._build_observation(step)})

        return messages

    def _build_claude_code_first_message(self) -> str:
        """Build first message for claude_code subprocess, including conversation history.

        For --continue runs, includes previous conversation as context so the
        subprocess can recall earlier messages without relying on --resume.
        """
        parts = []

        # Include previous conversation history if continuing
        if self.previous_messages:
            history_lines = [
                f"{msg.get('role', 'unknown').capitalize()}: {msg.get('content', '')}"
                for msg in self.previous_messages
            ]
            parts.append("<conversation_history>\n" + "\n\n".join(history_lines) + "\n</conversation_history>\n")

        # Add the current task
        parts.append(self.memory.task)
        return "\n".join(parts)

    def _build_system_prompt(self) -> str:
        """Build system prompt that teaches LLM how to solve tasks."""
        return build_system_prompt(self.tools, self.instructions)

    def _build_budget_tag(self, turn_num: int) -> str:
        """Build XML budget tag showing turn and token usage for the LLM."""
        turn = turn_num + 1
        parts = [f'turn="{turn}"', f'max_turns="{self.max_turns}"']
        if self.total_tokens > 0:
            parts.append(f'tokens_used="{self.total_tokens}"')
        if self.max_turns - turn <= 1:
            parts.append('warning="approaching turn limit, wrap up soon"')
        return f'\n<tsugite_budget {" ".join(parts)} />'

    def _parse_response(self, response) -> tuple[str, str, Optional[str]]:
        """Parse LLM response into thought, code, and final_answer.

        Returns:
            (thought, code, final_answer)
        """
        content = response.choices[0].message.content
        return self._parse_response_from_text(content)

    def _parse_response_from_text(self, content: str) -> tuple[str, str, Optional[str]]:
        """Parse text content into thought, code, and final_answer.

        Args:
            content: The text content to parse

        Returns:
            (thought, code, final_answer)
        """
        thought = ""
        code = ""

        # Extract thought (everything before code block)
        thought_start = content.find("Thought:")
        if thought_start != -1:
            thought_start += len("Thought:")
            code_block_start = content.find("```python", thought_start)
            if code_block_start != -1:
                thought = content[thought_start:code_block_start].strip()
            else:
                thought = content[thought_start:].strip()

        # Extract code block
        code_block_start = content.find("```python")
        if code_block_start != -1:
            code_start = code_block_start + len("```python")
            code_end = content.find("```", code_start)
            if code_end != -1:
                code = content[code_start:code_end].strip()

        return thought, code, None

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
