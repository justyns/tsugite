"""Core agent implementation using LiteLLM directly.

A simpler, more direct implementation that gives us full control over
model parameters and reasoning model support.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import litellm

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
)

from .executor import CodeExecutor, LocalExecutor
from .memory import AgentMemory, StepResult
from .tools import Tool


def build_system_prompt(tools: List[Tool], instructions: str = "", text_mode: bool = False) -> str:
    """Build system prompt for LLM with tools and instructions.

    This is shared between TsugiteAgent and the render command to ensure
    consistency between what's shown and what's sent to the LLM.

    Args:
        tools: List of Tool objects available to the agent
        instructions: Additional instructions from agent config
        text_mode: If True, use text mode (code blocks optional)

    Returns:
        Complete system prompt string
    """
    # Build tools section (only if tools exist)
    tools_section = build_tools_section(tools)
    has_tools = bool(tools)

    # Build mode-specific prompt
    if text_mode:
        return build_text_mode_prompt(tools_section, instructions, has_tools)
    else:
        return build_standard_mode_prompt(tools_section, instructions, has_tools)


@dataclass
class AgentResult:
    """Result from agent execution."""

    output: Any
    token_usage: Optional[int] = None
    cost: Optional[float] = None
    steps: Optional[List[StepResult]] = None
    error: Optional[str] = None


class TsugiteAgent:
    """Custom agent that uses Thought/Code/Observation loop.

    Provides direct access to LiteLLM features including reasoning models,
    custom parameters, and full control over the execution loop.

    How it works:
    1. Build system prompt from tools + instructions
    2. Loop: Send messages → Get response → Execute code → Repeat
    3. Stop when final_answer() is called or max_turns reached

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
        max_turns: int = 10,
        executor: CodeExecutor = None,
        model_kwargs: dict = None,
        event_bus: EventBus = None,
        model_name: str = None,
        text_mode: bool = False,
        attachments: List[tuple[str, str]] = None,
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
            text_mode: Allow text-only responses (code blocks optional)
            attachments: List of (name, content) tuples for prompt caching
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
        self.text_mode = text_mode
        self.attachments = attachments or []

        # Track cumulative cost across all steps
        self.total_cost = 0.0

        # Build tool map for quick lookup
        self.tool_map = {tool.name: tool for tool in tools}

        # Inject tools into executor namespace so they can be called from code
        self._inject_tools_into_executor()

        # Pre-compute LiteLLM params (filters unsupported params for reasoning models)
        # This is more efficient and correct - we filter once, not on every loop iteration
        self.litellm_params = get_model_params(model_string, **(model_kwargs or {}))

    def _inject_tools_into_executor(self):
        """Inject tools into executor namespace so they can be called from Python code.

        Creates wrapper functions for each tool that call the tool's execute() method.
        The LLM sees tools as Python functions and calls them directly in generated code.
        """
        import asyncio

        tool_functions = {}

        for tool in self.tools:
            # Create a wrapper function that calls the tool's execute method
            def make_tool_wrapper(tool_obj):
                """Create a wrapper for this specific tool."""

                def tool_wrapper(*args, **kwargs):
                    """Synchronous wrapper that calls async tool.execute().

                    Accepts both positional and keyword arguments for flexibility,
                    but tool.execute() expects keyword arguments only.
                    """
                    # Record that this tool was called
                    if hasattr(self.executor, "_tools_called"):
                        self.executor._tools_called.append(tool_obj.name)

                    # Convert positional args to keyword args if needed
                    if args:
                        # Map positional arguments to parameter names using function signature
                        import inspect

                        try:
                            sig = inspect.signature(tool_obj.function)
                            param_names = list(sig.parameters.keys())

                            # Convert positional args to kwargs
                            for i, arg in enumerate(args):
                                if i < len(param_names):
                                    param_name = param_names[i]
                                    # Don't override if already provided as kwarg
                                    if param_name not in kwargs:
                                        kwargs[param_name] = arg
                                else:
                                    # Too many positional args
                                    raise TypeError(
                                        f"Tool '{tool_obj.name}' takes at most {len(param_names)} "
                                        f"positional arguments but {len(args)} were given"
                                    )
                        except Exception:
                            # If signature inspection fails, fall back to error
                            raise TypeError(
                                f"Tool '{tool_obj.name}' must be called with keyword arguments, "
                                f"not positional arguments. "
                                f"Example: {tool_obj.name}(param1=value1, param2=value2)"
                            )

                    # Get the running event loop, or create one if needed
                    try:
                        loop = asyncio.get_running_loop()
                        # Already in async context - use thread to block on async result
                        import concurrent.futures
                        import contextvars

                        # Capture current context to propagate to executor thread
                        ctx = contextvars.copy_context()

                        with concurrent.futures.ThreadPoolExecutor() as executor:

                            def run_async():
                                new_loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(new_loop)
                                try:
                                    return new_loop.run_until_complete(tool_obj.execute(**kwargs))
                                finally:
                                    # Clean up any pending tasks before closing the loop
                                    # to prevent RuntimeWarning about tasks being destroyed
                                    pending = asyncio.all_tasks(new_loop)
                                    for task in pending:
                                        task.cancel()
                                    if pending:
                                        new_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                                    new_loop.close()

                            # Run in copied context to propagate UI context to thread
                            result = executor.submit(ctx.run, run_async).result()
                    except RuntimeError:
                        # No running event loop - create one and run synchronously
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            result = loop.run_until_complete(tool_obj.execute(**kwargs))
                        finally:
                            # Clean up any pending tasks before closing the loop
                            # to prevent RuntimeWarning about tasks being destroyed
                            pending = asyncio.all_tasks(loop)
                            for task in pending:
                                task.cancel()
                            if pending:
                                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                            loop.close()

                    return result

                # Copy the tool's function signature and metadata to the wrapper
                # This ensures the wrapper appears identical to the original function
                tool_wrapper.__name__ = tool_obj.name
                tool_wrapper.__doc__ = tool_obj.description
                if hasattr(tool_obj.function, "__signature__"):
                    tool_wrapper.__signature__ = tool_obj.function.__signature__
                if hasattr(tool_obj.function, "__annotations__"):
                    tool_wrapper.__annotations__ = tool_obj.function.__annotations__

                return tool_wrapper

            tool_functions[tool.name] = make_tool_wrapper(tool)

        # Inject all tool functions into executor namespace
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

        # Main agent loop
        for turn_num in range(self.max_turns):
            # Trigger turn start event
            if self.event_bus:
                self.event_bus.emit(StepStartEvent(step=turn_num + 1, max_turns=self.max_turns))

            # Build conversation messages from memory
            messages = self._build_messages()

            # Call LiteLLM directly with pre-computed params
            # Parameters are filtered for reasoning models (o1/o3/Claude)
            if stream:
                # Streaming mode: accumulate chunks and emit events
                accumulated_content = ""
                response = None

                # Add stream parameter to litellm params
                stream_params = {**self.litellm_params, "stream": True}

                # Get the streaming response generator
                stream_response = await litellm.acompletion(messages=messages, **stream_params)

                async for chunk in stream_response:
                    # Extract content from chunk
                    if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, "content") and delta.content:
                            chunk_text = delta.content
                            accumulated_content += chunk_text

                            # Emit stream chunk event
                            if self.event_bus:
                                self.event_bus.emit(StreamChunkEvent(chunk=chunk_text))

                    # Save the last chunk as response for usage/cost tracking
                    response = chunk

                # Emit stream complete event
                if self.event_bus:
                    self.event_bus.emit(StreamCompleteEvent())

                # Parse accumulated content
                thought, code, _ = self._parse_response_from_text(accumulated_content)
            else:
                # Non-streaming mode: get complete response
                response = await litellm.acompletion(messages=messages, **self.litellm_params)

                # Parse LLM response
                # Response should contain: Thought + Code OR final_answer()
                thought, code, _ = self._parse_response(response)

            # Track cost from this response
            step_cost = 0.0
            if hasattr(response, "_hidden_params") and "response_cost" in response._hidden_params:
                step_cost = response._hidden_params["response_cost"]
                if step_cost is not None:
                    self.total_cost += step_cost

            # Extract reasoning content if present (for o1/o3/Claude thinking)
            reasoning_content = self._extract_reasoning_content(response)
            if reasoning_content:
                self.memory.add_reasoning(reasoning_content)
                # Trigger reasoning content event
                if self.event_bus:
                    self.event_bus.emit(ReasoningContentEvent(content=reasoning_content, step=turn_num + 1))

            # Check for reasoning tokens (o1/o3 models)
            if response.usage and hasattr(response.usage, "completion_tokens_details"):
                details = response.usage.completion_tokens_details
                if hasattr(details, "reasoning_tokens") and details.reasoning_tokens:
                    if self.event_bus:
                        self.event_bus.emit(ReasoningTokensEvent(tokens=details.reasoning_tokens, step=turn_num + 1))

            # Show LLM's thought/reasoning (always show what the LLM is saying)
            # Skip this if streaming (already shown via STREAM_CHUNK events)
            # Skip if text mode with no code (thought will be shown as final answer)
            if self.event_bus and not stream:
                # If we parsed a thought, show it. Otherwise show the raw response
                # (this helps debug when LLM doesn't follow the expected format)
                display_content = thought if thought else response.choices[0].message.content

                # In text mode, if there's a thought but no code, skip showing the thought here
                # because it will be shown as the final answer (to avoid duplication)
                skip_llm_message = self.text_mode and thought and not (code and code.strip())

                if display_content and display_content.strip() and not skip_llm_message:
                    self.event_bus.emit(
                        LLMMessageEvent(
                            content=display_content, title=f"Turn {turn_num + 1} Reasoning", step=turn_num + 1
                        )
                    )

            # Only execute code if the LLM actually generated some
            if code and code.strip():
                # Trigger code execution event
                if self.event_bus:
                    self.event_bus.emit(CodeExecutionEvent(code=code))

                # Execute the code
                exec_result = await self.executor.execute(code)

                # Trigger observation event
                if self.event_bus:
                    observation = exec_result.output

                    if exec_result.error:
                        # Trigger error event for execution errors
                        self.event_bus.emit(
                            ErrorEvent(error=exec_result.error, error_type="Execution Error", step=turn_num + 1)
                        )
                    else:
                        self.event_bus.emit(ObservationEvent(observation=observation))
            else:
                # No code to execute - create a dummy result
                from .executor import ExecutionResult

                exec_result = ExecutionResult(output="", error=None, stdout="", stderr="")

                if self.text_mode:
                    # In text mode, code blocks are optional
                    # If there's a thought but no code, treat the thought as the final answer
                    if thought and thought.strip():
                        exec_result.final_answer = thought
                        # Don't show error - this is expected behavior in text mode
                    else:
                        # No thought and no code - this is an error even in text mode
                        if self.event_bus:
                            self.event_bus.emit(
                                ErrorEvent(
                                    error="No response generated. Expected at least a Thought.",
                                    error_type="Format Error",
                                    step=turn_num + 1,
                                )
                            )
                else:
                    # Standard mode: code is required
                    # Show a warning that the LLM didn't generate code
                    if self.event_bus:
                        self.event_bus.emit(
                            ErrorEvent(
                                error="LLM did not generate code. Expected format:\n\nThought: <explanation>\n```python\n<code>\n```",
                                error_type="Format Error",
                                step=turn_num + 1,
                            )
                        )

                    # Add a correction to memory to guide the LLM
                    # Instead of adding a step with empty code, add an observation telling LLM what to do
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
                    self.memory.add_step(
                        thought=thought if thought else "(No thought provided)",
                        code="",
                        output=correction_msg,
                        error=None,
                        tools_called=[],
                    )

                    # Continue to next turn - the correction will be in the observation
                    continue

            # Add this step to memory (only for successful executions or text mode)
            self.memory.add_step(
                thought=thought,
                code=code,
                output=exec_result.output,
                error=exec_result.error,
                tools_called=exec_result.tools_called,
            )

            # Check if final_answer was called during execution
            if exec_result.final_answer is not None:
                # Agent is done!
                self.memory.add_final_answer(exec_result.final_answer)

                # Trigger final answer event
                if self.event_bus:
                    self.event_bus.emit(
                        FinalAnswerEvent(
                            answer=str(exec_result.final_answer),
                            turns=turn_num + 1,
                            tokens=response.usage.total_tokens if response.usage else None,
                            cost=self.total_cost if self.total_cost > 0 else None,
                        )
                    )

                    # Trigger cost summary event
                    total_tokens = response.usage.total_tokens if response.usage else None
                    duration = time.time() - start_time

                    # Extract cache-related fields (supported by OpenAI, Anthropic, Bedrock, Deepseek)
                    cached_tokens = None
                    cache_creation_tokens = None
                    cache_read_tokens = None
                    if response.usage:
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

                if return_full_result:
                    return AgentResult(
                        output=exec_result.final_answer,
                        token_usage=response.usage.total_tokens if response.usage else None,
                        cost=self.total_cost if self.total_cost > 0 else None,
                        steps=self.memory.steps,
                    )
                return exec_result.final_answer

            # Continue loop (LLM will see the observation in next iteration)

        # If we get here, we hit max_turns
        error_msg = f"Agent reached max_turns ({self.max_turns}) without completing task"
        if self.event_bus:
            self.event_bus.emit(ErrorEvent(error=error_msg, error_type="RuntimeError"))

        # For benchmark/testing use cases that need execution trace even on error,
        # return AgentResult with error field set instead of raising
        if return_full_result:
            return AgentResult(
                output=None,
                token_usage=None,
                cost=self.total_cost,
                steps=self.memory.steps,
                error=error_msg,
            )
        else:
            # Backward compatibility: raise exception for non-benchmark usage
            raise RuntimeError(error_msg)

    def _build_messages(self) -> List[Dict]:
        """Build message list for LLM from memory.

        Uses system blocks with cache control when attachments are present
        for better prompt caching support.

        Format with attachments (system blocks):
        [
            {"role": "system", "content": [
                {"type": "text", "text": system_prompt},
                {"type": "text", "text": attachment1, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": attachment2, "cache_control": {"type": "ephemeral"}},
            ]},
            {"role": "user", "content": task},
            ...
        ]

        Format without attachments (legacy):
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
            ...
        ]
        """
        messages = []

        # Build system message with or without attachments
        if self.attachments:
            # Use system blocks with cache control for better caching
            system_blocks = [{"type": "text", "text": self._build_system_prompt()}]

            # Add each attachment as a separate cacheable block
            for name, content in self.attachments:
                system_blocks.append(
                    {
                        "type": "text",
                        "text": f"<Attachment: {name}>\n{content}\n</Attachment: {name}>",
                        "cache_control": {"type": "ephemeral"},
                    }
                )

            messages.append({"role": "system", "content": system_blocks})
        else:
            # Legacy format: simple string
            messages.append({"role": "system", "content": self._build_system_prompt()})

        # Task
        messages.append({"role": "user", "content": self.memory.task})

        # Previous steps (Thought/Code → Observation pairs)
        for step in self.memory.steps:
            # Assistant's thought + code
            assistant_msg = f"Thought: {step.thought}\n\n```python\n{step.code}\n```"
            messages.append({"role": "assistant", "content": assistant_msg})

            # Observation (code execution result)
            observation = f"Observation: {step.output}"
            if step.error:
                observation += f"\nError: {step.error}"

            messages.append({"role": "user", "content": observation})

        return messages

    def _build_system_prompt(self) -> str:
        """Build system prompt that teaches LLM how to solve tasks."""
        return build_system_prompt(self.tools, self.instructions, self.text_mode)

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
## Available tools:

You have access to these Python functions:

```python
{tool_definitions}
```
"""


def build_text_mode_prompt(tools_section: str, instructions: str, has_tools: bool) -> str:
    """Build system prompt for text mode (code blocks optional).

    Args:
        tools_section: Formatted tools section
        instructions: Additional instructions from agent config
        has_tools: Whether tools are available

    Returns:
        Complete system prompt for text mode
    """
    tool_rule = (
        "4. When using code, call tools with keyword arguments: result = tool_name(arg1=value1, arg2=value2)"
        if has_tools
        else "4. Use Python when you need to perform actions"
    )

    return f"""You are an expert assistant who helps with tasks.

You can respond in two ways:

**For conversational questions or simple responses:**
Just provide your Thought with the answer directly:

Thought: [Your response here]

**When you need to use tools or perform actions:**
Provide a Thought and write Python code:

Thought: [What you'll do and why]
```python
# Your code here
final_answer(result)
```
{tools_section}
## Rules:

1. Start with "Thought:" to explain your reasoning
2. Code blocks are OPTIONAL - only use them when you need tools or complex logic
3. For direct answers, just provide the Thought without code
{tool_rule}
5. When using code blocks, call final_answer() with the result
6. Variables persist across code blocks

{instructions}

Now begin!"""


def build_standard_mode_prompt(tools_section: str, instructions: str, has_tools: bool) -> str:
    """Build system prompt for standard mode (code blocks required).

    Args:
        tools_section: Formatted tools section
        instructions: Additional instructions from agent config
        has_tools: Whether tools are available

    Returns:
        Complete system prompt for standard mode
    """
    tool_rule = (
        "3. Call tools with keyword arguments: result = tool_name(arg1=value1, arg2=value2)"
        if has_tools
        else "3. Use standard Python to solve the task"
    )

    return f"""You are an expert assistant who solves tasks using Python code.

To solve a task, you proceed in steps using this pattern:

1. **Thought:** Explain your reasoning (what you'll do and why)
2. **Code:** Write Python code in a code block
3. **Observation:** You'll see the code execution result

You repeat this Thought → Code → Observation cycle until you have the final answer.

## How to write code:

- Always start with a Thought explaining your approach
- Write code in triple-backtick code blocks: ```python
- Use print() to output important information
- Variables persist between code blocks
- When you have the final answer, call: final_answer(your_answer)
{tools_section}
## Rules:

1. Always provide Thought before code
2. Only use variables you've defined
{tool_rule}
4. Call final_answer() when you have the answer
5. If you get an error, try a different approach
6. State persists - variables remain available across code blocks

{instructions}

Now begin!"""
