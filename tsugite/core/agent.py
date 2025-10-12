"""Core agent implementation using LiteLLM directly.

A simpler, more direct implementation that gives us full control over
model parameters and reasoning model support.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import litellm

from tsugite.ui import UIEvent

from .executor import CodeExecutor, LocalExecutor
from .memory import AgentMemory, StepResult
from .tools import Tool


@dataclass
class AgentResult:
    """Result from agent execution."""

    output: Any
    token_usage: Optional[int] = None
    cost: Optional[float] = None
    steps: Optional[List[StepResult]] = None


class TsugiteAgent:
    """Custom agent that uses Thought/Code/Observation loop.

    Provides direct access to LiteLLM features including reasoning models,
    custom parameters, and full control over the execution loop.

    How it works:
    1. Build system prompt from tools + instructions
    2. Loop: Send messages → Get response → Execute code → Repeat
    3. Stop when final_answer() is called or max_steps reached

    Example:
        agent = TsugiteAgent(
            model_string="openai:gpt-4o-mini",
            tools=[tool1, tool2],
            instructions="You are a helpful assistant",
            max_steps=10
        )

        result = await agent.run("Calculate 5 + 3")
        print(result)  # "8"
    """

    def __init__(
        self,
        model_string: str,
        tools: List[Tool],
        instructions: str = "",
        max_steps: int = 10,
        executor: CodeExecutor = None,
        model_kwargs: dict = None,
        ui_handler: Any = None,
        model_name: str = None,
        text_mode: bool = False,
    ):
        """Initialize the agent.

        Args:
            model_string: Model identifier like "openai:gpt-4o-mini"
            tools: List of Tool objects the agent can use
            instructions: Additional instructions to append to system prompt
            max_steps: Maximum number of reasoning steps before giving up
            executor: Code executor (microsandbox or local). If None, uses LocalExecutor
            model_kwargs: Extra parameters for LiteLLM (reasoning_effort, response_format, etc.)
            ui_handler: Optional UI handler for displaying progress
            model_name: Optional display name for the model (for UI)
            text_mode: Allow text-only responses (code blocks optional)
        """
        from tsugite.models import get_model_params

        self.model_string = model_string
        self.tools = tools
        self.instructions = instructions
        self.max_steps = max_steps
        self.executor = executor or LocalExecutor()
        self.memory = AgentMemory()
        self.ui_handler = ui_handler
        self.model_name = model_name or model_string
        self.text_mode = text_mode

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
                    # Convert positional args to keyword args if needed
                    if args:
                        # Tools should be called with keyword args, but if positional
                        # args are provided, this is likely an error
                        raise TypeError(
                            f"Tool '{tool_obj.name}' must be called with keyword arguments, "
                            f"not positional arguments. "
                            f"Example: {tool_obj.name}(param1=value1, param2=value2)"
                        )

                    # Get the running event loop, or create one if needed
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # Already in async context - use thread to block on async result
                            import concurrent.futures

                            with concurrent.futures.ThreadPoolExecutor() as executor:

                                def run_async():
                                    new_loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(new_loop)
                                    try:
                                        return new_loop.run_until_complete(tool_obj.execute(**kwargs))
                                    finally:
                                        new_loop.close()

                                return executor.submit(run_async).result()
                        else:
                            # Not in async context - run synchronously
                            return loop.run_until_complete(tool_obj.execute(**kwargs))
                    except RuntimeError:
                        # No event loop - create one
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            return loop.run_until_complete(tool_obj.execute(**kwargs))
                        finally:
                            loop.close()

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
            or AgentResult: Full result with token usage and steps

        Raises:
            RuntimeError: If agent reaches max_steps without finishing
        """
        # Add task to memory
        self.memory.add_task(task)

        # Trigger task start event
        if self.ui_handler:
            self.ui_handler.handle_event(UIEvent.TASK_START, {"task": task, "model": self.model_name})

        # Main agent loop
        for step_num in range(self.max_steps):
            # Trigger step start event
            if self.ui_handler:
                self.ui_handler.handle_event(UIEvent.STEP_START, {"step": step_num + 1})

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
                            if self.ui_handler:
                                self.ui_handler.handle_event(
                                    UIEvent.STREAM_CHUNK, {"chunk": chunk_text, "step": step_num + 1}
                                )

                    # Save the last chunk as response for usage/cost tracking
                    response = chunk

                # Emit stream complete event
                if self.ui_handler:
                    self.ui_handler.handle_event(UIEvent.STREAM_COMPLETE, {"step": step_num + 1})

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
                if self.ui_handler:
                    self.ui_handler.handle_event(
                        UIEvent.REASONING_CONTENT, {"content": reasoning_content, "step": step_num + 1}
                    )

            # Check for reasoning tokens (o1/o3 models)
            if response.usage and hasattr(response.usage, "completion_tokens_details"):
                details = response.usage.completion_tokens_details
                if hasattr(details, "reasoning_tokens") and details.reasoning_tokens:
                    if self.ui_handler:
                        self.ui_handler.handle_event(
                            UIEvent.REASONING_TOKENS, {"tokens": details.reasoning_tokens, "step": step_num + 1}
                        )

            # Show LLM's thought/reasoning (always show what the LLM is saying)
            # Skip this if streaming (already shown via STREAM_CHUNK events)
            if self.ui_handler and not stream:
                # If we parsed a thought, show it. Otherwise show the raw response
                # (this helps debug when LLM doesn't follow the expected format)
                display_content = thought if thought else response.choices[0].message.content
                if display_content and display_content.strip():
                    self.ui_handler.handle_event(
                        UIEvent.LLM_MESSAGE, {"content": display_content, "title": f"Step {step_num + 1} Reasoning"}
                    )

            # Only execute code if the LLM actually generated some
            if code and code.strip():
                # Trigger code execution event
                if self.ui_handler:
                    self.ui_handler.handle_event(UIEvent.CODE_EXECUTION, {"code": code})

                # Execute the code
                exec_result = await self.executor.execute(code)

                # Trigger observation event
                if self.ui_handler:
                    observation = exec_result.output
                    if exec_result.error:
                        # Trigger error event for execution errors
                        self.ui_handler.handle_event(
                            UIEvent.ERROR, {"error": exec_result.error, "error_type": "Execution Error"}
                        )
                    else:
                        self.ui_handler.handle_event(UIEvent.OBSERVATION, {"observation": observation})
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
                        if self.ui_handler:
                            self.ui_handler.handle_event(
                                UIEvent.ERROR,
                                {
                                    "error": "No response generated. Expected at least a Thought.",
                                    "error_type": "Format Error",
                                },
                            )
                else:
                    # Standard mode: code is required
                    # Show a warning that the LLM didn't generate code
                    if self.ui_handler:
                        self.ui_handler.handle_event(
                            UIEvent.ERROR,
                            {
                                "error": "LLM did not generate code. Expected format:\n\nThought: <explanation>\n```python\n<code>\n```",
                                "error_type": "Format Error",
                            },
                        )

            # Add this step to memory (always, even if final_answer was called)
            self.memory.add_step(thought=thought, code=code, output=exec_result.output, error=exec_result.error)

            # Check if final_answer was called during execution
            if exec_result.final_answer is not None:
                # Agent is done!
                self.memory.add_final_answer(exec_result.final_answer)

                # Trigger final answer event
                if self.ui_handler:
                    self.ui_handler.handle_event(UIEvent.FINAL_ANSWER, {"answer": str(exec_result.final_answer)})

                    # Trigger cost summary event
                    total_tokens = response.usage.total_tokens if response.usage else None
                    reasoning_tokens = None
                    if response.usage and hasattr(response.usage, "completion_tokens_details"):
                        details = response.usage.completion_tokens_details
                        if hasattr(details, "reasoning_tokens"):
                            reasoning_tokens = details.reasoning_tokens

                    self.ui_handler.handle_event(
                        UIEvent.COST_SUMMARY,
                        {
                            "cost": self.total_cost if self.total_cost > 0 else None,
                            "total_tokens": total_tokens,
                            "reasoning_tokens": reasoning_tokens,
                        },
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

        # If we get here, we hit max_steps
        error_msg = f"Agent reached max_steps ({self.max_steps}) without completing task"
        if self.ui_handler:
            self.ui_handler.handle_event(UIEvent.ERROR, {"error": error_msg, "error_type": "RuntimeError"})

        raise RuntimeError(error_msg)

    def _build_messages(self) -> List[Dict[str, str]]:
        """Build message list for LLM from memory.

        Format:
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
            {"role": "assistant", "content": thought + code},
            {"role": "user", "content": observation},
            ...
        ]
        """
        messages = []

        # System prompt (explains how to solve tasks)
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
        # Build tools section (only if tools exist)
        tools_section = ""
        if self.tools:
            tool_definitions = "\n\n".join([tool.to_code_prompt() for tool in self.tools])
            tools_section = f"""
## Available tools:

You have access to these Python functions:

```python
{tool_definitions}
```
"""

        if self.text_mode:
            # Text mode: code blocks are optional
            prompt = f"""You are an expert assistant who helps with tasks.

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
{"4. When using code, call tools with keyword arguments: result = tool_name(arg1=value1, arg2=value2)" if self.tools else "4. Use Python when you need to perform actions"}
5. When using code blocks, call final_answer() with the result
6. Variables persist across code blocks

{self.instructions}

Now begin!"""
        else:
            # Standard mode: code blocks required
            prompt = f"""You are an expert assistant who solves tasks using Python code.

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
{"3. Call tools with keyword arguments: result = tool_name(arg1=value1, arg2=value2)" if self.tools else "3. Use standard Python to solve the task"}
4. Call final_answer() when you have the answer
5. If you get an error, try a different approach
6. State persists - variables remain available across code blocks

{self.instructions}

Now begin!"""

        return prompt

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
