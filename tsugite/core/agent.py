"""Core agent implementation using LiteLLM directly.

A simpler, more direct implementation that gives us full control over
model parameters and reasoning model support.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import litellm

from tsugite.custom_ui import UIEvent

from .executor import CodeExecutor, LocalExecutor
from .memory import AgentMemory, StepResult
from .tools import Tool


@dataclass
class AgentResult:
    """Result from agent execution."""

    output: Any
    token_usage: Optional[int] = None
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

        # Build tool map for quick lookup
        self.tool_map = {tool.name: tool for tool in tools}

        # Pre-compute LiteLLM params (filters unsupported params for reasoning models)
        # This is more efficient and correct - we filter once, not on every loop iteration
        self.litellm_params = get_model_params(model_string, **(model_kwargs or {}))

    async def run(self, task: str, return_full_result: bool = False):
        """Run the agent on a task.

        Args:
            task: The task to solve
            return_full_result: If True, return AgentResult with metadata

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
            response = await litellm.acompletion(messages=messages, **self.litellm_params)

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

            # Parse LLM response
            # Response should contain: Thought + Code OR final_answer()
            thought, code, _ = self._parse_response(response)

            # Show LLM's thought/reasoning (always show what the LLM is saying)
            if self.ui_handler:
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

                if return_full_result:
                    return AgentResult(
                        output=exec_result.final_answer,
                        token_usage=response.usage.total_tokens if response.usage else None,
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
{"3. Call tools directly: result = tool_name(arg1, arg2)" if self.tools else "3. Use standard Python to solve the task"}
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
