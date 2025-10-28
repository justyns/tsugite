"""Plain text UI handler without colors, panels, or emojis."""

import re
from contextlib import contextmanager
from typing import Any, Dict, Generator

from tsugite.console import get_stdout_console
from tsugite.ui.base import CustomUIHandler
from tsugite.ui_context import clear_ui_context, set_ui_context


class PlainUIHandler(CustomUIHandler):
    """Plain text UI handler without colors, panels, animations, or emojis.

    This handler provides minimal, copy-paste friendly output by stripping
    all Rich formatting, emojis, and decorative elements. Ideal for:
    - Piped output
    - Copy-paste workflows
    - Screen readers
    - Logs and automation
    """

    def __init__(self):
        """Initialize plain UI handler with no-color console."""
        # Create a console with no color support and force terminal for Progress spinners
        no_color_console = get_stdout_console(no_color=True, force_terminal=True)

        # Initialize parent with panels disabled
        super().__init__(
            console=no_color_console,
            show_code=True,
            show_observations=True,
            show_llm_messages=False,
            show_execution_results=True,
            show_execution_logs=True,
            show_panels=False,
        )

    @staticmethod
    def _strip_emojis(text: str) -> str:
        """Remove all emojis from text.

        Args:
            text: Text potentially containing emojis

        Returns:
            Text with emojis removed
        """
        # Pattern matches emoji characters
        emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map symbols
            "\U0001f1e0-\U0001f1ff"  # flags
            "\U00002702-\U000027b0"
            "\U000024c2-\U0001f251"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub("", text).strip()

    @staticmethod
    def _strip_rich_markup(text: str) -> str:
        """Remove Rich markup tags from text.

        Args:
            text: Text potentially containing Rich markup like [bold], [cyan], etc.

        Returns:
            Text with Rich markup removed
        """
        # Remove Rich color/style tags like [cyan], [/cyan], [bold], etc.
        return re.sub(r"\[/?[a-z\s]+\]", "", text)

    def _handle_task_start(self, data: Dict[str, Any]) -> None:
        """Handle task start event with plain text output."""
        self.state.task = data.get("task")
        self.state.current_step = 0
        self.state.steps_history = []

        # Show plain text header
        self.console.print()
        self.console.rule("Starting Agent Execution")
        self.console.print(f"Task: {self.state.task}")
        model = data.get("model", "")
        if model:
            self.console.print(f"Model: {model}")
        self.console.rule()
        self.console.print()

    def _handle_step_start(self, data: Dict[str, Any]) -> None:
        """Handle step start event with plain text output."""
        self.state.current_step = data.get("step", self.state.current_step + 1)

        # Show "Turn" for reasoning iterations
        label = f"Turn {self.state.current_step}"

        self.console.print(f"{label}: Waiting for LLM response...")

        # Add step to history
        self.state.steps_history.append({"step": self.state.current_step, "status": "in_progress", "actions": []})

    def _handle_code_execution(self, data: Dict[str, Any]) -> None:
        """Handle code execution event with plain text output."""
        self.state.code_being_executed = data.get("code")

        self.console.print("Executing code...")

        if self.show_code and self.state.code_being_executed:
            self.console.print()
            self.console.rule("Executing Code", style="dim")
            self.console.print(self.state.code_being_executed)
            self.console.rule(style="dim")
            self.console.print()

    def _handle_tool_call(self, data: Dict[str, Any]) -> None:
        """Handle tool call event with plain text output."""
        content = data.get("content", "")

        self.console.print("Calling tool...")

        # Add to current step history
        if self.state.steps_history:
            self.state.steps_history[-1]["actions"].append({"type": "tool_call", "content": content})

    def _handle_observation(self, data: Dict[str, Any]) -> None:
        """Handle observation event with plain text output."""
        observation = data.get("observation", "")

        self.console.print("Processing results...")

        if observation:
            # Clean up observation for display
            clean_obs = observation.replace("|", "[").strip()

            # Check if this is a final answer
            is_final_answer = "__FINAL_ANSWER__:" in clean_obs

            # Check if this looks like an error using shared helper from parent class
            is_error = self._contains_error(clean_obs)

            # Always show final answers and errors, even if show_observations is False
            if self.show_observations or is_final_answer or is_error:
                if is_error:
                    # Display errors prominently
                    self.console.print()
                    self.console.rule("ERROR", style="dim")
                    self.console.print(clean_obs)
                    self.console.rule(style="dim")
                    self.console.print()
                elif is_final_answer:
                    # Extract answer content after "__FINAL_ANSWER__: " and render as markdown
                    answer_content = clean_obs.split("__FINAL_ANSWER__:", 1)[1].strip()
                    if answer_content:
                        from rich.markdown import Markdown

                        self.console.print()
                        self.console.print(Markdown(answer_content))
                else:
                    # Normal observation - truncate if too long
                    if len(clean_obs) > 200:
                        clean_obs = clean_obs[:200] + "..."
                    self.console.print(f"Result: {clean_obs}")

        # Add to current step history
        if self.state.steps_history:
            self.state.steps_history[-1]["actions"].append({"type": "observation", "content": observation})
            self.state.steps_history[-1]["status"] = "completed"

    def _handle_final_answer(self, data: Dict[str, Any]) -> None:
        """Handle final answer event with plain text output."""
        answer = data.get("answer", "")

        self.console.print("Finalizing answer...")

        self.console.print()
        self.console.rule("FINAL ANSWER")
        self.console.print(answer)
        self.console.rule()
        self.console.print()

    def _handle_error(self, data: Dict[str, Any]) -> None:
        """Handle error event with plain text output."""
        error = data.get("error", "")
        error_type = data.get("error_type", "Error")

        self.console.print("Error occurred...")

        # Always show errors prominently
        self.console.print()
        self.console.rule(f"{error_type}")
        self.console.print(error)
        self.console.rule()
        self.console.print()

        # Add to current step history
        if self.state.steps_history:
            self.state.steps_history[-1]["actions"].append({"type": "error", "content": error})
            self.state.steps_history[-1]["status"] = "error"

    def _handle_llm_message(self, data: Dict[str, Any]) -> None:
        """Handle LLM reasoning message event with plain text output."""
        if not self.show_llm_messages:
            return

        content = data.get("content", "")
        title = data.get("title", "Agent Reasoning")

        if content.strip():
            self.console.print()
            self.console.rule(title, style="dim")
            self.console.print(content.strip())
            self.console.rule(style="dim")
            self.console.print()

    def _handle_reasoning_content(self, data: Dict[str, Any]) -> None:
        """Handle reasoning content with plain text output."""
        content = data.get("content", "")
        step = data.get("step")

        if content and content.strip():
            # Build title with turn number if available
            if step is not None:
                title = f"Model Reasoning (Turn {step})"
            else:
                title = "Model Reasoning"

            self.console.print("Processing reasoning content...")

            # Truncate very long reasoning content for display
            max_length = 2000
            display_content = content.strip()
            if len(display_content) > max_length:
                display_content = display_content[:max_length] + "\n\n... (truncated)"

            self.console.print()
            self.console.rule(title, style="dim")
            self.console.print(display_content)
            self.console.rule(style="dim")
            self.console.print()

    def _handle_reasoning_tokens(self, data: Dict[str, Any]) -> None:
        """Handle reasoning token counts with plain text output."""
        tokens = data.get("tokens", 0)
        step = data.get("step")

        if tokens:
            # Build message with turn number if available
            if step is not None:
                message = f"Turn {step}: Used {tokens} reasoning tokens"
            else:
                message = f"Used {tokens} reasoning tokens"

            self.console.print(message)

    def _handle_cost_summary(self, data: Dict[str, Any]) -> None:
        """Handle cost summary display after final answer."""
        cost = data.get("cost")
        total_tokens = data.get("total_tokens")
        reasoning_tokens = data.get("reasoning_tokens")

        summary_text = self._build_cost_summary_text(cost, total_tokens, reasoning_tokens, include_emojis=False)
        if not summary_text:
            return

        self.console.print(f"\n{summary_text}\n")

    def _handle_execution_result(self, data: Dict[str, Any]) -> None:
        """Handle code execution result event with plain text output."""
        if not self.show_execution_results:
            return

        self.console.print("Processing execution results...")

        content = data.get("content", "")

        if content.strip():
            # Parse execution logs and output using shared helper from parent class
            execution_logs, output_lines = self._parse_execution_content(content)

            # Display execution logs if present
            if execution_logs:
                logs_text = "\n".join(execution_logs)
                if logs_text.strip():
                    self.console.print(f"Logs: {logs_text}")

            # Display output if present and meaningful
            if output_lines:
                output_text = "\n".join(output_lines)
                contains_error = self._contains_error(output_text)

                # Check if this is a final answer
                is_final_answer = "__FINAL_ANSWER__:" in output_text

                # Always show errors, filter non-meaningful outputs otherwise
                if contains_error:
                    self.console.print()
                    self.console.rule("ERROR IN OUTPUT", style="dim")
                    self.console.print(output_text)
                    self.console.rule(style="dim")
                    self.console.print()
                elif is_final_answer:
                    # Extract answer content after "__FINAL_ANSWER__: " and render as markdown
                    answer_content = output_text.split("__FINAL_ANSWER__:", 1)[1].strip()
                    if answer_content:
                        from rich.markdown import Markdown

                        self.console.print()
                        self.console.print(Markdown(answer_content))
                elif output_text.strip() and output_text.strip().lower() not in ("none", "null", ""):
                    self.console.print(f"Output: {output_text}")

    def _handle_execution_logs(self, data: Dict[str, Any]) -> None:
        """Handle execution logs event with plain text output."""
        if not self.show_execution_logs:
            return

        content = data.get("content", "")

        if content.strip() and "Execution logs:" in content:
            # Extract just the log content
            logs = content.replace("Execution logs:", "").strip()
            if logs:
                self.console.print(f"Logs: {logs}")

    def _handle_subagent_start(self, data: Dict[str, Any]) -> None:
        """Handle subagent start event with plain text output."""
        agent_name = data.get("agent_name", "unknown")

        self.console.print(f"Spawning {agent_name}...")
        self.console.print()
        self.console.rule(f"{agent_name} agent")
        self.console.print()

    def _handle_subagent_end(self, data: Dict[str, Any]) -> None:
        """Handle subagent end event with plain text output."""
        agent_name = data.get("agent_name", "unknown")

        self.console.print()
        self.console.rule()
        self.console.print(f"{agent_name} completed")
        self.console.print()

    @contextmanager
    def progress_context(self) -> Generator[None, None, None]:
        """Context manager for showing progress during execution.

        Plain UI handler shows minimal progress spinners for subagent tracking.
        When no_color is enabled, completely disables progress spinners to avoid ANSI codes.
        """
        # If no_color is enabled, skip all progress/spinner output
        if self.console.no_color:
            # Just set the UI context without any progress
            set_ui_context(console=self.console, progress=None, ui_handler=self)
            try:
                yield
            finally:
                clear_ui_context()
            return

        from rich.progress import Progress, SpinnerColumn, TextColumn

        # Create progress with simple spinner for minimal UI
        # Non-transient so subagent spinners remain visible during execution
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=False,
            refresh_per_second=4,  # Ensure regular refreshes for spinner animation
        )

        # Store console, progress, and ui_handler in thread-local for tool access
        set_ui_context(console=self.console, progress=progress, ui_handler=self)

        # Use context manager to start live rendering
        with progress:
            # Add a hidden task to keep Progress Live display active
            # Without this, the Live display may not render properly when tasks are added dynamically
            dummy_task = progress.add_task("[dim]Executing...[/dim]", total=None)

            try:
                yield
            finally:
                if dummy_task is not None:
                    progress.remove_task(dummy_task)
                clear_ui_context()

    def update_progress(self, description: str) -> None:
        """Update progress description.

        Plain UI handler silently ignores progress updates to avoid clutter.
        """
        # No-op in plain mode - we don't show progress spinners
        pass
