"""Base UI handler and core UI system components."""

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Generator, List, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.text import Text

from tsugite.ui_context import clear_ui_context, set_ui_context


class UIEvent(IntEnum):
    """Events that the UI can handle."""

    TASK_START = 1
    STEP_START = 2
    CODE_EXECUTION = 3
    TOOL_CALL = 4
    OBSERVATION = 5
    ERROR = 8
    FINAL_ANSWER = 9
    LLM_MESSAGE = 10
    EXECUTION_RESULT = 11
    EXECUTION_LOGS = 12
    REASONING_CONTENT = 13
    REASONING_TOKENS = 14
    COST_SUMMARY = 15
    STREAM_CHUNK = 16  # For streaming LLM responses
    STREAM_COMPLETE = 17  # When streaming finishes


@dataclass
class UIState:
    """Tracks the current state of the agent execution."""

    task: Optional[str] = None
    current_step: int = 0
    total_steps: Optional[int] = None
    code_being_executed: Optional[str] = None
    steps_history: List[Dict[str, Any]] = None
    multistep_context: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.steps_history is None:
            self.steps_history = []


class CustomUILogger:
    """Simple logger wrapper for TsugiteAgent.

    Provides console and ui_handler access for displaying reasoning content
    and multi-step progress. This is a minimal wrapper providing access to
    the UI handler and console for rendering.
    """

    def __init__(self, ui_handler: "CustomUIHandler", console: Console):
        """Initialize logger.

        Args:
            ui_handler: Handler for UI events
            console: Rich console for output
        """
        self.ui_handler = ui_handler
        self.console = console


class CustomUIHandler:
    """Handles UI events and displays custom progress interface."""

    def __init__(
        self,
        console: Console,
        show_code: bool = True,
        show_observations: bool = True,
        show_llm_messages: bool = False,
        show_execution_results: bool = True,
        show_execution_logs: bool = True,
        show_panels: bool = True,
    ):
        self.console = console
        self.state = UIState()
        self.show_code = show_code
        self.show_observations = show_observations
        self.show_llm_messages = show_llm_messages
        self.show_execution_results = show_execution_results
        self.show_execution_logs = show_execution_logs
        self.show_panels = show_panels
        self.progress = None
        self.task_id = None
        self._lock = threading.Lock()

        # Streaming state
        self.streaming_content = ""
        self.is_streaming = False

    def handle_event(self, event: UIEvent, data: Dict[str, Any]) -> None:
        """Handle a UI event and update the display."""
        with self._lock:
            if event == UIEvent.TASK_START:
                self._handle_task_start(data)
            elif event == UIEvent.STEP_START:
                self._handle_step_start(data)
            elif event == UIEvent.CODE_EXECUTION:
                self._handle_code_execution(data)
            elif event == UIEvent.TOOL_CALL:
                self._handle_tool_call(data)
            elif event == UIEvent.OBSERVATION:
                self._handle_observation(data)
            elif event == UIEvent.FINAL_ANSWER:
                self._handle_final_answer(data)
            elif event == UIEvent.ERROR:
                self._handle_error(data)
            elif event == UIEvent.LLM_MESSAGE:
                self._handle_llm_message(data)
            elif event == UIEvent.EXECUTION_RESULT:
                self._handle_execution_result(data)
            elif event == UIEvent.EXECUTION_LOGS:
                self._handle_execution_logs(data)
            elif event == UIEvent.REASONING_CONTENT:
                self._handle_reasoning_content(data)
            elif event == UIEvent.REASONING_TOKENS:
                self._handle_reasoning_tokens(data)
            elif event == UIEvent.COST_SUMMARY:
                self._handle_cost_summary(data)
            elif event == UIEvent.STREAM_CHUNK:
                self._handle_stream_chunk(data)
            elif event == UIEvent.STREAM_COMPLETE:
                self._handle_stream_complete(data)

            self._update_display()

    def _get_display_prefix(self) -> str:
        """Get display prefix for nested multi-step context."""
        if self.state.multistep_context:
            return "  └─ "
        return ""

    @staticmethod
    def _parse_execution_content(content: str) -> tuple:
        """Parse execution result content into logs and output.

        Args:
            content: The execution result content to parse

        Returns:
            Tuple of (execution_logs, output_lines)
        """
        lines = content.split("\n")
        execution_logs = []
        output_lines = []

        current_section = None
        for line in lines:
            if line.startswith("Execution logs:"):
                current_section = "logs"
            elif line.startswith("Out:"):
                current_section = "output"
                output_lines.append(line[4:].strip())  # Remove 'Out:' prefix
            elif current_section == "logs" and line.strip():
                execution_logs.append(line.strip())
            elif current_section == "output" and line.strip():
                output_lines.append(line.strip())

        return execution_logs, output_lines

    @staticmethod
    def _contains_error(text: str) -> bool:
        """Check if text contains error keywords.

        Args:
            text: Text to check for error keywords

        Returns:
            True if text contains error indicators
        """
        error_keywords = ["error", "failed", "exception", "not found", "invalid", "traceback"]
        return any(keyword in text.lower() for keyword in error_keywords)

    def _handle_task_start(self, data: Dict[str, Any]) -> None:
        """Handle task start event."""
        self.state.task = data.get("task")
        self.state.current_step = 0
        self.state.steps_history = []

        # Show initial task panel
        if self.show_panels:
            self.console.print(
                Panel(
                    f"[bold]{self.state.task}[/bold]",
                    title="[bold cyan]🚀 Starting Agent Execution[/bold cyan]",
                    subtitle=data.get("model", ""),
                    border_style="cyan",
                )
            )

    def _handle_step_start(self, data: Dict[str, Any]) -> None:
        """Handle step start event."""
        self.state.current_step = data.get("step", self.state.current_step + 1)

        prefix = self._get_display_prefix()

        # Show "Turn" for reasoning iterations
        # (workflow steps are shown separately in multistep_context)
        label = f"Turn {self.state.current_step}"

        # Update progress
        self.update_progress(f"{prefix}🤔 {label}: Waiting for LLM response...")

        # Add step to history
        self.state.steps_history.append({"step": self.state.current_step, "status": "in_progress", "actions": []})

    def _handle_code_execution(self, data: Dict[str, Any]) -> None:
        """Handle code execution event."""
        self.state.code_being_executed = data.get("code")

        prefix = self._get_display_prefix()
        # Update progress
        self.update_progress(f"{prefix}⚡ Executing code...")

        if self.show_code and self.show_panels and self.state.code_being_executed:
            self.console.print(
                Panel(
                    Syntax(self.state.code_being_executed, "python", theme="monokai"),
                    title="[bold yellow]⚡ Executing Code[/bold yellow]",
                    border_style="yellow",
                )
            )
        elif not self.show_panels:
            # In minimal mode, show lightweight indicator
            self.console.print("[dim yellow]⚡ Executing code...[/dim yellow]")

    def _handle_tool_call(self, data: Dict[str, Any]) -> None:
        """Handle tool call event."""
        content = data.get("content", "")

        prefix = self._get_display_prefix()
        # Update progress
        self.update_progress(f"{prefix}🔧 Calling tool...")

        # In minimal mode, show lightweight indicator with tool name
        if not self.show_panels and content:
            # Parse tool name from content (format: "Tool: tool_name")
            tool_name = content.replace("Tool: ", "").strip() if "Tool: " in content else content
            self.console.print(f"[dim cyan]🔧 Called: {tool_name}[/dim cyan]")

        # Add to current step history
        if self.state.steps_history:
            self.state.steps_history[-1]["actions"].append({"type": "tool_call", "content": content})

    def _handle_observation(self, data: Dict[str, Any]) -> None:
        """Handle observation event."""
        observation = data.get("observation", "")

        prefix = self._get_display_prefix()
        # Update progress
        self.update_progress(f"{prefix}💡 Processing results...")

        if self.show_observations and observation:
            # Clean up observation for display
            clean_obs = observation.replace("|", "[").strip()

            # Check if this is a final answer
            is_final_answer = "__FINAL_ANSWER__:" in clean_obs

            # Check if this looks like an error using shared helper
            is_error = self._contains_error(clean_obs)

            if is_error:
                # Display errors prominently in red without truncation
                if self.show_panels:
                    self.console.print(
                        Panel(
                            f"[red]{clean_obs}[/red]",
                            title="[bold red]⚠️  Error[/bold red]",
                            border_style="red",
                        )
                    )
                else:
                    self.console.print(f"[red]⚠️  {clean_obs}[/red]")
            elif is_final_answer:
                # Final answer - don't truncate, render as markdown in minimal mode
                # Extract answer content after "__FINAL_ANSWER__: "
                answer_content = clean_obs.split("__FINAL_ANSWER__:", 1)[1].strip()
                if self.show_panels:
                    # In panel mode, display with panel
                    self.console.print(
                        Panel(
                            answer_content,
                            title="[bold green]✅ Final Answer[/bold green]",
                            border_style="green",
                        )
                    )
                else:
                    # In minimal mode, render as markdown
                    self.console.print(Markdown(answer_content))
            else:
                # Normal observation - truncate if too long
                if len(clean_obs) > 200:
                    clean_obs = clean_obs[:200] + "..."
                self.console.print(f"[dim]💡 {clean_obs}[/dim]")
        elif not self.show_panels and observation:
            # In minimal mode when not showing observations, show completion indicator
            # But still show errors
            clean_obs = observation.replace("|", "[").strip()
            is_error = self._contains_error(clean_obs)
            if is_error:
                self.console.print(f"[red]⚠️  {clean_obs}[/red]")
            else:
                self.console.print("[dim green]✓ Completed[/dim green]")

        # Add to current step history
        if self.state.steps_history:
            self.state.steps_history[-1]["actions"].append({"type": "observation", "content": observation})
            self.state.steps_history[-1]["status"] = "completed"

    def _handle_final_answer(self, data: Dict[str, Any]) -> None:
        """Handle final answer event."""
        answer = data.get("answer", "")

        prefix = self._get_display_prefix()
        # Update progress
        self.update_progress(f"{prefix}✅ Finalizing answer...")

        if self.show_panels:
            self.console.print(
                Panel(
                    f"[bold green]{answer}[/bold green]",
                    title="[bold green]✅ Final Answer[/bold green]",
                    border_style="green",
                )
            )

    def _handle_error(self, data: Dict[str, Any]) -> None:
        """Handle error event."""
        error = data.get("error", "")
        error_type = data.get("error_type", "Error")

        prefix = self._get_display_prefix()
        # Update progress
        self.update_progress(f"{prefix}❌ Error occurred...")

        # Always show errors prominently
        if self.show_panels:
            self.console.print(
                Panel(
                    f"[bold red]{error}[/bold red]",
                    title=f"[bold red]⚠️  {error_type}[/bold red]",
                    border_style="red",
                )
            )
        else:
            self.console.print(f"[bold red]⚠️  {error_type}: {error}[/bold red]")

        # Add to current step history
        if self.state.steps_history:
            self.state.steps_history[-1]["actions"].append({"type": "error", "content": error})
            self.state.steps_history[-1]["status"] = "error"

    def _handle_llm_message(self, data: Dict[str, Any]) -> None:
        """Handle LLM reasoning message event."""
        if not self.show_llm_messages:
            return

        content = data.get("content", "")
        title = data.get("title", "Agent Reasoning")

        if content.strip():
            # Clean up the content and show as reasoning
            if self.show_panels:
                self.console.print(
                    Panel(
                        content.strip(),
                        title=f"[bold blue]🤔 {title}[/bold blue]",
                        border_style="blue",
                        padding=(0, 1),
                    )
                )
            else:
                # In minimal mode, render as markdown
                self.console.print(Markdown(content.strip()))

    def _handle_reasoning_content(self, data: Dict[str, Any]) -> None:
        """Handle reasoning content from reasoning models (Claude, Deepseek with exposed reasoning)."""
        content = data.get("content", "")
        step = data.get("step")

        if content and content.strip():
            prefix = self._get_display_prefix()

            # Build title with step number if available
            title_parts = ["[bold magenta]🧠 Model Reasoning"]
            if step is not None:
                title_parts.append(f" (Turn {step})")
            title_parts.append("[/bold magenta]")
            title = "".join(title_parts)

            # Update progress
            self.update_progress(f"{prefix}🧠 Processing reasoning content...")

            if self.show_panels:
                # Truncate very long reasoning content for display
                max_length = 2000
                display_content = content.strip()
                if len(display_content) > max_length:
                    display_content = display_content[:max_length] + "\n\n[dim]... (truncated)[/dim]"

                self.console.print(
                    Panel(
                        display_content,
                        title=title,
                        border_style="magenta",
                        padding=(0, 1),
                    )
                )
            else:
                # In headless/no-panel mode, print with prefix
                self.console.print(f"[magenta]🧠 Reasoning: {content.strip()}[/magenta]")

    def _handle_reasoning_tokens(self, data: Dict[str, Any]) -> None:
        """Handle reasoning token counts from models like o1/o3 that don't expose reasoning content."""
        tokens = data.get("tokens", 0)
        step = data.get("step")

        if tokens:
            # Build message with turn number if available
            if step is not None:
                message = f"🧠 Turn {step}: Used {tokens} reasoning tokens"
            else:
                message = f"🧠 Used {tokens} reasoning tokens"

            if self.show_panels:
                self.console.print(Text(message, style="dim magenta"))
            else:
                # In headless mode, still show it but more concisely
                self.console.print(f"[dim magenta]{message}[/dim magenta]")

    def _build_cost_summary_text(
        self,
        cost: Optional[float],
        total_tokens: Optional[int],
        reasoning_tokens: Optional[int],
        include_emojis: bool = True,
    ) -> Optional[str]:
        """Build cost summary text from metrics.

        Args:
            cost: Execution cost in dollars
            total_tokens: Total tokens used
            reasoning_tokens: Reasoning tokens used
            include_emojis: Whether to include emoji decorations

        Returns:
            Formatted summary text or None if no metrics available
        """
        if cost is None and total_tokens is None:
            return None

        parts = []
        if cost is not None and cost > 0:
            cost_prefix = "💰 " if include_emojis else ""
            parts.append(f"{cost_prefix}Cost: ${cost:.6f}")

        if total_tokens is not None:
            token_prefix = "📊 " if include_emojis else ""
            if reasoning_tokens is not None and reasoning_tokens > 0:
                parts.append(f"{token_prefix}Tokens: {total_tokens:,} total ({reasoning_tokens:,} reasoning)")
            else:
                parts.append(f"{token_prefix}Tokens: {total_tokens:,}")

        if not parts:
            return None

        return " | ".join(parts)

    def _handle_cost_summary(self, data: Dict[str, Any]) -> None:
        """Handle cost summary display after final answer."""
        cost = data.get("cost")
        total_tokens = data.get("total_tokens")
        reasoning_tokens = data.get("reasoning_tokens")

        summary_text = self._build_cost_summary_text(cost, total_tokens, reasoning_tokens)
        if not summary_text:
            return

        if self.show_panels:
            self.console.print(Text(summary_text, style="dim cyan"))
        else:
            self.console.print(f"[dim cyan]{summary_text}[/dim cyan]")

    def _handle_execution_result(self, data: Dict[str, Any]) -> None:
        """Handle code execution result event."""
        if not self.show_execution_results:
            return

        prefix = self._get_display_prefix()
        # Update progress
        self.update_progress(f"{prefix}📊 Processing execution results...")

        content = data.get("content", "")

        if content.strip():
            # Parse execution logs and output using shared helper
            execution_logs, output_lines = self._parse_execution_content(content)

            # Display execution logs if present (always show with execution results)
            if execution_logs:
                logs_text = "\n".join(execution_logs)
                if logs_text.strip():
                    self.console.print(f"[dim]📝 {logs_text}[/dim]")

            # Display output if present and meaningful
            if output_lines:
                output_text = "\n".join(output_lines)
                contains_error = self._contains_error(output_text)

                # Always show errors, filter non-meaningful outputs otherwise
                if contains_error:
                    # Show errors prominently
                    if self.show_panels:
                        self.console.print(
                            Panel(
                                f"[red]{output_text}[/red]",
                                title="[bold red]⚠️  Error in Output[/bold red]",
                                border_style="red",
                            )
                        )
                    else:
                        self.console.print(f"[red]📤 Output (Error): {output_text}[/red]")
                elif output_text.strip() and output_text.strip().lower() not in ("none", "null", ""):
                    # Show normal meaningful output
                    self.console.print(f"[bold cyan]📤 Output:[/bold cyan] {output_text}")

    def _handle_execution_logs(self, data: Dict[str, Any]) -> None:
        """Handle execution logs event."""
        if not self.show_execution_logs:
            return

        content = data.get("content", "")

        if content.strip() and "Execution logs:" in content:
            # Extract just the log content
            logs = content.replace("Execution logs:", "").strip()
            if logs:
                self.console.print(f"[dim]📝 {logs}[/dim]")

    def _handle_stream_chunk(self, data: Dict[str, Any]) -> None:
        """Handle streaming chunk event."""
        chunk = data.get("chunk", "")
        self.streaming_content += chunk
        self.is_streaming = True

        prefix = self._get_display_prefix()
        # Update progress to show streaming
        self.update_progress(f"{prefix}💬 Streaming response...")

        # Print the chunk directly for real-time feedback
        self.console.print(chunk, end="", highlight=False)

    def _handle_stream_complete(self, data: Dict[str, Any]) -> None:
        """Handle streaming complete event."""
        self.is_streaming = False

        # Print newline after streaming completes
        self.console.print()

        prefix = self._get_display_prefix()
        # Update progress
        self.update_progress(f"{prefix}✅ Streaming complete")

        # Clear streaming content for next step
        self.streaming_content = ""

    def _update_display(self) -> None:
        """Update the live display with current state."""
        # This could be enhanced with a live progress display
        # For now, we use discrete updates
        pass

    @contextmanager
    def progress_context(self) -> Generator[None, None, None]:
        """Context manager for showing progress during execution."""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}", justify="left"),
            transient=True,
            console=self.console,
        )

        # Store console, progress, and ui_handler in thread-local for tool access
        set_ui_context(console=self.console, progress=self.progress, ui_handler=self)

        with self.progress:
            self.task_id = self.progress.add_task("Starting agent...", total=None)
            try:
                yield
            finally:
                self.progress.stop()
                clear_ui_context()

    def update_progress(self, description: str) -> None:
        """Update progress description."""
        if self.progress and self.task_id is not None:
            self.progress.update(self.task_id, description=description)

    @contextmanager
    def pause_for_input(self) -> Generator[None, None, None]:
        """Pause the progress display for user input.

        Stops the progress, shows the prompt, then restarts it.
        The transient=True flag ensures clean redraw when restarted.
        """
        if self.progress is not None:
            self.progress.stop()

        try:
            yield
        finally:
            if self.progress is not None:
                self.progress.start()

    def set_multistep_context(self, step_number: int, step_name: str, total_steps: int) -> None:
        """Set multi-step execution context.

        Args:
            step_number: Current multi-step number (1-indexed)
            step_name: Name of current multi-step
            total_steps: Total number of multi-steps
        """
        self.state.multistep_context = {
            "step_number": step_number,
            "step_name": step_name,
            "total_steps": total_steps,
        }

    def clear_multistep_context(self) -> None:
        """Clear multi-step execution context."""
        self.state.multistep_context = None
