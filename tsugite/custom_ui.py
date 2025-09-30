"""Custom UI system for controlling agent execution display."""

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Generator, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from smolagents.monitoring import AgentLogger, LogLevel


class UIEvent(IntEnum):
    """Events that the UI can handle."""

    TASK_START = 1
    STEP_START = 2
    CODE_EXECUTION = 3
    TOOL_CALL = 4
    OBSERVATION = 5
    STEP_END = 6
    TASK_END = 7
    ERROR = 8
    FINAL_ANSWER = 9
    LLM_MESSAGE = 10
    EXECUTION_RESULT = 11
    EXECUTION_LOGS = 12


@dataclass
class UIState:
    """Tracks the current state of the agent execution."""

    task: Optional[str] = None
    current_step: int = 0
    total_steps: Optional[int] = None
    current_action: Optional[str] = None
    code_being_executed: Optional[str] = None
    last_observation: Optional[str] = None
    steps_history: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.steps_history is None:
            self.steps_history = []


class CustomUILogger(AgentLogger):
    """Custom logger that captures events and forwards them to our UI system."""

    def __init__(self, ui_handler: "CustomUIHandler", level: LogLevel = LogLevel.OFF):
        # Initialize with OFF to suppress all output by default
        super().__init__(level=level, console=Console(file=open("/dev/null", "w")))
        self.ui_handler = ui_handler

    def log_task(self, content: str, subtitle: str, title: str | None = None, level: LogLevel = LogLevel.INFO) -> None:
        """Capture task start event."""
        self.ui_handler.handle_event(UIEvent.TASK_START, {"task": content.strip(), "model": subtitle, "title": title})

    def log_rule(self, title: str, level: int = LogLevel.INFO) -> None:
        """Capture step start event."""
        if "Step" in title:
            step_num = int(title.split()[1]) if len(title.split()) > 1 else 0
            self.ui_handler.handle_event(UIEvent.STEP_START, {"step": step_num, "title": title})

    def log_code(self, title: str, content: str, level: int = LogLevel.INFO) -> None:
        """Capture code execution event."""
        self.ui_handler.handle_event(UIEvent.CODE_EXECUTION, {"title": title, "code": content})

    def log_markdown(self, content: str, title: str | None = None, level=LogLevel.INFO, style=None) -> None:
        """Capture LLM markdown output."""
        if title and "Output message of the LLM" in title:
            self.ui_handler.handle_event(UIEvent.LLM_MESSAGE, {"content": content, "title": title, "level": level})

    def log(self, *args, level: int | str | LogLevel = LogLevel.INFO, **kwargs) -> None:
        """Capture general log events."""
        from rich.text import Text

        # Handle Rich Group objects (execution results)
        if len(args) == 1 and hasattr(args[0], "renderables"):
            group = args[0]
            texts = []
            for renderable in group.renderables:
                if hasattr(renderable, "plain"):
                    texts.append(renderable.plain)
                elif hasattr(renderable, "__str__"):
                    texts.append(str(renderable))

            combined_text = "\n".join(texts)

            # Check if this is execution output
            if "Execution logs:" in combined_text or "Out:" in combined_text:
                self.ui_handler.handle_event(UIEvent.EXECUTION_RESULT, {"content": combined_text, "level": level})
                return

        # Handle individual Text objects
        if len(args) == 1 and isinstance(args[0], Text):
            text_content = args[0].plain
            if text_content.startswith("[Step ") and "Duration" in text_content:
                # This is step timing info, skip or handle differently
                return

        # Convert all args to string for pattern matching
        content = " ".join(str(arg) for arg in args)

        # Detect different types of events based on content
        if "Calling tool:" in content:
            # Extract tool name and arguments
            self.ui_handler.handle_event(UIEvent.TOOL_CALL, {"content": content})
        elif "Observations:" in content:
            # Extract observation
            observation = content.replace("Observations:", "").strip()
            self.ui_handler.handle_event(UIEvent.OBSERVATION, {"observation": observation})
        elif "Final answer:" in content:
            # Extract final answer
            answer = content.replace("Final answer:", "").strip()
            self.ui_handler.handle_event(UIEvent.FINAL_ANSWER, {"answer": answer})
        elif "Execution logs:" in content:
            # Handle execution logs
            self.ui_handler.handle_event(UIEvent.EXECUTION_LOGS, {"content": content, "level": level})
        else:
            # Check if this looks like an error message that doesn't fit expected patterns
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in ["error", "failed", "exception", "traceback"]):
                # This is an unhandled error - display it prominently
                self.ui_handler.handle_event(UIEvent.ERROR, {"error": content, "error_type": "Unexpected Error"})


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
        self.live = None
        self._lock = threading.Lock()

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

            self._update_display()

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
        self.state.current_action = "Thinking..."

        # Update progress
        self.update_progress(f"🤔 Step {self.state.current_step}: Waiting for LLM response...")

        # Add step to history
        self.state.steps_history.append({"step": self.state.current_step, "status": "in_progress", "actions": []})

    def _handle_code_execution(self, data: Dict[str, Any]) -> None:
        """Handle code execution event."""
        self.state.code_being_executed = data.get("code")
        self.state.current_action = "Executing code..."

        # Update progress
        self.update_progress("⚡ Executing code...")

        if self.show_code and self.show_panels and self.state.code_being_executed:
            self.console.print(
                Panel(
                    Syntax(self.state.code_being_executed, "python", theme="monokai"),
                    title="[bold yellow]⚡ Executing Code[/bold yellow]",
                    border_style="yellow",
                )
            )

    def _handle_tool_call(self, data: Dict[str, Any]) -> None:
        """Handle tool call event."""
        content = data.get("content", "")
        self.state.current_action = "Calling tool..."

        # Update progress
        self.update_progress("🔧 Calling tool...")

        # Add to current step history
        if self.state.steps_history:
            self.state.steps_history[-1]["actions"].append({"type": "tool_call", "content": content})

    def _handle_observation(self, data: Dict[str, Any]) -> None:
        """Handle observation event."""
        observation = data.get("observation", "")
        self.state.last_observation = observation

        # Update progress
        self.update_progress("💡 Processing results...")

        if self.show_observations and observation:
            # Clean up observation for display
            clean_obs = observation.replace("|", "[").strip()

            # Check if this looks like an error
            is_error = any(
                keyword in clean_obs.lower()
                for keyword in ["error", "failed", "exception", "traceback", "not found", "invalid"]
            )

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
            else:
                # Normal observation - truncate if too long
                if len(clean_obs) > 200:
                    clean_obs = clean_obs[:200] + "..."
                self.console.print(f"[dim]💡 {clean_obs}[/dim]")

        # Add to current step history
        if self.state.steps_history:
            self.state.steps_history[-1]["actions"].append({"type": "observation", "content": observation})
            self.state.steps_history[-1]["status"] = "completed"

    def _handle_final_answer(self, data: Dict[str, Any]) -> None:
        """Handle final answer event."""
        answer = data.get("answer", "")

        # Update progress
        self.update_progress("✅ Finalizing answer...")

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

        # Update progress
        self.update_progress("❌ Error occurred...")

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

        if content.strip():
            # Clean up the content and show as reasoning
            if self.show_panels:
                self.console.print(
                    Panel(
                        content.strip(),
                        title="[bold blue]🤔 Agent Reasoning[/bold blue]",
                        border_style="blue",
                        padding=(0, 1),
                    )
                )
            else:
                # In headless/no-panel mode, just print the content
                self.console.print(content.strip())

    def _handle_execution_result(self, data: Dict[str, Any]) -> None:
        """Handle code execution result event."""
        if not self.show_execution_results:
            return

        # Update progress
        self.update_progress("📊 Processing execution results...")

        content = data.get("content", "")

        if content.strip():
            # Parse execution logs and output
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

            # Display execution logs if present (always show with execution results)
            if execution_logs:
                logs_text = "\n".join(execution_logs)
                if logs_text.strip():
                    self.console.print(f"[dim]📝 {logs_text}[/dim]")

            # Display output if present and meaningful
            if output_lines:
                output_text = "\n".join(output_lines)
                # Check if output contains error information
                contains_error = any(
                    keyword in output_text.lower()
                    for keyword in ["error", "failed", "exception", "not found", "invalid", "traceback"]
                )

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

        with self.progress:
            self.task_id = self.progress.add_task("Starting agent...", total=None)
            try:
                yield
            finally:
                self.progress.stop()

    def update_progress(self, description: str) -> None:
        """Update progress description."""
        if self.progress and self.task_id is not None:
            self.progress.update(self.task_id, description=description)


@contextmanager
def custom_agent_ui(
    console: Console,
    show_code: bool = True,
    show_observations: bool = True,
    show_progress: bool = True,
    show_llm_messages: bool = False,
    show_execution_results: bool = True,
    show_execution_logs: bool = True,
    show_panels: bool = True,
) -> Generator[CustomUILogger, None, None]:
    """Context manager for custom agent UI.

    Args:
        console: Rich console instance
        show_code: Whether to display executed code
        show_observations: Whether to display tool observations
        show_progress: Whether to show progress spinner
        show_llm_messages: Whether to show LLM reasoning messages
        show_execution_results: Whether to show code execution results
        show_execution_logs: Whether to show execution logs
        show_panels: Whether to show Rich panels (borders and decorations)

    Yields:
        CustomUILogger: Logger instance to pass to the agent
    """
    ui_handler = CustomUIHandler(
        console,
        show_code=show_code,
        show_observations=show_observations,
        show_llm_messages=show_llm_messages,
        show_execution_results=show_execution_results,
        show_execution_logs=show_execution_logs,
        show_panels=show_panels,
    )
    logger = CustomUILogger(ui_handler)

    if show_progress:
        with ui_handler.progress_context():
            yield logger
    else:
        yield logger


# Convenience function for completely silent execution
def create_silent_logger() -> AgentLogger:
    """Create a completely silent logger that suppresses all output."""
    return AgentLogger(level=LogLevel.OFF, console=Console(file=open("/dev/null", "w")))
