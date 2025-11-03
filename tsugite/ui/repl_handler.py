"""REPL UI handler for simple, interactive chat sessions."""

import threading
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.syntax import Syntax

from tsugite.events import (
    BaseEvent,
    CodeExecutionEvent,
    CostSummaryEvent,
    DebugMessageEvent,
    ErrorEvent,
    ExecutionLogsEvent,
    ExecutionResultEvent,
    FinalAnswerEvent,
    InfoEvent,
    LLMMessageEvent,
    ObservationEvent,
    ReasoningContentEvent,
    ReasoningTokensEvent,
    StepProgressEvent,
    StepStartEvent,
    StreamChunkEvent,
    StreamCompleteEvent,
    TaskStartEvent,
    ToolCallEvent,
    WarningEvent,
)


class ReplUIHandler:
    """Simplified UI handler for REPL chat mode.

    Uses rich for beautiful output while keeping things minimal and readable.
    Designed for interactive chat sessions where simplicity and clarity are key.
    """

    def __init__(
        self,
        console: Console,
        show_code: bool = False,
        show_observations: bool = True,
        show_llm_messages: bool = False,
        show_debug_messages: bool = False,
        compact: bool = True,
    ):
        """Initialize REPL UI handler.

        Args:
            console: Rich console for output
            show_code: Show code execution blocks
            show_observations: Show tool observations
            show_llm_messages: Show LLM reasoning messages
            show_debug_messages: Show debug messages
            compact: Use compact mode (minimal output)
        """
        self.console = console
        self.show_code = show_code
        self.show_observations = show_observations
        self.show_llm_messages = show_llm_messages
        self.show_debug_messages = show_debug_messages
        self.compact = compact
        self._lock = threading.Lock()

        # State
        self.current_status: Optional[Status] = None
        self.streaming_content = ""
        self.is_streaming = False
        self.current_step = 0

    def handle_event(self, event: BaseEvent) -> None:
        """Handle a UI event and update the display."""
        with self._lock:
            if isinstance(event, TaskStartEvent):
                self._handle_task_start(event)
            elif isinstance(event, StepStartEvent):
                self._handle_step_start(event)
            elif isinstance(event, CodeExecutionEvent):
                self._handle_code_execution(event)
            elif isinstance(event, ToolCallEvent):
                self._handle_tool_call(event)
            elif isinstance(event, ObservationEvent):
                self._handle_observation(event)
            elif isinstance(event, FinalAnswerEvent):
                self._handle_final_answer(event)
            elif isinstance(event, ErrorEvent):
                self._handle_error(event)
            elif isinstance(event, LLMMessageEvent):
                self._handle_llm_message(event)
            elif isinstance(event, ExecutionResultEvent):
                self._handle_execution_result(event)
            elif isinstance(event, ExecutionLogsEvent):
                self._handle_execution_logs(event)
            elif isinstance(event, ReasoningContentEvent):
                self._handle_reasoning_content(event)
            elif isinstance(event, ReasoningTokensEvent):
                self._handle_reasoning_tokens(event)
            elif isinstance(event, CostSummaryEvent):
                self._handle_cost_summary(event)
            elif isinstance(event, StreamChunkEvent):
                self._handle_stream_chunk(event)
            elif isinstance(event, StreamCompleteEvent):
                self._handle_stream_complete(event)
            elif isinstance(event, InfoEvent):
                self._handle_info(event)
            elif isinstance(event, DebugMessageEvent):
                self._handle_debug_message(event)
            elif isinstance(event, WarningEvent):
                self._handle_warning(event)
            elif isinstance(event, StepProgressEvent):
                self._handle_step_progress(event)

    def _handle_task_start(self, event: TaskStartEvent) -> None:
        """Handle task start - show model in compact mode."""
        if self.show_debug_messages:
            self.console.print(f"[dim]Model: {event.model}[/dim]")

    def _handle_step_start(self, event: StepStartEvent) -> None:
        """Handle step start - show spinner during thinking."""
        self.current_step = event.step

        # Stop previous status if any
        if self.current_status:
            self.current_status.stop()

        # Show spinner while agent is thinking
        status_msg = f"Turn {event.step}: Thinking..."
        if event.recovering_from_error:
            status_msg = f"Turn {event.step}: Recovering from error..."

        self.current_status = self.console.status(status_msg, spinner="dots")
        self.current_status.start()

    def _handle_code_execution(self, event: CodeExecutionEvent) -> None:
        """Handle code execution - optionally show code."""
        if self.show_code and event.code:
            # Stop status spinner for code display
            if self.current_status:
                self.current_status.stop()
                self.current_status = None

            self.console.print()
            self.console.print(Syntax(event.code, "python", theme="monokai", background_color="default"))
            self.console.print()
        elif self.compact:
            # Just show a minimal indicator
            pass  # Spinner already shows activity

    def _handle_tool_call(self, event: ToolCallEvent) -> None:
        """Handle tool call - show compact tool name."""
        if self.compact:
            tool_name = event.tool.replace("Tool: ", "").strip() if event.tool else "unknown"
            self.console.print(f"[dim cyan]🔧 {tool_name}[/dim cyan]")

    def _handle_observation(self, event: ObservationEvent) -> None:
        """Handle observation - show if enabled."""
        if not self.show_observations or not event.observation:
            return

        observation = event.observation.strip()
        if not observation:
            return

        # Skip final answer observations (handled separately)
        if "__FINAL_ANSWER__:" in observation:
            return

        # Check if error
        is_error = self._contains_error(observation)

        if is_error:
            self.console.print(f"[red]⚠️  {observation}[/red]")
        elif len(observation) > 500:
            self.console.print(f"[dim]💡 {observation[:500]}...[/dim]")
        else:
            self.console.print(f"[dim]💡 {observation}[/dim]")

    def _handle_final_answer(self, event: FinalAnswerEvent) -> None:
        """Handle final answer - render with rich markdown."""
        # Stop status spinner
        if self.current_status:
            self.current_status.stop()
            self.current_status = None

        # Render the answer beautifully
        self.console.print()
        self.console.print(Markdown(str(event.answer)))
        self.console.print()

    def _handle_error(self, event: ErrorEvent) -> None:
        """Handle error - show with panel for prominence."""
        # Skip suppressible errors unless debug is enabled
        if event.suppress_from_ui and not self.show_debug_messages:
            return

        # Stop status spinner
        if self.current_status:
            self.current_status.stop()
            self.current_status = None

        error_type = event.error_type or "Error"

        # Show error in panel for visibility
        error_panel = Panel(
            f"[red]{event.error}[/red]",
            title=f"⚠️  {error_type}",
            border_style="red",
        )
        self.console.print(error_panel)

    def _handle_llm_message(self, event: LLMMessageEvent) -> None:
        """Handle LLM message - show if enabled."""
        if not self.show_llm_messages or not event.content:
            return

        content = event.content.strip()
        if content:
            # Stop status for message display
            if self.current_status:
                self.current_status.stop()
                self.current_status = None

            self.console.print(Markdown(content))

    def _handle_execution_result(self, event: ExecutionResultEvent) -> None:
        """Handle execution result."""
        # Stop status spinner
        if self.current_status:
            self.current_status.stop()
            self.current_status = None

        # Show logs if present
        if event.logs:
            logs_text = "\n".join(event.logs)
            if logs_text.strip():
                self.console.print(f"[dim]📝 {logs_text}[/dim]")

        # Show output if meaningful
        if event.output:
            output_text = event.output.strip()
            if "__FINAL_ANSWER__:" in output_text:
                return  # Skip, handled by final answer event

            is_error = self._contains_error(output_text)
            if is_error:
                self.console.print(f"[red]📤 {output_text}[/red]")
            elif output_text and output_text.lower() not in ("none", "null", ""):
                self.console.print(f"[cyan]📤 {output_text}[/cyan]")

    def _handle_execution_logs(self, event: ExecutionLogsEvent) -> None:
        """Handle execution logs."""
        if not event.logs:
            return

        content = event.logs.strip()
        if "Execution logs:" in content:
            logs = content.replace("Execution logs:", "").strip()
            if logs:
                self.console.print(f"[dim]📝 {logs}[/dim]")

    def _handle_reasoning_content(self, event: ReasoningContentEvent) -> None:
        """Handle reasoning content from models that expose it."""
        if not event.content:
            return

        content = event.content.strip()
        if not content:
            return

        # Stop status for reasoning display
        if self.current_status:
            self.current_status.stop()
            self.current_status = None

        # Truncate if very long
        max_length = 2000
        if len(content) > max_length:
            content = content[:max_length] + "\n\n[dim]... (truncated)[/dim]"

        title = "🧠 Model Reasoning"
        if event.step is not None:
            title = f"🧠 Model Reasoning (Turn {event.step})"

        self.console.print(f"[magenta]{title}:[/magenta]")
        self.console.print(f"[dim magenta]{content}[/dim magenta]")

    def _handle_reasoning_tokens(self, event: ReasoningTokensEvent) -> None:
        """Handle reasoning token counts."""
        if not event.tokens:
            return

        if event.step is not None:
            message = f"🧠 Turn {event.step}: Used {event.tokens:,} reasoning tokens"
        else:
            message = f"🧠 Used {event.tokens:,} reasoning tokens"

        self.console.print(f"[dim magenta]{message}[/dim magenta]")

    def _handle_cost_summary(self, event: CostSummaryEvent) -> None:
        """Handle cost summary - show compact stats."""
        parts = []

        # Duration
        if event.duration_seconds is not None:
            if event.duration_seconds < 60:
                parts.append(f"⏱️  {event.duration_seconds:.1f}s")
            else:
                minutes = int(event.duration_seconds // 60)
                seconds = event.duration_seconds % 60
                parts.append(f"⏱️  {minutes}m {seconds:.1f}s")

        # Cost
        if event.cost is not None and event.cost > 0:
            parts.append(f"💰 ${event.cost:.6f}")

        # Tokens
        if event.tokens is not None:
            parts.append(f"📊 {event.tokens:,} tokens")

        if parts:
            summary = " | ".join(parts)
            self.console.print(f"[dim cyan]{summary}[/dim cyan]")

        # Cache stats
        cache_parts = []
        if event.cached_tokens and event.cached_tokens > 0:
            cache_parts.append(f"💾 {event.cached_tokens:,} cached")
        if event.cache_read_input_tokens and event.cache_read_input_tokens > 0:
            cache_parts.append(f"📖 {event.cache_read_input_tokens:,} cache reads")

        if cache_parts:
            cache_summary = " | ".join(cache_parts)
            self.console.print(f"[dim green]{cache_summary}[/dim green]")

    def _handle_stream_chunk(self, event: StreamChunkEvent) -> None:
        """Handle streaming chunk - accumulate but don't print.

        In REPL mode, we don't want to show the raw LLM response streaming
        (which includes Thought/Code formatting). Instead, we let structured
        events (FinalAnswerEvent, ToolCallEvent, etc.) handle the display.
        """
        self.streaming_content += event.chunk
        self.is_streaming = True

        # Don't print chunks - let structured events handle display

    def _handle_stream_complete(self, event: StreamCompleteEvent) -> None:
        """Handle streaming complete."""
        self.is_streaming = False
        self.console.print()  # Newline after streaming
        self.streaming_content = ""

    def _handle_info(self, event: InfoEvent) -> None:
        """Handle info message."""
        if event.message:
            self.console.print(f"[dim]{event.message}[/dim]")

    def _handle_debug_message(self, event: DebugMessageEvent) -> None:
        """Handle debug message."""
        if self.show_debug_messages and event.message:
            self.console.print(f"[dim blue]{event.message}[/dim blue]")

    def _handle_warning(self, event: WarningEvent) -> None:
        """Handle warning."""
        if event.message:
            self.console.print(f"[yellow]{event.message}[/yellow]")

    def _handle_step_progress(self, event: StepProgressEvent) -> None:
        """Handle step progress."""
        if event.message:
            self.console.print(f"[cyan]{event.message}[/cyan]")

    @staticmethod
    def _contains_error(text: str) -> bool:
        """Check if text contains error keywords."""
        error_keywords = ["error", "failed", "exception", "not found", "invalid", "traceback"]
        return any(keyword in text.lower() for keyword in error_keywords)

    def stop(self) -> None:
        """Stop any active status display."""
        if self.current_status:
            self.current_status.stop()
            self.current_status = None
