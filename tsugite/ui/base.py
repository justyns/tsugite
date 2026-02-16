"""Base UI handler and core UI system components."""

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax

from tsugite.events import (
    BaseEvent,
    CodeExecutionEvent,
    CostSummaryEvent,
    DebugMessageEvent,
    ErrorEvent,
    FileReadEvent,
    FinalAnswerEvent,
    InfoEvent,
    LLMMessageEvent,
    ObservationEvent,
    ReasoningContentEvent,
    ReasoningTokensEvent,
    SkillLoadedEvent,
    SkillLoadFailedEvent,
    SkillUnloadedEvent,
    StepProgressEvent,
    StepStartEvent,
    StreamChunkEvent,
    StreamCompleteEvent,
    TaskStartEvent,
    WarningEvent,
)
from tsugite.ui_context import clear_ui_context, set_ui_context


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
        show_panels: bool = True,
        show_debug_messages: bool = False,
    ):
        self.console = console
        self.state = UIState()
        self.show_code = show_code
        self.show_observations = show_observations
        self.show_llm_messages = show_llm_messages
        self.show_panels = show_panels
        self.show_debug_messages = show_debug_messages
        self.progress = None
        self.task_id = None
        self._lock = threading.Lock()

        # Streaming state
        self.streaming_content = ""
        self.is_streaming = False

        # File read event buffering (buffer until task starts)
        self.file_read_buffer = []
        self.buffer_active = True

    def _print(self, *args, **kwargs) -> None:
        """Print helper that uses progress.console when progress is active.

        This ensures output doesn't interfere with Rich's Progress Live display.
        When Progress Live is active, direct console.print() calls break rendering.
        """
        if self.progress:
            self.progress.console.print(*args, **kwargs)
        else:
            self.console.print(*args, **kwargs)

    def handle_event(self, event: BaseEvent) -> None:
        """Handle a UI event and update the display."""
        with self._lock:
            if isinstance(event, TaskStartEvent):
                self._handle_task_start(event)
            elif isinstance(event, StepStartEvent):
                self._handle_step_start(event)
            elif isinstance(event, CodeExecutionEvent):
                self._handle_code_execution(event)
            elif isinstance(event, ObservationEvent):
                self._handle_observation(event)
            elif isinstance(event, FinalAnswerEvent):
                self._handle_final_answer(event)
            elif isinstance(event, ErrorEvent):
                self._handle_error(event)
            elif isinstance(event, LLMMessageEvent):
                self._handle_llm_message(event)
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
            elif isinstance(event, SkillLoadedEvent):
                self._handle_skill_loaded(event)
            elif isinstance(event, SkillLoadFailedEvent):
                self._handle_skill_load_failed(event)
            elif isinstance(event, SkillUnloadedEvent):
                self._handle_skill_unloaded(event)
            elif isinstance(event, DebugMessageEvent):
                self._handle_debug_message(event)
            elif isinstance(event, WarningEvent):
                self._handle_warning(event)
            elif isinstance(event, StepProgressEvent):
                self._handle_step_progress(event)
            elif isinstance(event, FileReadEvent):
                self._handle_file_read(event)

            self._update_display()

    def _get_display_prefix(self) -> str:
        """Get display prefix for nested multi-step context."""
        if self.state.multistep_context:
            return "  └─ "
        return ""

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

    def _handle_task_start(self, event: TaskStartEvent) -> None:
        """Handle task start event."""
        self.state.task = event.task
        self.state.current_step = 0
        self.state.steps_history = []

        # Show task start in minimal mode (panels removed)
        # Only show full prompt with --verbose, otherwise just show model
        if self.show_debug_messages:
            self._print(f"[bold]Task:[/bold] {self.state.task}")
        self._print(f"[dim]Model: {event.model}[/dim]")
        self._print("")

        # Flush buffered file read events after Model line
        if self.buffer_active and self.file_read_buffer:
            for file_read_event in self.file_read_buffer:
                self._handle_file_read(file_read_event)
            self._print("")
            self.file_read_buffer = []
            self.buffer_active = False

    def _handle_step_start(self, event: StepStartEvent) -> None:
        """Handle step start event."""
        self.state.current_step = event.step

        prefix = self._get_display_prefix()

        # Show "Turn" for reasoning iterations
        # (workflow steps are shown separately in multistep_context)
        label = f"Turn {self.state.current_step}"

        # Update progress - show recovery context if recovering from error
        if event.recovering_from_error:
            self.update_progress(f"{prefix}⚠️  {label}: Recovering from previous error...")
        else:
            self.update_progress(f"{prefix}🤔 {label}: Waiting for LLM response...")

        # Add step to history
        self.state.steps_history.append({"step": self.state.current_step, "status": "in_progress", "actions": []})

    def _handle_code_execution(self, event: CodeExecutionEvent) -> None:
        """Handle code execution event."""
        self.state.code_being_executed = event.code

        prefix = self._get_display_prefix()
        # Update progress
        self.update_progress(f"{prefix}⚡ Executing code...")

        if self.show_code and self.state.code_being_executed:
            # Show code without panel (progress indicator already shown above)
            self._print("")
            self._print(Syntax(self.state.code_being_executed, "python", theme="monokai", background_color="default"))
            self._print("")
        elif not self.show_code:
            # Code display disabled - just show indicator
            self._print("[dim yellow]⚡ Executing code...[/dim yellow]")

    def _handle_observation(self, event: ObservationEvent) -> None:
        """Handle observation event."""
        observation = event.observation

        prefix = self._get_display_prefix()
        # Update progress
        self.update_progress(f"{prefix}💡 Processing results...")

        if observation:
            # Clean up observation for display
            clean_obs = observation.replace("|", "[").strip()

            # Check if this is a final answer
            is_final_answer = "__FINAL_ANSWER__:" in clean_obs

            # Check if this looks like an error using shared helper
            is_error = self._contains_error(clean_obs)

            # Always show observations in minimal mode (panels removed, no filtering)
            if is_error:
                # Display errors prominently in red without truncation
                self._print(f"[red]⚠️  {clean_obs}[/red]")
            elif is_final_answer:
                # Skip displaying final answer here - it will be displayed by _handle_final_answer event
                pass
            elif self.show_observations:
                # Normal observation - show with truncation if needed
                if len(clean_obs) > 500:
                    clean_obs = clean_obs[:500] + "..."
                self._print(f"[dim]💡 {clean_obs}[/dim]")

        # Add to current step history
        if self.state.steps_history:
            self.state.steps_history[-1]["actions"].append({"type": "observation", "content": observation})
            self.state.steps_history[-1]["status"] = "completed"

    def _handle_final_answer(self, event: FinalAnswerEvent) -> None:
        """Handle final answer event."""
        answer = event.answer

        prefix = self._get_display_prefix()
        self.update_progress(f"{prefix}✅ Finalizing answer...")

        self._print(Markdown(str(answer)))

    def _handle_error(self, event: ErrorEvent) -> None:
        """Handle error event."""
        # Skip suppressible errors unless debug/verbose is enabled
        if event.suppress_from_ui and not self.show_debug_messages:
            return

        error = event.error
        error_type = event.error_type or "Error"

        prefix = self._get_display_prefix()
        # Update progress
        self.update_progress(f"{prefix}❌ Error occurred...")

        # Always show errors prominently (panels removed)
        self._print(f"[bold red]⚠️  {error_type}: {error}[/bold red]")

        # Add to current step history
        if self.state.steps_history:
            self.state.steps_history[-1]["actions"].append({"type": "error", "content": error})
            self.state.steps_history[-1]["status"] = "error"

    def _handle_llm_message(self, event: LLMMessageEvent) -> None:
        """Handle LLM reasoning message event."""
        if not self.show_llm_messages:
            return

        content = event.content

        if content.strip():
            # If showing code blocks separately, strip them from reasoning to avoid duplication
            if self.show_code:
                content = self._strip_code_blocks(content)

            # Clean up the content and show as reasoning (panels removed)
            if content.strip():
                self._print(Markdown(content.strip()))

    @staticmethod
    def _strip_code_blocks(content: str) -> str:
        """Strip markdown code blocks from content.

        Args:
            content: Text content that may contain markdown code blocks

        Returns:
            Content with code blocks removed
        """
        import re

        # Remove fenced code blocks (```...```)
        content = re.sub(r"```[\s\S]*?```", "", content)
        # Remove indented code blocks (4+ spaces at line start)
        content = re.sub(r"(?m)^[ ]{4,}.*$", "", content)
        return content

    def _handle_reasoning_content(self, event: ReasoningContentEvent) -> None:
        """Handle reasoning content from reasoning models (Claude, Deepseek with exposed reasoning)."""
        content = event.content
        step = event.step

        if content and content.strip():
            prefix = self._get_display_prefix()

            # Build title with step number if available
            title_prefix = "🧠 Model Reasoning"
            if step is not None:
                title_prefix = f"🧠 Model Reasoning (Turn {step})"

            # Update progress
            self.update_progress(f"{prefix}🧠 Processing reasoning content...")

            # Truncate very long reasoning content for display (panels removed)
            max_length = 2000
            display_content = content.strip()
            if len(display_content) > max_length:
                display_content = display_content[:max_length] + "\n\n[dim]... (truncated)[/dim]"

            self._print(f"[magenta]{title_prefix}:[/magenta]")
            self._print(f"[dim magenta]{display_content}[/dim magenta]")

    def _handle_reasoning_tokens(self, event: ReasoningTokensEvent) -> None:
        """Handle reasoning token counts from models like o1/o3 that don't expose reasoning content."""
        tokens = event.tokens
        step = event.step

        if tokens:
            # Build message with turn number if available (panels removed)
            if step is not None:
                message = f"🧠 Turn {step}: Used {tokens} reasoning tokens"
            else:
                message = f"🧠 Used {tokens} reasoning tokens"

            self._print(f"[dim magenta]{message}[/dim magenta]")

    def _build_cost_summary_text(
        self,
        cost: Optional[float],
        total_tokens: Optional[int],
        reasoning_tokens: Optional[int],
        include_emojis: bool = True,
        duration_seconds: Optional[float] = None,
    ) -> Optional[str]:
        """Build cost summary text from metrics.

        Args:
            cost: Execution cost in dollars
            total_tokens: Total tokens used
            reasoning_tokens: Reasoning tokens used
            include_emojis: Whether to include emoji decorations
            duration_seconds: Execution duration in seconds

        Returns:
            Formatted summary text or None if no metrics available
        """
        if cost is None and total_tokens is None and duration_seconds is None:
            return None

        parts = []

        # Duration (show first if available)
        if duration_seconds is not None:
            duration_prefix = "⏱️  " if include_emojis else ""
            if duration_seconds < 60:
                parts.append(f"{duration_prefix}Duration: {duration_seconds:.1f}s")
            else:
                minutes = int(duration_seconds // 60)
                seconds = duration_seconds % 60
                parts.append(f"{duration_prefix}Duration: {minutes}m {seconds:.1f}s")

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

    def _handle_cost_summary(self, event: CostSummaryEvent) -> None:
        """Handle cost summary display after final answer."""
        cost = event.cost
        total_tokens = event.tokens
        reasoning_tokens = None
        duration_seconds = event.duration_seconds
        cached_tokens = event.cached_tokens
        cache_creation_tokens = event.cache_creation_input_tokens
        cache_read_tokens = event.cache_read_input_tokens

        summary_text = self._build_cost_summary_text(
            cost, total_tokens, reasoning_tokens, duration_seconds=duration_seconds
        )
        if not summary_text:
            return

        # Add cache statistics if available
        cache_parts = []
        if cached_tokens and cached_tokens > 0:
            cache_parts.append(f"💾 Cached: {cached_tokens:,} tokens")
        if cache_creation_tokens and cache_creation_tokens > 0:
            cache_parts.append(f"📝 Cache write: {cache_creation_tokens:,} tokens")
        if cache_read_tokens and cache_read_tokens > 0:
            cache_parts.append(f"📖 Cache read: {cache_read_tokens:,} tokens")

        self._print(f"[dim cyan]{summary_text}[/dim cyan]")
        if cache_parts:
            cache_summary = " | ".join(cache_parts)
            self._print(f"[dim green]{cache_summary}[/dim green]")

    def _handle_stream_chunk(self, event: StreamChunkEvent) -> None:
        """Handle streaming chunk event."""
        chunk = event.chunk
        self.streaming_content += chunk
        self.is_streaming = True

        prefix = self._get_display_prefix()
        # Update progress to show streaming
        self.update_progress(f"{prefix}💬 Streaming response...")

        # Print the chunk directly for real-time feedback
        self._print(chunk, end="", highlight=False)

    def _handle_stream_complete(self, event: StreamCompleteEvent) -> None:
        """Handle streaming complete event."""
        self.is_streaming = False

        # Print newline after streaming completes
        self._print()

        prefix = self._get_display_prefix()
        # Update progress
        self.update_progress(f"{prefix}✅ Streaming complete")

        # Clear streaming content for next step
        self.streaming_content = ""

    def _handle_info(self, event: InfoEvent) -> None:
        """Handle info event for informational messages."""
        message = event.message
        if message:
            self._print(f"[dim]{message}[/dim]")

    def _handle_skill_loaded(self, event: SkillLoadedEvent) -> None:
        """Handle skill loaded event."""
        skill_name = event.skill_name
        description = event.description
        if description:
            self._print(f"[dim]📚 Loaded skill: {skill_name} ({description})[/dim]")
        else:
            self._print(f"[dim]📚 Loaded skill: {skill_name}[/dim]")

    def _handle_skill_unloaded(self, event: SkillUnloadedEvent) -> None:
        """Handle skill unloaded event."""
        skill_name = event.skill_name
        self._print(f"[dim]📚 Unloaded skill: {skill_name}[/dim]")

    def _handle_skill_load_failed(self, event: SkillLoadFailedEvent) -> None:
        """Handle skill load failed event."""
        skill_name = event.skill_name
        error_message = event.error_message
        self._print(f"[yellow]⚠️  Failed to load skill '{skill_name}': {error_message}[/yellow]")

    def _handle_debug_message(self, event: DebugMessageEvent) -> None:
        """Handle debug message event."""
        if not self.show_debug_messages:
            return
        message = event.message
        if message:
            self._print(f"[dim blue]{message}[/dim blue]")

    def _handle_warning(self, event: WarningEvent) -> None:
        """Handle warning event."""
        message = event.message
        if message:
            self._print(f"[yellow]{message}[/yellow]")

    def _handle_step_progress(self, event: StepProgressEvent) -> None:
        """Handle step progress event."""
        message = event.message
        if message:
            prefix = self._get_display_prefix()
            self._print(f"[cyan]{prefix}{message}[/cyan]")

    def _handle_file_read(self, event: FileReadEvent) -> None:
        """Handle file read event."""
        if self.buffer_active:
            self.file_read_buffer.append(event)
            return

        from tsugite.utils import format_file_size

        size_str = format_file_size(event.byte_count)
        self._print(f"[dim]Read {event.path} ({event.line_count} lines, {size_str})[/dim]")

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
            transient=True,  # Auto-clear progress when done for cleaner output
            console=self.console,
            refresh_per_second=20,  # Higher refresh rate for real-time updates
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
