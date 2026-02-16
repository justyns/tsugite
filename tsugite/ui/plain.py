"""Plain text UI handler without colors, panels, or emojis."""

import re
from contextlib import contextmanager
from typing import Generator

from tsugite.console import get_stderr_console
from tsugite.events import (
    CodeExecutionEvent,
    CostSummaryEvent,
    ErrorEvent,
    FileReadEvent,
    FinalAnswerEvent,
    LLMMessageEvent,
    ObservationEvent,
    ReasoningContentEvent,
    ReasoningTokensEvent,
    StepStartEvent,
    TaskStartEvent,
)
from tsugite.ui.base import CustomUIHandler
from tsugite.ui_context import clear_ui_context, set_ui_context

# Display constants for plain UI output
MAX_OBSERVATION_PREVIEW_LENGTH = 200  # Truncate long observations in plain output
MAX_REASONING_DISPLAY_LENGTH = 2000  # Limit reasoning content to keep output manageable


class PlainUIHandler(CustomUIHandler):
    """Plain text UI handler without colors, panels, animations, or emojis."""

    def __init__(self):
        """Initialize plain UI handler with no-color console."""
        no_color_console = get_stderr_console(no_color=True)

        # Initialize parent with panels disabled
        super().__init__(
            console=no_color_console,
            show_code=True,
            show_observations=True,
            show_llm_messages=False,
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

    @staticmethod
    def _is_final_answer(text: str) -> bool:
        """Check if text contains a final answer marker.

        Args:
            text: Text to check

        Returns:
            True if text contains final answer marker
        """
        return "__FINAL_ANSWER__:" in text

    def _handle_task_start(self, event: TaskStartEvent) -> None:
        """Handle task start event with plain text output."""
        self.state.task = event.task
        self.state.current_step = 0
        self.state.steps_history = []

        # Show plain text task details (main banner already shown by CLI)
        # Only show full prompt with --verbose (via show_debug_messages flag)
        if self.show_debug_messages:
            self.console.print(f"Task: {self.state.task}")
        model = event.model
        if model:
            self.console.print(f"Model: {model}")
        self.console.print()

        # Flush buffered file read events (inherited from base class)
        if self.buffer_active and self.file_read_buffer:
            for file_read_event in self.file_read_buffer:
                self._handle_file_read(file_read_event)
            self.console.print()
            self.file_read_buffer = []
            self.buffer_active = False

    def _handle_step_start(self, event: StepStartEvent) -> None:
        """Handle step start event with plain text output."""
        self.state.current_step = event.step

        # Show "Turn" for reasoning iterations
        label = f"Turn {self.state.current_step}"

        # Show recovery context if recovering from error
        if event.recovering_from_error:
            self.console.print(f"{label}: Recovering from previous error...")
        else:
            self.console.print(f"{label}: Waiting for LLM response...")

        # Add step to history
        self.state.steps_history.append({"step": self.state.current_step, "status": "in_progress", "actions": []})

    def _handle_code_execution(self, event: CodeExecutionEvent) -> None:
        """Handle code execution event with plain text output."""
        self.state.code_being_executed = event.code

        self.console.print("Executing code...")

        if self.show_code and self.state.code_being_executed:
            self.console.print()
            self.console.rule("Executing Code", style="dim")
            self.console.print(self.state.code_being_executed)
            self.console.rule(style="dim")
            self.console.print()

    def _handle_observation(self, event: ObservationEvent) -> None:
        """Handle observation event with plain text output."""
        observation = event.observation

        self.console.print("Processing results...")

        if observation:
            # Clean up observation for display
            clean_obs = observation.replace("|", "[").strip()

            # Check if this is a final answer or error
            is_final_answer = self._is_final_answer(clean_obs)
            is_error = self._contains_error(clean_obs)

            # Always show errors, even if show_observations is False
            # Skip final answers here - they will be displayed by _handle_final_answer event
            if is_error:
                # Display errors prominently
                self.console.print()
                self.console.rule("ERROR", style="dim")
                self.console.print(clean_obs)
                self.console.rule(style="dim")
                self.console.print()
            elif is_final_answer:
                # Skip displaying final answer here - it will be displayed by _handle_final_answer event
                pass
            elif self.show_observations:
                # Normal observation - truncate if too long
                if len(clean_obs) > MAX_OBSERVATION_PREVIEW_LENGTH:
                    clean_obs = clean_obs[:MAX_OBSERVATION_PREVIEW_LENGTH] + "..."
                self.console.print(f"Result: {clean_obs}")

        # Add to current step history
        if self.state.steps_history:
            self.state.steps_history[-1]["actions"].append({"type": "observation", "content": observation})
            self.state.steps_history[-1]["status"] = "completed"

    def _handle_final_answer(self, event: FinalAnswerEvent) -> None:
        """Handle final answer event with plain text output."""
        answer = str(event.answer)

        self.console.print("Finalizing answer...")

        # Render final answer as markdown
        from rich.markdown import Markdown

        self.console.print()
        self.console.rule("FINAL ANSWER")
        self.console.print(Markdown(answer))
        self.console.rule()
        self.console.print()

    def _handle_error(self, event: ErrorEvent) -> None:
        """Handle error event with plain text output."""
        # Skip suppressible errors unless debug/verbose is enabled
        if event.suppress_from_ui and not self.show_debug_messages:
            return

        error = event.error
        error_type = event.error_type or "Error"

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

    def _handle_llm_message(self, event: LLMMessageEvent) -> None:
        """Handle LLM reasoning message event with plain text output."""
        if not self.show_llm_messages:
            return

        content = event.content
        title = event.title or "Agent Reasoning"

        if content.strip():
            self.console.print()
            self.console.rule(title, style="dim")
            self.console.print(content.strip())
            self.console.rule(style="dim")
            self.console.print()

    def _handle_reasoning_content(self, event: ReasoningContentEvent) -> None:
        """Handle reasoning content with plain text output."""
        content = event.content
        step = event.step

        if content and content.strip():
            # Build title with turn number if available
            if step is not None:
                title = f"Model Reasoning (Turn {step})"
            else:
                title = "Model Reasoning"

            self.console.print("Processing reasoning content...")

            # Truncate very long reasoning content for display
            display_content = content.strip()
            if len(display_content) > MAX_REASONING_DISPLAY_LENGTH:
                display_content = display_content[:MAX_REASONING_DISPLAY_LENGTH] + "\n\n... (truncated)"

            self.console.print()
            self.console.rule(title, style="dim")
            self.console.print(display_content)
            self.console.rule(style="dim")
            self.console.print()

    def _handle_reasoning_tokens(self, event: ReasoningTokensEvent) -> None:
        """Handle reasoning token counts with plain text output."""
        tokens = event.tokens
        step = event.step

        if tokens:
            # Build message with turn number if available
            if step is not None:
                message = f"Turn {step}: Used {tokens} reasoning tokens"
            else:
                message = f"Used {tokens} reasoning tokens"

            self.console.print(message)

    def _handle_cost_summary(self, event: CostSummaryEvent) -> None:
        """Handle cost summary display after final answer."""
        cost = event.cost
        tokens = event.tokens
        duration_seconds = event.duration_seconds
        cached_tokens = event.cached_tokens
        cache_creation_tokens = event.cache_creation_input_tokens
        cache_read_tokens = event.cache_read_input_tokens

        summary_text = self._build_cost_summary_text(
            cost, tokens, None, include_emojis=False, duration_seconds=duration_seconds
        )
        if not summary_text:
            return

        # Add cache statistics if available
        cache_parts = []
        if cached_tokens and cached_tokens > 0:
            cache_parts.append(f"Cached: {cached_tokens:,} tokens")
        if cache_creation_tokens and cache_creation_tokens > 0:
            cache_parts.append(f"Cache write: {cache_creation_tokens:,} tokens")
        if cache_read_tokens and cache_read_tokens > 0:
            cache_parts.append(f"Cache read: {cache_read_tokens:,} tokens")

        if cache_parts:
            cache_summary = " | ".join(cache_parts)
            self.console.print(f"\n{summary_text}")
            self.console.print(f"{cache_summary}\n")
        else:
            self.console.print(f"\n{summary_text}\n")

    def _handle_file_read(self, event: FileReadEvent) -> None:
        """Handle file read event with plain text output."""
        from tsugite.utils import format_file_size

        size_str = format_file_size(event.byte_count)
        self.console.print(f"Read {event.path} ({event.line_count} lines, {size_str})")

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
