"""Textual UI handler for chat interface."""

from typing import Any, Callable, Dict, Optional

from tsugite.events import (
    BaseEvent,
    CodeExecutionEvent,
    ErrorEvent,
    FinalAnswerEvent,
    LLMMessageEvent,
    ObservationEvent,
    StepStartEvent,
    StreamChunkEvent,
    StreamCompleteEvent,
    TaskStartEvent,
)
from tsugite.ui.base import CustomUIHandler


class TextualUIHandler(CustomUIHandler):
    """UI handler that updates Textual reactive variables via callbacks."""

    def __init__(
        self,
        on_status_change: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[str], None]] = None,
        on_stream_chunk: Optional[Callable[[str], None]] = None,
        on_stream_complete: Optional[Callable[[], None]] = None,
        on_intermediate_message: Optional[Callable[[str], None]] = None,
        on_thought_log: Optional[Callable[[str, str], None]] = None,
    ):
        """Initialize Textual UI handler.

        Args:
            on_status_change: Callback for status updates
            on_tool_call: Callback when tool is called
            on_stream_chunk: Callback for streaming chunks
            on_stream_complete: Callback when streaming completes
            on_intermediate_message: Callback for intermediate agent messages
            on_thought_log: Callback for thought log entries (type, content)
        """
        # Initialize with a no-op console (we won't use it)
        from io import StringIO

        from rich.console import Console

        # Use a StringIO console to capture any output we don't want
        super().__init__(
            console=Console(file=StringIO(), force_terminal=False),
            show_panels=False,  # Don't render panels
        )

        self.on_status_change = on_status_change
        self.on_tool_call = on_tool_call
        self.on_stream_chunk = on_stream_chunk
        self.on_stream_complete = on_stream_complete
        self.on_intermediate_message = on_intermediate_message
        self.on_thought_log = on_thought_log

        # Track tools used in current turn
        self.current_tools = []
        self._last_status = ""

    def handle_event(self, event: BaseEvent) -> None:
        """Handle UI event by calling appropriate callbacks."""
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
            elif isinstance(event, StreamChunkEvent):
                self._handle_stream_chunk(event)
            elif isinstance(event, StreamCompleteEvent):
                self._handle_stream_complete(event)

    def _update_status(self, new_status: str) -> None:
        """Update status with thread safety.

        Args:
            new_status: New status message
        """
        if new_status != self._last_status:
            self._last_status = new_status
            if self.on_status_change:
                self.on_status_change(new_status)

    def _handle_task_start(self, event: TaskStartEvent) -> None:
        """Handle task start - reset tools list."""
        self.current_tools = []
        self._update_status("Starting task...")

    def _handle_step_start(self, event: StepStartEvent) -> None:
        """Handle step start."""
        step = event.step

        # Show recovery context if recovering from error
        if event.recovering_from_error:
            status_msg = f"Step {step}: Recovering from error..."
        else:
            status_msg = f"Step {step}: Waiting for LLM response..."

        self._update_status(status_msg)

        # Log to thought log
        if self.on_thought_log:
            if event.recovering_from_error:
                self.on_thought_log("step", f"⚠️ Step {step} (recovering)")
            else:
                self.on_thought_log("step", f"Step {step}")

    def _handle_code_execution(self, event: CodeExecutionEvent) -> None:
        """Handle code execution."""
        code = event.code
        # Show preview of code being executed
        preview = code[:50] if code else "code"
        if len(code) > 50:
            preview += "..."
        self._update_status(f"Executing: {preview}")

        # Log to thought log instead of chat
        if self.on_thought_log:
            self.on_thought_log("code_execution", code)

    def _handle_observation(self, event: ObservationEvent) -> None:
        """Handle observation."""
        observation = event.observation
        self._update_status("Processing results...")

        # Log to thought log if meaningful
        if self.on_thought_log and observation.strip():
            # Clean up observation and truncate if too long
            clean_obs = observation.replace("|", "[").strip()
            if len(clean_obs) > 150:
                clean_obs = clean_obs[:150] + "..."
            self.on_thought_log("observation", clean_obs)

    def _handle_final_answer(self, event: FinalAnswerEvent) -> None:
        """Handle final answer."""
        self._update_status("Finalizing answer...")

    def _handle_error(self, event: ErrorEvent) -> None:
        """Handle error."""
        # Skip suppressible errors unless debug/verbose is enabled
        if event.suppress_from_ui and not self.show_debug_messages:
            return

        error = event.error
        self._update_status(f"Error: {error}")

    def _handle_llm_message(self, event: LLMMessageEvent) -> None:
        """Handle intermediate LLM message (thoughts from agent)."""
        content = event.content
        if content:
            # Route thoughts to thought log instead of chat
            if self.on_thought_log:
                # Clean up the content - remove any leading/trailing whitespace
                clean_content = content.strip()
                if clean_content:
                    self.on_thought_log("step", f"💭 {clean_content}")

    def _handle_stream_chunk(self, event: StreamChunkEvent) -> None:
        """Handle streaming chunk."""
        chunk = event.chunk
        self.streaming_content += chunk
        self.is_streaming = True

        self._update_status("Streaming response...")
        if self.on_stream_chunk:
            self.on_stream_chunk(chunk)

    def _handle_stream_complete(self, data: Dict[str, Any]) -> None:
        """Handle streaming complete."""
        self.is_streaming = False
        self.streaming_content = ""

        self._update_status("Ready")
        if self.on_stream_complete:
            self.on_stream_complete()

    def get_tools_used(self):
        """Get list of tools used in current turn."""
        return self.current_tools.copy()

    def clear_tools(self):
        """Clear tools list for new turn."""
        self.current_tools = []

