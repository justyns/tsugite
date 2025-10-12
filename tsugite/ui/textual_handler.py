"""Textual UI handler for chat interface."""

import threading
from typing import Any, Callable, Dict, Optional

from tsugite.ui.base import CustomUIHandler, UIEvent, UIState


class TextualUIHandler(CustomUIHandler):
    """UI handler that updates Textual reactive variables via callbacks."""

    def __init__(
        self,
        on_status_change: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[str], None]] = None,
        on_stream_chunk: Optional[Callable[[str], None]] = None,
        on_stream_complete: Optional[Callable[[], None]] = None,
        on_intermediate_message: Optional[Callable[[str], None]] = None,
    ):
        """Initialize Textual UI handler.

        Args:
            on_status_change: Callback for status updates
            on_tool_call: Callback when tool is called
            on_stream_chunk: Callback for streaming chunks
            on_stream_complete: Callback when streaming completes
            on_intermediate_message: Callback for intermediate agent messages
        """
        # Initialize with a no-op console (we won't use it)
        from rich.console import Console
        from io import StringIO

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

        # Track tools used in current turn
        self.current_tools = []

        # Lock for status updates to prevent corruption
        self._status_lock = threading.Lock()
        self._last_status = ""

    def handle_event(self, event: UIEvent, data: Dict[str, Any]) -> None:
        """Handle UI event by calling appropriate callbacks."""
        # Update internal state
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
            elif event == UIEvent.STREAM_CHUNK:
                self._handle_stream_chunk(data)
            elif event == UIEvent.STREAM_COMPLETE:
                self._handle_stream_complete(data)

    def _update_status(self, new_status: str) -> None:
        """Update status with thread safety.

        Args:
            new_status: New status message
        """
        with self._status_lock:
            if new_status != self._last_status:
                self._last_status = new_status
                if self.on_status_change:
                    self.on_status_change(new_status)

    def _handle_task_start(self, data: Dict[str, Any]) -> None:
        """Handle task start - reset tools list."""
        self.current_tools = []
        self._update_status("Starting task...")

    def _handle_step_start(self, data: Dict[str, Any]) -> None:
        """Handle step start."""
        step = data.get("step", 1)
        self._update_status(f"Step {step}: Waiting for LLM response...")

    def _handle_code_execution(self, data: Dict[str, Any]) -> None:
        """Handle code execution."""
        code = data.get("code", "")
        # Show preview of code being executed
        preview = code[:50] if code else "code"
        if len(code) > 50:
            preview += "..."
        self._update_status(f"Executing: {preview}")

    def _handle_tool_call(self, data: Dict[str, Any]) -> None:
        """Handle tool call."""
        content = data.get("content", "")
        # Extract tool name from content (format: "Tool: tool_name")
        tool_name = content.split(":")[1].strip() if ":" in content else content
        self.current_tools.append(tool_name)

        self._update_status(f"Using tool: {tool_name}")
        if self.on_tool_call:
            self.on_tool_call(tool_name)

    def _handle_observation(self, data: Dict[str, Any]) -> None:
        """Handle observation."""
        self._update_status("Processing results...")

    def _handle_final_answer(self, data: Dict[str, Any]) -> None:
        """Handle final answer."""
        self._update_status("Finalizing answer...")

    def _handle_error(self, data: Dict[str, Any]) -> None:
        """Handle error."""
        error = data.get("error", "Unknown error")
        self._update_status(f"Error: {error}")

    def _handle_llm_message(self, data: Dict[str, Any]) -> None:
        """Handle intermediate LLM message."""
        content = data.get("content", "")
        if content and self.on_intermediate_message:
            self.on_intermediate_message(content)

    def _handle_stream_chunk(self, data: Dict[str, Any]) -> None:
        """Handle streaming chunk."""
        chunk = data.get("chunk", "")
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
