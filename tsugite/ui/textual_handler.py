"""Textual UI handler for chat interface."""

from typing import Any, Callable, Dict, Optional

from tsugite.ui.base import CustomUIHandler, UIEvent


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

    def handle_event(self, event: UIEvent, data: Dict[str, Any]) -> None:
        """Handle UI event by calling appropriate callbacks."""
        handlers = {
            UIEvent.TASK_START: self._handle_task_start,
            UIEvent.STEP_START: self._handle_step_start,
            UIEvent.CODE_EXECUTION: self._handle_code_execution,
            UIEvent.TOOL_CALL: self._handle_tool_call,
            UIEvent.OBSERVATION: self._handle_observation,
            UIEvent.FINAL_ANSWER: self._handle_final_answer,
            UIEvent.ERROR: self._handle_error,
            UIEvent.LLM_MESSAGE: self._handle_llm_message,
            UIEvent.STREAM_CHUNK: self._handle_stream_chunk,
            UIEvent.STREAM_COMPLETE: self._handle_stream_complete,
            UIEvent.EXECUTION_RESULT: self._handle_execution_result,
            UIEvent.SUBAGENT_START: self._handle_subagent_start,
            UIEvent.SUBAGENT_END: self._handle_subagent_end,
        }

        with self._lock:
            handler = handlers.get(event)
            if handler:
                handler(data)

    def _update_status(self, new_status: str) -> None:
        """Update status with thread safety.

        Args:
            new_status: New status message
        """
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
        status_msg = f"Step {step}: Waiting for LLM response..."
        self._update_status(status_msg)

        # Log to thought log
        if self.on_thought_log:
            self.on_thought_log("step", f"Step {step}")

    def _handle_code_execution(self, data: Dict[str, Any]) -> None:
        """Handle code execution."""
        code = data.get("code", "")
        # Show preview of code being executed
        preview = code[:50] if code else "code"
        if len(code) > 50:
            preview += "..."
        self._update_status(f"Executing: {preview}")

        # Log to thought log instead of chat
        if self.on_thought_log:
            self.on_thought_log("code_execution", code)

    def _handle_tool_call(self, data: Dict[str, Any]) -> None:
        """Handle tool call."""
        content = data.get("content", "")
        # Extract tool name from content (format: "Tool: tool_name")
        tool_name = content.split(":")[1].strip() if ":" in content else content
        self.current_tools.append(tool_name)

        self._update_status(f"Using tool: {tool_name}")
        if self.on_tool_call:
            self.on_tool_call(tool_name)

        # Log to thought log instead of chat
        if self.on_thought_log:
            self.on_thought_log("tool_call", tool_name)

    def _handle_observation(self, data: Dict[str, Any]) -> None:
        """Handle observation."""
        observation = data.get("observation", "")
        self._update_status("Processing results...")

        # Log to thought log if meaningful
        if self.on_thought_log and observation.strip():
            # Clean up observation and truncate if too long
            clean_obs = observation.replace("|", "[").strip()
            if len(clean_obs) > 150:
                clean_obs = clean_obs[:150] + "..."
            self.on_thought_log("observation", clean_obs)

    def _handle_final_answer(self, data: Dict[str, Any]) -> None:
        """Handle final answer."""
        self._update_status("Finalizing answer...")

    def _handle_error(self, data: Dict[str, Any]) -> None:
        """Handle error."""
        error = data.get("error", "Unknown error")
        self._update_status(f"Error: {error}")

    def _handle_llm_message(self, data: Dict[str, Any]) -> None:
        """Handle intermediate LLM message (thoughts from agent)."""
        content = data.get("content", "")
        if content:
            # Route thoughts to thought log instead of chat
            if self.on_thought_log:
                # Clean up the content - remove any leading/trailing whitespace
                clean_content = content.strip()
                if clean_content:
                    self.on_thought_log("step", f"💭 {clean_content}")

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

    def _handle_execution_result(self, data: Dict[str, Any]) -> None:
        """Handle execution result."""
        content = data.get("content", "")

        if self.on_thought_log and content.strip():
            # Parse execution result content
            lines = content.split("\n")
            output_lines = []

            # Extract meaningful output
            for line in lines:
                if line.startswith("Out:"):
                    output_lines.append(line[4:].strip())
                elif not line.startswith("Execution logs:") and line.strip():
                    output_lines.append(line.strip())

            if output_lines:
                result_text = "\n".join(output_lines)
                # Truncate if too long
                if len(result_text) > 200:
                    result_text = result_text[:200] + "..."
                self.on_thought_log("execution_result", result_text)

    def _handle_subagent_start(self, data: Dict[str, Any]) -> None:
        """Handle subagent start."""
        agent_name = data.get("agent_name", "unknown")
        self._update_status(f"Spawning {agent_name} agent...")

        # Log to thought log
        if self.on_thought_log:
            self.on_thought_log("subagent", f"→ Spawning {agent_name}")

    def _handle_subagent_end(self, data: Dict[str, Any]) -> None:
        """Handle subagent end."""
        agent_name = data.get("agent_name", "unknown")
        result = data.get("result", "")

        self._update_status(f"{agent_name} completed")

        # Log to thought log
        if self.on_thought_log:
            log_msg = f"✓ {agent_name} completed"
            # Add result preview if available and not too long
            if result:
                result_preview = str(result)[:100]
                if len(str(result)) > 100:
                    result_preview += "..."
                log_msg += f"\n  Result: {result_preview}"
            self.on_thought_log("subagent", log_msg)
