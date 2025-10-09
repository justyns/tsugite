"""Chat UI handler for interactive conversations."""

from typing import List, Optional

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner

from tsugite.ui.base import CustomUIHandler, UIEvent


class ChatUIHandler(CustomUIHandler):
    """Prettier UI handler for chat mode with live updates."""

    def __init__(self, console: Console):
        super().__init__(
            console=console,
            show_code=False,
            show_observations=False,
            show_llm_messages=False,
            show_execution_results=False,
            show_execution_logs=False,
            show_panels=False,
        )
        self.live_display: Optional[Live] = None
        self.tool_actions: List[dict] = []
        self.is_thinking = False
        self.current_tool = None

    def handle_event(self, event: UIEvent, data: dict) -> None:
        """Handle UI events for chat mode with prettier output."""
        # Debug: print all events (uncomment to debug)
        # self.console.print(f"[dim]DEBUG: {event.name} - {data}[/dim]")

        if event == UIEvent.TASK_START:
            self.tool_actions = []
            self.is_thinking = True
            self._show_spinner("Thinking...")

        elif event == UIEvent.STEP_START:
            if not self.is_thinking:
                self.is_thinking = True
                self._show_spinner("Processing...")

        elif event == UIEvent.TOOL_CALL:
            content = data.get("content", "")
            if "Calling tool:" in content:
                tool_name = content.split("Calling tool:")[1].split("with")[0].strip()

                # Skip only final_answer from display (it's implicit in the response)
                if tool_name != "final_answer":
                    self.current_tool = {"tool": tool_name, "args": None, "result": None}
                    self._show_spinner(f"Using {tool_name}...")

        elif event == UIEvent.CODE_EXECUTION:
            # Capture code being executed
            code = data.get("code", "")
            if code and "final_answer" not in code.lower():
                # Store code execution (unless it's just final_answer)
                self.current_tool = {"action": "code", "code": code}
                self._show_spinner("Executing code...")

        elif event == UIEvent.EXECUTION_RESULT:
            # Capture execution result
            content = data.get("content", "")
            if self.current_tool and content:
                # Store the execution result
                self.current_tool["result"] = content
                self.tool_actions.append(self.current_tool)
                self.current_tool = None

        elif event == UIEvent.FINAL_ANSWER:
            self._stop_spinner()
            self.is_thinking = False

        elif event == UIEvent.ERROR:
            self._stop_spinner()
            self.is_thinking = False

    def _show_spinner(self, message: str):
        """Show a spinner with message."""
        if self.live_display is None:
            spinner = Spinner("dots", text=f"[dim]{message}[/dim]")
            self.live_display = Live(spinner, console=self.console, refresh_per_second=10)
            self.live_display.start()
        else:
            # Update existing spinner
            self.live_display.update(Spinner("dots", text=f"[dim]{message}[/dim]"))

    def _stop_spinner(self):
        """Stop the spinner."""
        if self.live_display:
            self.live_display.stop()
            self.live_display = None
