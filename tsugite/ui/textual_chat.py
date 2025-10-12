"""Textual-based chat UI for interactive conversations."""

import tempfile
from pathlib import Path
from typing import Optional

from rich.console import Console
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input
from textual.worker import Worker, WorkerState

from tsugite.chat import ChatManager
from tsugite.md_agents import parse_agent_file
from tsugite.ui import CustomUILogger
from tsugite.ui.textual_handler import TextualUIHandler
from tsugite.ui.widgets import MessageList, StatusBar


class ChatApp(App):
    """Textual application for chat interface."""

    CSS_PATH = "chat.tcss"
    TITLE = "Tsugite Chat"
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+d", "quit", "Quit"),
        ("escape", "quit", "Quit"),
    ]

    def __init__(
        self,
        agent_path: Path,
        model_override: Optional[str] = None,
        max_history: int = 50,
        stream: bool = False,
        show_execution_details: bool = True,
    ):
        """Initialize chat app.

        Args:
            agent_path: Path to agent markdown file
            model_override: Optional model override
            max_history: Maximum conversation history turns
            stream: Whether to stream responses
            show_execution_details: Whether to show tool calls and code execution
        """
        super().__init__()
        self.agent_path = agent_path
        self.model_override = model_override
        self.max_history = max_history
        self.stream_enabled = stream
        self.show_execution_details = show_execution_details

        # Parse agent info
        agent = parse_agent_file(agent_path)
        self.agent_name = agent.config.name or agent_path.stem
        self.model = model_override or agent.config.model or "default"

        # Chat manager will be initialized in on_mount
        self.manager: Optional[ChatManager] = None

        # UI handler for agent execution
        self.ui_handler: Optional[TextualUIHandler] = None

        # State
        self.turn_count = 0
        self.current_user_message = ""
        self.streaming_message = ""

    def compose(self) -> ComposeResult:
        """Compose the UI."""
        # Update title with agent name
        self.sub_title = f"{self.agent_name} | {self.model}"

        yield Header(show_clock=True)
        yield StatusBar()
        yield MessageList()
        yield Input(placeholder="Type your message... (Esc to quit)")
        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        # Create UI handler with callbacks to update Textual widgets
        self.ui_handler = TextualUIHandler(
            on_status_change=self._update_status,
            on_tool_call=self._add_tool,
            on_stream_chunk=self._handle_stream_chunk,
            on_stream_complete=self._handle_stream_complete,
            on_intermediate_message=self._handle_intermediate_message,
            on_execution_event=self._handle_execution_event,
        )

        # Create custom logger for agent
        console = Console()
        custom_logger = CustomUILogger(ui_handler=self.ui_handler, console=console)

        # Check agent config and override max_steps if too low for chat
        agent = parse_agent_file(self.agent_path)
        original_max_steps = agent.config.max_steps or 5
        min_chat_steps = 10
        adjusted_max_steps = None
        agent_path_to_use = self.agent_path

        if original_max_steps < min_chat_steps:
            adjusted_max_steps = 30  # Increased to 30 for better chat handling
            # Create a temporary agent file with increased max_steps
            agent_content = self.agent_path.read_text()
            # Replace max_steps in frontmatter
            import re

            agent_content = re.sub(
                r"^max_steps:\s*\d+", f"max_steps: {adjusted_max_steps}", agent_content, flags=re.MULTILINE
            )
            # Write to temp file
            temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
            temp_file.write(agent_content)
            temp_file.close()
            agent_path_to_use = Path(temp_file.name)

        # Initialize chat manager with custom logger
        self.manager = ChatManager(
            agent_path=agent_path_to_use,
            model_override=self.model_override,
            max_history=self.max_history,
            custom_logger=custom_logger,
            stream=self.stream_enabled,
        )

        # Focus the input
        self.query_one(Input).focus()

        # Show welcome message
        message_list = self.query_one(MessageList)
        message_list.add_message("status", f"💬 Chat with {self.agent_name} ({self.model})")
        message_list.add_message("status", "Type your message and press Enter to send")

        # Show max_steps adjustment notice if applicable
        if adjusted_max_steps:
            message_list.add_message(
                "status",
                f"ℹ️  Note: Agent max_steps increased from {original_max_steps} to {adjusted_max_steps} for chat mode",
            )

        message_list.add_separator()

    def _update_status(self, status: str) -> None:
        """Update status bar (called from UI handler).

        Args:
            status: Status message
        """
        # Call from main thread to update reactive variable
        self.call_from_thread(self._set_status, status)

    def _set_status(self, status: str) -> None:
        """Set status on main thread.

        Args:
            status: Status message
        """
        status_bar = self.query_one(StatusBar)
        status_bar.status = status

    def _add_tool(self, tool_name: str) -> None:
        """Add tool to list (called from UI handler).

        Args:
            tool_name: Name of tool being used
        """
        self.call_from_thread(self._append_tool, tool_name)

    def _append_tool(self, tool_name: str) -> None:
        """Append tool on main thread.

        Args:
            tool_name: Name of tool
        """
        status_bar = self.query_one(StatusBar)
        tools = list(status_bar.tools_used)
        tools.append(tool_name)
        status_bar.tools_used = tools

    def _handle_stream_chunk(self, chunk: str) -> None:
        """Handle streaming chunk.

        Args:
            chunk: Text chunk from stream
        """
        self.streaming_message += chunk
        # Update message list with streaming content
        self.call_from_thread(self._update_streaming_message)

    def _update_streaming_message(self) -> None:
        """Update message list with current streaming content."""
        # For now, we'll just accumulate - could add progressive display
        pass

    def _handle_stream_complete(self) -> None:
        """Handle stream completion."""
        # Streaming done, final message will be added by worker result
        self.streaming_message = ""

    def _handle_intermediate_message(self, content: str) -> None:
        """Handle intermediate agent message.

        Args:
            content: Message content from agent
        """
        self.call_from_thread(self._add_intermediate_message, content)

    def _add_intermediate_message(self, content: str) -> None:
        """Add intermediate message on main thread.

        Args:
            content: Message content
        """
        message_list = self.query_one(MessageList)
        # Show as agent message but slightly different styling could be added
        message_list.add_message("agent", f"[Step] {content}")

    def _handle_execution_event(self, event_type: str, content: str) -> None:
        """Handle execution event from UI handler.

        Args:
            event_type: Type of execution event (tool_call, code_execution, etc.)
            content: Event content
        """
        self.call_from_thread(self._add_execution_event, event_type, content)

    def _add_execution_event(self, event_type: str, content: str) -> None:
        """Add execution event on main thread.

        Args:
            event_type: Type of execution event
            content: Event content
        """
        if self.show_execution_details:
            message_list = self.query_one(MessageList)
            message_list.add_message(event_type, content)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user message submission.

        Args:
            event: Input submitted event
        """
        # Get the message
        message = event.value.strip()
        if not message:
            return

        # Clear the input
        event.input.clear()

        # Handle slash commands
        if message.startswith("/"):
            await self.handle_command(message)
            return

        # Add user message to display
        message_list = self.query_one(MessageList)
        message_list.add_message("user", message)

        # Update turn counter and reset tools
        self.turn_count += 1
        status_bar = self.query_one(StatusBar)
        # Don't set status here - let UI handler set it to avoid race condition
        # where "Turn X: Processing..." gets partially overwritten by "Step X: ..."
        status_bar.tools_used = []

        # Clear UI handler tools for new turn
        if self.ui_handler:
            self.ui_handler.clear_tools()

        # Store current message
        self.current_user_message = message
        self.streaming_message = ""

        # Run agent in worker thread (use lambda to pass message)
        self.run_worker(
            lambda: self._run_agent_turn(message),
            name=f"agent_turn_{self.turn_count}",
            thread=True,  # Run in thread since _run_agent_turn is synchronous
        )

    def _run_agent_turn(self, message: str) -> str:
        """Run agent turn in worker thread.

        Args:
            message: User message

        Returns:
            Agent response
        """
        if not self.manager:
            raise RuntimeError("Chat manager not initialized")
        try:
            response = self.manager.run_turn(message)
            return response
        except Exception:
            raise

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle worker state changes.

        Args:
            event: Worker state change event
        """
        if event.state == WorkerState.SUCCESS:
            # Get response from worker
            response = event.worker.result

            # Add agent response
            message_list = self.query_one(MessageList)
            if response is not None:
                message_list.add_message("agent", str(response))
            else:
                message_list.add_message("agent", "No response received")
            message_list.add_separator()

            # Update status
            status_bar = self.query_one(StatusBar)
            status_bar.status = "Ready"

            # Focus input again
            self.query_one(Input).focus()

        elif event.state == WorkerState.ERROR:
            # Show error
            message_list = self.query_one(MessageList)
            error_msg = str(event.worker.error) if event.worker.error else "Unknown error"
            message_list.add_message("status", f"❌ Error: {error_msg}")
            message_list.add_separator()

            # Update status
            status_bar = self.query_one(StatusBar)
            status_bar.status = "Ready"

            # Focus input again
            self.query_one(Input).focus()

    async def handle_command(self, command: str) -> None:
        """Handle slash commands.

        Args:
            command: Command string starting with "/"
        """
        message_list = self.query_one(MessageList)

        parts = command[1:].lower().split()
        cmd = parts[0] if parts else ""

        if cmd in ("exit", "quit", "q"):
            self.exit()

        elif cmd == "clear":
            # Clear history
            if self.manager:
                self.manager.clear_history()
            message_list.messages = []
            self.turn_count = 0
            message_list.add_message("status", "✓ History cleared")

        elif cmd == "help":
            message_list.add_message("status", "Available commands:")
            message_list.add_message("status", "  /help - Show this help")
            message_list.add_message("status", "  /clear - Clear conversation history")
            message_list.add_message("status", "  /stats - Show session statistics")
            message_list.add_message("status", "  /toggle - Toggle execution details visibility")
            message_list.add_message("status", "  /exit, /quit - Exit chat")
            message_list.add_message("status", "  Esc or Ctrl+C - Exit chat")

        elif cmd == "stats":
            if self.manager:
                stats = self.manager.get_stats()
                message_list.add_message("status", "Session Statistics:")
                message_list.add_message("status", f"  Total Turns: {stats['total_turns']}")
                tokens = stats.get("total_tokens")
                if tokens:
                    message_list.add_message("status", f"  Total Tokens: {tokens:,}")
                cost = stats.get("total_cost")
                if cost and cost > 0:
                    message_list.add_message("status", f"  Total Cost: ${cost:.4f}")
                duration = stats["session_duration"]
                if duration >= 60:
                    mins = int(duration // 60)
                    secs = int(duration % 60)
                    duration_str = f"{mins}m {secs}s"
                else:
                    duration_str = f"{duration:.0f}s"
                message_list.add_message("status", f"  Duration: {duration_str}")
            else:
                message_list.add_message("status", "❌ Chat manager not available")

        elif cmd == "toggle":
            # Toggle execution details visibility
            self.show_execution_details = not self.show_execution_details
            status = "enabled" if self.show_execution_details else "disabled"
            message_list.add_message("status", f"✓ Execution details {status}")

        else:
            message_list.add_message("status", f"❌ Unknown command: /{cmd}")
            message_list.add_message("status", "Type /help for available commands")

    async def action_quit(self) -> None:
        """Quit the application."""
        self.exit()


def run_textual_chat(
    agent_path: Path,
    model_override: Optional[str] = None,
    max_history: int = 50,
    stream: bool = False,
    show_execution_details: bool = True,
) -> None:
    """Run the Textual chat interface.

    Args:
        agent_path: Path to agent markdown file
        model_override: Optional model override
        max_history: Maximum conversation history turns
        stream: Whether to stream responses
        show_execution_details: Whether to show tool calls and code execution
    """
    app = ChatApp(
        agent_path=agent_path,
        model_override=model_override,
        max_history=max_history,
        stream=stream,
        show_execution_details=show_execution_details,
    )
    app.run()
