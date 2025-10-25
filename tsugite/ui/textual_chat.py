"""Textual-based chat UI for interactive conversations."""

from pathlib import Path
from typing import Optional

import litellm
from rich.console import Console
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Input
from textual.worker import Worker, WorkerState
from textual_autocomplete import AutoComplete, DropdownItem, TargetState

from tsugite.config import get_chat_theme
from tsugite.md_agents import parse_agent_file
from tsugite.ui import CustomUILogger
from tsugite.ui.chat import ChatManager
from tsugite.ui.textual_handler import TextualUIHandler
from tsugite.ui.widgets import MessageList, ThoughtLog

# Slash commands for autocomplete
SLASH_COMMANDS = [
    "/help",
    "/clear",
    "/stats",
    "/toggle",
    "/markdown",
    "/exit",
    "/quit",
]


class ChatApp(App):
    """Textual application for chat interface."""

    CSS_PATH = "chat.tcss"
    TITLE = "Tsugite Chat"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
        Binding("ctrl+d", "quit", "Quit", show=False, priority=True),
        Binding("escape", "quit", "Quit", show=True, priority=True),
        Binding("ctrl+n", "focus_next", "Next Pane", show=True, priority=True),
        Binding("ctrl+k", "toggle_markdown", "Markdown", show=True, priority=True),
    ]

    def __init__(
        self,
        agent_path: Path,
        model_override: Optional[str] = None,
        max_history: int = 50,
        stream: bool = False,
        show_execution_details: bool = True,
        disable_history: bool = False,
    ):
        """Initialize chat app.

        Args:
            agent_path: Path to agent markdown file
            model_override: Optional model override
            max_history: Maximum conversation history turns
            stream: Whether to stream responses
            show_execution_details: Whether to show tool calls and code execution
            disable_history: Disable conversation history persistence
        """
        super().__init__()
        self.agent_path = agent_path
        self.model_override = model_override
        self.max_history = max_history
        self.stream_enabled = stream
        self.show_execution_details = show_execution_details
        self.disable_history = disable_history

        # Parse agent info
        agent = parse_agent_file(agent_path)
        self.agent_name = agent.config.name or agent_path.stem
        self.model = model_override or agent.config.model or "default"

        # Load theme from config
        self.chat_theme = get_chat_theme()

        # Chat manager will be initialized in on_mount
        self.manager: Optional[ChatManager] = None

        # UI handler for agent execution
        self.ui_handler: Optional[TextualUIHandler] = None

        # State
        self.turn_count = 0
        self.current_user_message = ""
        self.streaming_message = ""

        # Token and cost tracking
        self.total_tokens = 0
        self.total_cost = 0.0

    def compose(self) -> ComposeResult:
        """Compose the UI."""
        # Initial subtitle with agent name
        self._update_subtitle()

        yield Header(show_clock=True)
        yield ThoughtLog()
        yield MessageList()

        # Create input widget
        input_widget = Input(placeholder="Type your message... (Esc to quit)")
        yield input_widget

        # Add autocomplete for slash commands only
        yield AutoComplete(input_widget, candidates=self._get_autocomplete_candidates)

        yield Footer()

    def _get_autocomplete_candidates(self, state: TargetState) -> list[DropdownItem]:
        """Get autocomplete candidates - only show for slash commands.

        Args:
            state: Current target state containing input text

        Returns:
            List of dropdown items for slash commands
        """
        # Get current input text
        value = state.text

        # Only show suggestions if input starts with "/"
        if not value.startswith("/"):
            return []

        # Get matching slash commands
        matches = [cmd for cmd in SLASH_COMMANDS if cmd.startswith(value.lower())]
        return [DropdownItem(cmd) for cmd in matches]

    def _get_model_context_limit(self) -> Optional[int]:
        """Get context limit for the current model from LiteLLM's database.

        Returns:
            Context limit in tokens, or None if unknown
        """
        model_name = self.model.split(":")[-1] if ":" in self.model else self.model

        # Try exact match in LiteLLM's model database
        model_info = litellm.model_cost.get(model_name)
        if model_info and "max_input_tokens" in model_info:
            return model_info["max_input_tokens"]

        # Try fuzzy match in LiteLLM (e.g., "claude-3-5-sonnet" matches "claude-3-5-sonnet-20241022")
        model_lower = model_name.lower()
        for litellm_model, info in litellm.model_cost.items():
            if model_lower in litellm_model.lower() and "max_input_tokens" in info:
                # Skip image generation models
                if not litellm_model.startswith(("1024", "256", "512")):
                    return info["max_input_tokens"]

        return None

    def _update_subtitle(self) -> None:
        """Update header subtitle with token/cost info."""
        parts = [f"{self.agent_name} | {self.model}"]

        if self.total_tokens > 0:
            # Get context limit if available
            context_limit = self._get_model_context_limit()

            if context_limit:
                # Calculate percentage used
                usage_pct = (self.total_tokens / context_limit) * 100

                # Format with warning if getting close to limit
                if usage_pct >= 90:
                    token_str = f"🔢 {self.total_tokens:,}/{context_limit:,} (⚠️ {usage_pct:.0f}%)"
                elif usage_pct >= 75:
                    token_str = f"🔢 {self.total_tokens:,}/{context_limit:,} ({usage_pct:.0f}%)"
                else:
                    token_str = f"🔢 {self.total_tokens:,}/{context_limit:,}"
            else:
                # No limit known, just show total
                token_str = f"🔢 {self.total_tokens:,} tokens"

            parts.append(token_str)

        if self.total_cost > 0:
            # Show cost with 4 decimal places
            parts.append(f"💰 ${self.total_cost:.4f}")

        self.sub_title = " | ".join(parts)

    def on_mount(self) -> None:
        """Called when app is mounted."""
        # Apply theme from config (built-in themes are already registered)
        self.theme = self.chat_theme

        # Create UI handler with callbacks to update Textual widgets
        self.ui_handler = TextualUIHandler(
            on_status_change=self._update_status,
            on_tool_call=self._add_tool,
            on_stream_chunk=self._handle_stream_chunk,
            on_stream_complete=self._handle_stream_complete,
            on_intermediate_message=self._handle_intermediate_message,
            on_thought_log=self._handle_thought_log,
        )

        # Create custom logger for agent
        console = Console()
        custom_logger = CustomUILogger(ui_handler=self.ui_handler, console=console)

        # Initialize chat manager with custom logger
        self.manager = ChatManager(
            agent_path=self.agent_path,
            model_override=self.model_override,
            max_history=self.max_history,
            custom_logger=custom_logger,
            stream=self.stream_enabled,
            disable_history=self.disable_history,
        )

        # Focus the input
        self.query_one(Input).focus()

        # Show welcome message
        message_list = self.query_one(MessageList)
        message_list.add_message("status", f"💬 Chat with {self.agent_name} ({self.model})")
        message_list.add_message("status", "Type your message and press Enter to send")
        message_list.add_message(
            "status", "💡 Tip: Type / to see command dropdown (↑↓ to navigate, Tab/Enter to select)"
        )
        message_list.add_message("status", "Type /help for all commands")
        message_list.add_separator()

    def _update_status(self, status: str) -> None:
        """Update status (called from UI handler).

        Args:
            status: Status message
        """
        # Log status to thought log instead of status bar
        self.call_from_thread(self._add_thought_entry, "status", status)

    def _add_tool(self, tool_name: str) -> None:
        """Add tool to thought log (called from UI handler).

        Args:
            tool_name: Name of tool being used
        """
        # This is handled by thought log callback now
        pass

    def _handle_thought_log(self, entry_type: str, content: str) -> None:
        """Handle thought log entry from UI handler.

        Args:
            entry_type: Type of thought entry (step, tool_call, code_execution, etc.)
            content: Entry content
        """
        self.call_from_thread(self._add_thought_entry, entry_type, content)

    def _add_thought_entry(self, entry_type: str, content: str) -> None:
        """Add entry to thought log on main thread.

        Args:
            entry_type: Type of entry
            content: Entry content
        """
        thought_log = self.query_one(ThoughtLog)
        thought_log.add_entry(entry_type, content)

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

        # Update turn counter
        self.turn_count += 1

        # Add separator in thought log for new turn
        thought_log = self.query_one(ThoughtLog)
        thought_log.add_entry("status", f"─── Turn {self.turn_count} ───")

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

            # Add agent response (clean it first - remove Thought: lines)
            message_list = self.query_one(MessageList)
            if response is not None:
                clean_response = self._clean_response(str(response))
                if clean_response:
                    message_list.add_message("agent", clean_response)
                else:
                    message_list.add_message("agent", "No response received")
            else:
                message_list.add_message("agent", "No response received")
            message_list.add_separator()

            # Update token and cost tracking from manager
            if self.manager:
                stats = self.manager.get_stats()
                self.total_tokens = stats.get("total_tokens") or 0
                self.total_cost = stats.get("total_cost") or 0.0
                self._update_subtitle()

                # Check for context limit warnings
                context_limit = self._get_model_context_limit()
                if context_limit and self.total_tokens > 0:
                    usage_pct = (self.total_tokens / context_limit) * 100
                    if usage_pct >= 90:
                        thought_log = self.query_one(ThoughtLog)
                        thought_log.add_entry(
                            "error",
                            f"⚠️ Context usage at {usage_pct:.0f}%! Consider using /clear to reset history.",
                        )
                    elif usage_pct >= 75:
                        thought_log = self.query_one(ThoughtLog)
                        thought_log.add_entry(
                            "status",
                            f"Context usage at {usage_pct:.0f}% ({self.total_tokens:,}/{context_limit:,} tokens)",
                        )

            # Update thought log
            thought_log = self.query_one(ThoughtLog)
            thought_log.add_entry("status", "✓ Turn complete")

            # Focus input again
            self.query_one(Input).focus()

        elif event.state == WorkerState.ERROR:
            # Show error
            message_list = self.query_one(MessageList)
            error_msg = str(event.worker.error) if event.worker.error else "Unknown error"
            message_list.add_message("status", f"❌ Error: {error_msg}")
            message_list.add_separator()

            # Update thought log
            thought_log = self.query_one(ThoughtLog)
            thought_log.add_entry("error", f"Error: {error_msg}")

            # Focus input again
            self.query_one(Input).focus()

    def _clean_response(self, response: str) -> str:
        """Remove Thought: lines from response - those go to thought log only.

        Args:
            response: Raw agent response

        Returns:
            Cleaned response without Thought: lines
        """
        cleaned_lines = (line for line in response.split("\n") if not line.strip().lower().startswith("thought:"))
        return "\n".join(cleaned_lines).strip()

    async def _cmd_exit(self, message_list: MessageList) -> None:
        """Exit the application."""
        self.exit()

    async def _cmd_clear(self, message_list: MessageList) -> None:
        """Clear conversation history."""
        if self.manager:
            self.manager.clear_history()
        message_list.messages = []
        thought_log = self.query_one(ThoughtLog)
        thought_log.clear_log()
        self.turn_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self._update_subtitle()
        message_list.add_message("status", "✓ History cleared")

    async def _cmd_help(self, message_list: MessageList) -> None:
        """Show help message."""
        message_list.add_message("status", "Available commands:")
        message_list.add_message("status", "  /help - Show this help")
        message_list.add_message("status", "  /clear - Clear conversation history")
        message_list.add_message("status", "  /stats - Show session statistics")
        message_list.add_message("status", "  /toggle - Toggle thought log visibility")
        message_list.add_message("status", "  /markdown - Toggle markdown rendering for agent responses")
        message_list.add_message("status", "  /exit, /quit - Exit chat")
        message_list.add_message("status", "")
        message_list.add_message("status", "Navigation:")
        message_list.add_message("status", "  When dropdown visible: ↑↓ navigate, Tab/Enter select, Esc dismiss")
        message_list.add_message("status", "  Ctrl+N - Cycle through panes (Thought Log → Messages → Input)")
        message_list.add_message("status", "  Ctrl+P - Command palette")
        message_list.add_message("status", "  ↑↓ - Scroll focused pane (when not in input)")
        message_list.add_message("status", "  Esc - Exit chat (or dismiss dropdown if open)")

    async def _cmd_stats(self, message_list: MessageList) -> None:
        """Show session statistics."""
        if not self.manager:
            message_list.add_message("status", "❌ Chat manager not available")
            return

        stats = self.manager.get_stats()
        message_list.add_message("status", "Session Statistics:")
        message_list.add_message("status", f"  Total Turns: {stats['total_turns']}")

        tokens = stats.get("total_tokens")
        if tokens:
            context_limit = self._get_model_context_limit()
            if context_limit:
                usage_pct = (tokens / context_limit) * 100
                message_list.add_message("status", f"  Total Tokens: {tokens:,} / {context_limit:,} ({usage_pct:.1f}%)")
            else:
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

    async def _cmd_toggle(self, message_list: MessageList) -> None:
        """Toggle thought log visibility."""
        thought_log = self.query_one(ThoughtLog)
        current_display = thought_log.styles.display
        if current_display == "none":
            thought_log.styles.display = "block"
            message_list.add_message("status", "✓ Thought log enabled")
        else:
            thought_log.styles.display = "none"
            message_list.add_message("status", "✓ Thought log disabled")

    async def _cmd_markdown(self, message_list: MessageList) -> None:
        """Toggle markdown rendering."""
        new_state = message_list.toggle_markdown()
        if new_state:
            message_list.add_message("status", "✓ Markdown rendering enabled")
        else:
            message_list.add_message("status", "✓ Markdown rendering disabled (raw view)")

    async def handle_command(self, command: str) -> None:
        """Handle slash commands.

        Args:
            command: Command string starting with "/"
        """
        message_list = self.query_one(MessageList)
        parts = command[1:].lower().split()
        cmd = parts[0] if parts else ""

        command_handlers = {
            "exit": self._cmd_exit,
            "quit": self._cmd_exit,
            "q": self._cmd_exit,
            "clear": self._cmd_clear,
            "help": self._cmd_help,
            "stats": self._cmd_stats,
            "toggle": self._cmd_toggle,
            "markdown": self._cmd_markdown,
        }

        handler = command_handlers.get(cmd)
        if handler:
            await handler(message_list)
        else:
            message_list.add_message("status", f"❌ Unknown command: /{cmd}")
            message_list.add_message("status", "Type /help for available commands")

    async def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    async def action_toggle_markdown(self) -> None:
        """Toggle markdown rendering mode."""
        message_list = self.query_one(MessageList)
        new_state = message_list.toggle_markdown()

        # Show status message
        if new_state:
            message_list.add_message("status", "✓ Markdown rendering enabled")
        else:
            message_list.add_message("status", "✓ Markdown rendering disabled (raw view)")


def run_textual_chat(
    agent_path: Path,
    model_override: Optional[str] = None,
    max_history: int = 50,
    stream: bool = False,
    show_execution_details: bool = True,
    disable_history: bool = False,
) -> None:
    """Run the Textual chat interface.

    Args:
        agent_path: Path to agent markdown file
        model_override: Optional model override
        max_history: Maximum conversation history turns
        stream: Whether to stream responses
        show_execution_details: Whether to show tool calls and code execution
        disable_history: Disable conversation history persistence
    """
    app = ChatApp(
        agent_path=agent_path,
        model_override=model_override,
        max_history=max_history,
        stream=stream,
        show_execution_details=show_execution_details,
        disable_history=disable_history,
    )
    app.run()
