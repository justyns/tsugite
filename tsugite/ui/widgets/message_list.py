"""Scrollable message history widget for chat UI."""

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.text import Text
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widgets import Static

from .base_scroll_log import BaseScrollLog


class Message(Static):
    """Individual message display widget."""

    def __init__(
        self,
        sender: str,
        content: str,
        sender_style: str = "bold",
        content_style: str = "",
        render_markdown: bool = False,
    ):
        """Initialize message widget.

        Args:
            sender: Sender label (e.g., "You", "Agent")
            content: Message content
            sender_style: Rich style for sender label
            content_style: Rich style for content
            render_markdown: Whether to render content as markdown
        """
        super().__init__(markup=False)
        self.sender = sender
        self._message_content = content
        self.sender_style = sender_style
        self.content_style = content_style
        self.render_markdown = render_markdown

    def render(self) -> RenderableType:
        """Render the message with styling."""
        # Create sender label
        sender_text = Text()
        sender_text.append(f"{self.sender}: ", style=self.sender_style)

        # Render content based on mode
        if self.render_markdown:
            # Use markdown rendering
            content_rendered = Markdown(str(self._message_content), code_theme="monokai", inline_code_theme="monokai")
            # Group sender and markdown content together
            return Group(sender_text, content_rendered)
        else:
            # Use plain text rendering
            sender_text.append(str(self._message_content), style=self.content_style)
            return sender_text


class MessageList(BaseScrollLog):
    """Scrollable container for chat messages."""

    messages = reactive([], recompose=True)
    markdown_mode = reactive(True, recompose=True)

    def __init__(self):
        """Initialize message list."""
        super().__init__()

    def compose(self) -> ComposeResult:
        """Compose the message list from reactive messages."""
        for msg in self.messages:
            if msg["type"] == "user":
                yield Message("You", msg["content"], sender_style="bold #b8bb26", content_style="#b8bb26")
            elif msg["type"] == "agent":
                # Render agent messages with markdown if enabled
                yield Message(
                    "Agent",
                    msg["content"],
                    sender_style="bold #83a598",
                    content_style="#83a598",
                    render_markdown=self.markdown_mode,
                )
            elif msg["type"] == "tool_call":
                yield Message("🔧 Tool", msg["content"], sender_style="#fabd2f", content_style="dim #fabd2f")
            elif msg["type"] == "code_execution":
                yield Message("⚡ Code", msg["content"], sender_style="#d3869b", content_style="dim")
            elif msg["type"] == "execution_result":
                yield Message("📤 Result", msg["content"], sender_style="#8ec07c", content_style="dim #8ec07c")
            elif msg["type"] == "observation":
                yield Message("💡 Info", msg["content"], sender_style="#83a598", content_style="dim #83a598")
            elif msg["type"] == "separator":
                yield Static("─" * 40, classes="separator", markup=False)
            elif msg["type"] == "status":
                yield Static(msg["content"], classes="status", markup=False)

    def add_message(self, sender_type: str, content: str):
        """Add a new message to the list.

        Args:
            sender_type: Type of sender ("user", "agent", "status", "separator")
            content: Message content
        """
        new_messages = self.messages.copy()
        new_messages.append({"type": sender_type, "content": content})
        self.messages = new_messages

    def add_separator(self):
        """Add a visual separator between exchanges."""
        self.add_message("separator", "")

    def toggle_markdown(self) -> bool:
        """Toggle markdown rendering mode.

        Returns:
            New markdown mode state
        """
        self.markdown_mode = not self.markdown_mode
        return self.markdown_mode

    def watch_messages(self):
        """Called when messages change - auto-scroll to bottom."""
        self.call_after_refresh(self.scroll_end)
