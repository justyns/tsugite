"""Scrollable message history widget for chat UI."""

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.widgets import Static


class Message(Static):
    """Individual message display widget."""

    def __init__(self, sender: str, content: str, sender_style: str = "bold", content_style: str = ""):
        """Initialize message widget.

        Args:
            sender: Sender label (e.g., "You", "Agent")
            content: Message content
            sender_style: Rich style for sender label
            content_style: Rich style for content
        """
        super().__init__()
        self.sender = sender
        self.content = content
        self.sender_style = sender_style
        self.content_style = content_style

    def render(self) -> Text:
        """Render the message with styling."""
        text = Text()
        text.append(f"{self.sender}: ", style=self.sender_style)
        text.append(self.content, style=self.content_style)
        return text


class MessageList(VerticalScroll):
    """Scrollable container for chat messages."""

    messages = reactive([], recompose=True)

    def __init__(self):
        """Initialize message list."""
        super().__init__()
        self.can_focus = False  # Don't steal focus from input

    def compose(self) -> ComposeResult:
        """Compose the message list from reactive messages."""
        for msg in self.messages:
            if msg["type"] == "user":
                yield Message("You", msg["content"], sender_style="bold green", content_style="green")
            elif msg["type"] == "agent":
                yield Message("Agent", msg["content"], sender_style="bold blue", content_style="blue")
            elif msg["type"] == "separator":
                yield Static("─" * 40, classes="separator")
            elif msg["type"] == "status":
                yield Static(msg["content"], classes="status")

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

    def watch_messages(self):
        """Called when messages change - auto-scroll to bottom."""
        # Schedule scroll to happen after recompose
        self.call_after_refresh(self.scroll_end)

    def on_mount(self):
        """Called when widget is mounted - initial scroll."""
        self.scroll_end(animate=False)
