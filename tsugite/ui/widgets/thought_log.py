"""Thought log widget showing execution summaries and LLM thinking."""

from rich.text import Text
from textual.app import ComposeResult
from textual.widgets import Static

from .base_scroll_log import BaseScrollLog


class ThoughtEntry(Static):
    """Individual thought/execution entry."""

    def __init__(self, icon: str, content: str, style: str = "dim"):
        """Initialize thought entry.

        Args:
            icon: Emoji or symbol for entry type
            content: Brief description of what's happening
            style: Rich style for content
        """
        # Store values before calling super().__init__
        self.icon = icon
        self.entry_content = content  # Use different name to avoid conflict with Static.content
        self.entry_style = style
        super().__init__()

    def render(self) -> Text:
        """Render the thought entry."""
        text = Text()
        text.append(f"{self.icon} ", style=self.entry_style)
        text.append(str(self.entry_content), style=self.entry_style)
        return text


class ThoughtLog(BaseScrollLog):
    """Scrollable log of agent thoughts and execution summaries."""

    def __init__(self):
        """Initialize thought log."""
        super().__init__()

    def compose(self) -> ComposeResult:
        """Compose the thought log from reactive entries."""
        for entry in self.entries:
            entry_type = entry["type"]
            content = entry["content"]

            if entry_type == "step":
                yield ThoughtEntry("🧠", content, style="bold #83a598")  # gruvbox blue
            elif entry_type == "tool_call":
                yield ThoughtEntry("🔧", f"Using tool: {content}", style="#fabd2f")  # gruvbox yellow
            elif entry_type == "code_execution":
                # Summarize code execution
                code = content
                lines = code.count("\n") + 1
                preview = code[:40].replace("\n", " ") if code else "code"
                if len(code) > 40:
                    preview += "..."
                yield ThoughtEntry(
                    "⚡", f"Executing {lines} line(s) of code: {preview}", style="#d3869b"
                )  # gruvbox purple
            elif entry_type == "observation":
                yield ThoughtEntry("💡", f"Result: {content}", style="#8ec07c")  # gruvbox aqua
            elif entry_type == "execution_result":
                yield ThoughtEntry("📤", f"Output: {content}", style="#8ec07c")  # gruvbox aqua
            elif entry_type == "status":
                yield ThoughtEntry("ℹ️", content, style="dim #a89984")  # gruvbox gray
            elif entry_type == "error":
                yield ThoughtEntry("❌", content, style="bold #fb4934")  # gruvbox red

    def add_entry(self, entry_type: str, content: str):
        """Add a new entry to the thought log.

        Args:
            entry_type: Type of entry (step, tool_call, code_execution, etc.)
            content: Entry content/description
        """
        new_entries = self.entries.copy()
        new_entries.append({"type": entry_type, "content": content})
        self.entries = new_entries
