"""Execution status bar widget for chat UI."""

from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static


class StatusBar(Static):
    """Widget showing current execution status and tools used."""

    status = reactive("Ready")
    tools_used = reactive([])
    is_streaming = reactive(False)
    code_executing = reactive(None)

    def __init__(self):
        """Initialize status bar."""
        super().__init__()
        self.can_focus = False

    def render(self) -> Text:
        """Render the status bar with current state."""
        text = Text()

        # Title with separator
        text.append("━━━ ", style="cyan")
        text.append("Execution Status", style="bold cyan")
        text.append(" ", style="cyan")
        text.append("━" * 40, style="cyan")
        text.append("\n\n")

        # Current activity
        if self.is_streaming:
            text.append("⚡ ", style="yellow")
            text.append("Streaming response...", style="yellow")
        elif self.code_executing:
            text.append("⚡ ", style="yellow")
            text.append("Executing code: ", style="yellow")
            code_preview = self.code_executing[:50]
            if len(self.code_executing) > 50:
                code_preview += "..."
            text.append(code_preview, style="dim")
        elif self.tools_used:
            # Show last tool being used
            last_tool = self.tools_used[-1]
            text.append("⚡ ", style="yellow")
            text.append("Using tool: ", style="yellow")
            text.append(last_tool, style="magenta")
        elif self.status != "Ready":
            text.append("⚡ ", style="yellow")
            text.append(self.status, style="yellow")
        else:
            text.append("✓ ", style="green")
            text.append("Ready for input", style="green")

        # Show list of tools used in this turn
        if self.tools_used:
            text.append("\n\n")
            text.append("Tools used: ", style="dim")
            # Remove duplicates while preserving order
            unique_tools = list(dict.fromkeys(self.tools_used))
            text.append(", ".join(unique_tools), style="magenta")

        # Bottom separator
        text.append("\n\n")
        text.append("━" * 60, style="dim")

        return text
