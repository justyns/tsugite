"""Base class for scrollable reactive log widgets."""

from textual.containers import VerticalScroll
from textual.reactive import reactive


class BaseScrollLog(VerticalScroll):
    """Base class for scrollable log widgets with reactive entry lists."""

    entries = reactive([], recompose=True)

    def __init__(self):
        """Initialize base scroll log."""
        super().__init__()
        self.can_focus = True

    def watch_entries(self):
        """Called when entries change - auto-scroll to bottom."""
        self.call_after_refresh(self.scroll_end)

    def on_mount(self):
        """Called when widget is mounted - initial scroll."""
        self.scroll_end(animate=False)

    def clear_log(self):
        """Clear all entries from the log."""
        self.entries = []
