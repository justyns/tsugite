"""UI system for controlling agent execution display.

This module provides a flexible UI system with multiple handlers:
- CustomUIHandler: Rich UI with panels, colors, and emojis (default)
- PlainUIHandler: Plain text output for copy-paste workflows
- TextualUIHandler: Textual TUI chat interface handler

Use the helper functions to create loggers:
- custom_agent_ui(): Context manager for rich UI
- create_plain_logger(): Plain text logger
"""

from tsugite.ui.base import CustomUIHandler, CustomUILogger, UIState
from tsugite.ui.chat import ChatManager, ChatTurn
from tsugite.ui.helpers import (
    create_plain_logger,
    custom_agent_ui,
)
from tsugite.ui.plain import PlainUIHandler
from tsugite.ui.textual_handler import TextualUIHandler

# Import textual_chat to make it accessible as attribute for tests
# Must be after other imports since textual_chat imports from tsugite.ui
from tsugite.ui import textual_chat  # noqa: F401  # isort: skip

__all__ = [
    # Core classes
    "UIState",
    "CustomUILogger",
    # UI Handlers
    "CustomUIHandler",
    "PlainUIHandler",
    "TextualUIHandler",
    # Chat functionality
    "ChatManager",
    "ChatTurn",
    # Helper functions
    "custom_agent_ui",
    "create_plain_logger",
]
