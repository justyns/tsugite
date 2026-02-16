"""UI system for controlling agent execution display.

This module provides a flexible UI system with multiple handlers:
- CustomUIHandler: Rich UI with panels, colors, and emojis (default)
- PlainUIHandler: Plain text output for copy-paste workflows

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

__all__ = [
    # Core classes
    "UIState",
    "CustomUILogger",
    # UI Handlers
    "CustomUIHandler",
    "PlainUIHandler",
    # Chat functionality
    "ChatManager",
    "ChatTurn",
    # Helper functions
    "custom_agent_ui",
    "create_plain_logger",
]
