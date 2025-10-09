"""UI system for controlling agent execution display.

This module provides a flexible UI system with multiple handlers:
- CustomUIHandler: Rich UI with panels, colors, and emojis (default)
- PlainUIHandler: Plain text output for copy-paste workflows
- ChatUIHandler: Interactive chat UI with live updates

Use the helper functions to create loggers:
- custom_agent_ui(): Context manager for rich UI
- create_plain_logger(): Plain text logger
- create_silent_logger(): Silent logger (no output)
"""

from tsugite.ui.base import CustomUIHandler, CustomUILogger, UIEvent, UIState
from tsugite.ui.chat import ChatUIHandler
from tsugite.ui.helpers import create_plain_logger, create_silent_logger, custom_agent_ui
from tsugite.ui.plain import PlainUIHandler

__all__ = [
    # Core classes
    "UIEvent",
    "UIState",
    "CustomUILogger",
    # UI Handlers
    "CustomUIHandler",
    "PlainUIHandler",
    "ChatUIHandler",
    # Helper functions
    "custom_agent_ui",
    "create_silent_logger",
    "create_plain_logger",
]
