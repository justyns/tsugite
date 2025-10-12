"""UI system for controlling agent execution display.

This module provides a flexible UI system with multiple handlers:
- CustomUIHandler: Rich UI with panels, colors, and emojis (default)
- PlainUIHandler: Plain text output for copy-paste workflows
- LiveTemplateHandler: Live Display with Tree and interactive prompts
- TextualUIHandler: Textual TUI chat interface handler

Use the helper functions to create loggers:
- custom_agent_ui(): Context manager for rich UI
- create_plain_logger(): Plain text logger
- create_silent_logger(): Silent logger (no output)
- create_live_template_logger(): Live display with tree visualization
"""

from tsugite.ui.base import CustomUIHandler, CustomUILogger, UIEvent, UIState
from tsugite.ui.helpers import (
    create_live_template_logger,
    create_plain_logger,
    create_silent_logger,
    custom_agent_ui,
)
from tsugite.ui.live_template import LiveTemplateHandler
from tsugite.ui.plain import PlainUIHandler
from tsugite.ui.textual_handler import TextualUIHandler

__all__ = [
    # Core classes
    "UIEvent",
    "UIState",
    "CustomUILogger",
    # UI Handlers
    "CustomUIHandler",
    "PlainUIHandler",
    "LiveTemplateHandler",
    "TextualUIHandler",
    # Helper functions
    "custom_agent_ui",
    "create_silent_logger",
    "create_plain_logger",
    "create_live_template_logger",
]
