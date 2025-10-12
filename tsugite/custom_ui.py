"""Backward compatibility shim for tsugite.custom_ui.

This module re-exports all UI components from tsugite.ui for backward compatibility.
New code should import from tsugite.ui directly.

Deprecated: Use 'from tsugite.ui import ...' instead.
"""

# Re-export everything from tsugite.ui
from tsugite.ui import (  # noqa: F401
    CustomUIHandler,
    CustomUILogger,
    PlainUIHandler,
    UIEvent,
    UIState,
    create_plain_logger,
    create_silent_logger,
    custom_agent_ui,
)

__all__ = [
    "UIEvent",
    "UIState",
    "CustomUILogger",
    "CustomUIHandler",
    "PlainUIHandler",
    "custom_agent_ui",
    "create_silent_logger",
    "create_plain_logger",
]
