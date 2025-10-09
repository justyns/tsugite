"""SSE UI handler for web interface - compatibility re-export.

This module re-exports SSEUIHandler from tsugite.ui for backward compatibility.
The actual implementation is in tsugite/ui/sse.py.

Deprecated: Use 'from tsugite.ui import SSEUIHandler' instead.
"""

from tsugite.ui import SSEUIHandler  # noqa: F401

__all__ = ["SSEUIHandler"]
