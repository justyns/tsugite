"""Workspace system for tsugite.

Workspaces represent persistent conversation contexts with:
- Personality/identity (PERSONA.md, SOUL.md)
- User context (USER.md)
- Memory (MEMORY.md, memory/*.md)
- Session persistence (auto-continue conversations)

Convention over configuration: No config.yaml required.
"""

from .models import (
    DEFAULT_COMPACTION_THRESHOLD,
    Workspace,
    WorkspaceNotFoundError,
)
from .session import WorkspaceSession

__all__ = [
    "Workspace",
    "WorkspaceNotFoundError",
    "WorkspaceSession",
    "DEFAULT_COMPACTION_THRESHOLD",
]
