"""Workspace system for tsugite.

Workspaces represent persistent conversation contexts with:
- Personality/identity (PERSONA.md, SOUL.md)
- User context (USER.md)
- Memory (MEMORY.md, memory/*.md)
- Session persistence (auto-continue conversations)

Convention over configuration: No config.yaml required.
"""

from .context import build_workspace_attachments
from .models import (
    DEFAULT_COMPACTION_THRESHOLD,
    DEFAULT_MEMORY_INJECT_DAYS,
    WORKSPACE_FILES,
    Workspace,
    WorkspaceNotFoundError,
)
from .session import WorkspaceSession

__all__ = [
    "Workspace",
    "WorkspaceNotFoundError",
    "WorkspaceSession",
    "build_workspace_attachments",
    "WORKSPACE_FILES",
    "DEFAULT_MEMORY_INJECT_DAYS",
    "DEFAULT_COMPACTION_THRESHOLD",
]
