"""Workspace context attachment builder."""

from typing import List

from tsugite.attachments.base import Attachment
from tsugite.attachments.file import FileHandler

from .models import DEFAULT_MEMORY_INJECT_DAYS, Workspace


def build_workspace_attachments(
    workspace: Workspace,
    memory_days: int = DEFAULT_MEMORY_INJECT_DAYS,
) -> List[Attachment]:
    """Build attachment list from workspace files (convention-based).

    Args:
        workspace: Workspace to build attachments for
        memory_days: Number of days of memory files to include

    Returns:
        List of attachments for workspace identity and memory files
    """
    handler = FileHandler()
    attachments = []

    paths = workspace.get_workspace_files() + workspace.get_memory_files(days=memory_days)
    for path in paths:
        try:
            att = handler.fetch(str(path))
            att.source = path
            attachments.append(att)
        except Exception:
            pass

    return attachments


__all__ = ["build_workspace_attachments"]
