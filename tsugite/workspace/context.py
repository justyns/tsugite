"""Workspace context attachment builder."""

from typing import List

from tsugite.attachments.base import Attachment
from tsugite.attachments.file import FileHandler

from .models import Workspace


def build_workspace_attachments(
    workspace: Workspace,
) -> List[Attachment]:
    """Build attachment list from workspace files (convention-based).

    Args:
        workspace: Workspace to build attachments for

    Returns:
        List of attachments for workspace identity files
    """
    handler = FileHandler()
    attachments = []

    paths = workspace.get_workspace_files()
    for path in paths:
        try:
            att = handler.fetch(str(path))
            att.source = path
            attachments.append(att)
        except Exception:
            pass

    return attachments


__all__ = ["build_workspace_attachments"]
