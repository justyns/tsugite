"""Attachment handler system for different content sources."""

from typing import List

from tsugite.attachments.auto_context import AutoContextHandler
from tsugite.attachments.base import AttachmentHandler
from tsugite.attachments.file import FileHandler
from tsugite.attachments.inline import InlineHandler
from tsugite.attachments.storage import (
    add_attachment,
    get_attachment,
    get_attachments_path,
    list_attachments,
    remove_attachment,
    search_attachments,
)
from tsugite.attachments.url import GenericURLHandler
from tsugite.attachments.youtube import YouTubeHandler

__all__ = [
    # Handlers
    "AttachmentHandler",
    "InlineHandler",
    "FileHandler",
    "YouTubeHandler",
    "GenericURLHandler",
    "AutoContextHandler",
    "HANDLERS",
    "get_handler",
    # Storage functions
    "add_attachment",
    "get_attachment",
    "get_attachments_path",
    "list_attachments",
    "remove_attachment",
    "search_attachments",
]

# Handler registry - order matters! More specific handlers first
HANDLERS: List[AttachmentHandler] = [
    InlineHandler(),
    AutoContextHandler(),  # Before FileHandler (might match file paths)
    YouTubeHandler(),  # Before GenericURLHandler
    FileHandler(),
    GenericURLHandler(),  # Catch-all for URLs
]


def get_handler(source: str) -> AttachmentHandler:
    """Get appropriate handler for a source.

    Args:
        source: Source string

    Returns:
        Handler that can process this source

    Raises:
        ValueError: If no handler found
    """
    for handler in HANDLERS:
        if handler.can_handle(source):
            return handler

    raise ValueError(f"No handler found for source: {source}")
