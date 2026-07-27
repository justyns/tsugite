"""Attachment handler system for different content sources."""

from typing import List, Optional

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

__all__ = [
    # Handlers
    "AttachmentHandler",
    "InlineHandler",
    "FileHandler",
    "GenericURLHandler",
    "AutoContextHandler",
    "get_handler",
    "get_handlers",
    "get_specific_handler",
    # Storage functions
    "add_attachment",
    "get_attachment",
    "get_attachments_path",
    "list_attachments",
    "remove_attachment",
    "search_attachments",
]

# Built-in handlers. Order matters: specific handlers first, generic fallbacks last.
# Plugin handlers (tsugite.attachments) are inserted between the two so they can
# claim sources before the file/URL fallbacks.
_SPECIFIC_HANDLERS: List[AttachmentHandler] = [
    InlineHandler(),
    AutoContextHandler(),
]
_FALLBACK_HANDLERS: List[AttachmentHandler] = [
    FileHandler(),
    GenericURLHandler(),
]


def get_handlers() -> List[AttachmentHandler]:
    """Return all handlers: built-in specific, then plugin, then built-in fallback."""
    from tsugite.plugins import get_attachment_handlers

    return _SPECIFIC_HANDLERS + list(get_attachment_handlers()) + _FALLBACK_HANDLERS


def get_handler(source: str) -> AttachmentHandler:
    """Get appropriate handler for a source.

    Args:
        source: Source string

    Returns:
        Handler that can process this source

    Raises:
        ValueError: If no handler found
    """
    for handler in get_handlers():
        if handler.can_handle(source):
            return handler

    raise ValueError(f"No handler found for source: {source}")


def get_specific_handler(source: str) -> Optional[AttachmentHandler]:
    """A non-fallback handler that claims ``source`` (a built-in specific handler
    or a plugin one, e.g. the YouTube transcript handler), or None when only the
    generic file/URL fallback would match.

    Lets a caller prefer a smart handler for known sources while keeping its own
    generic path for everything else: the web context detector runs a pasted URL
    through this so a YouTube link yields a transcript, but a plain page still
    falls back to its readable-article scrape instead of the fallback handler's
    raw markdown.
    """
    from tsugite.plugins import get_attachment_handlers

    for handler in _SPECIFIC_HANDLERS + list(get_attachment_handlers()):
        if handler.can_handle(source):
            return handler
    return None
