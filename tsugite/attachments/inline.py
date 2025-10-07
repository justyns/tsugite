"""Inline text handler for attachments."""

from tsugite.attachments.base import AttachmentHandler


class InlineHandler(AttachmentHandler):
    """Handler for inline text content."""

    def can_handle(self, source: str) -> bool:
        """Check if source is inline text.

        Args:
            source: Source string

        Returns:
            True if source is "inline" or "text"
        """
        return source.lower() in ("inline", "text")

    def fetch(self, source: str) -> str:
        """Inline content is stored in attachments JSON, not fetched.

        Args:
            source: Source string (ignored)

        Returns:
            Empty string (inline content comes from attachments.json)
        """
        # Inline content is handled specially in resolve_attachments
        # This method should not be called for inline content
        return ""
