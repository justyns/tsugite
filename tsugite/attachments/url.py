"""Generic URL handler for HTTP/HTTPS attachments."""

import logging
import urllib.request
from urllib.parse import urlparse

from tsugite.attachments.base import Attachment, AttachmentContentType, AttachmentHandler

logger = logging.getLogger(__name__)


class GenericURLHandler(AttachmentHandler):
    """Handler for generic HTTP(S) URLs."""

    def can_handle(self, source: str) -> bool:
        """Check if source is an HTTP(S) URL.

        Args:
            source: Source string

        Returns:
            True if source is HTTP or HTTPS URL
        """
        return source.startswith("http://") or source.startswith("https://")

    def _get_content_type(self, source: str) -> str:
        """Get content-type from URL using HEAD request.

        Args:
            source: URL to check

        Returns:
            Content-Type header value (lowercase)
        """
        try:
            request = urllib.request.Request(source, method="HEAD")
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.headers.get("Content-Type", "").lower()
        except Exception as e:
            logger.debug("HEAD request failed for %s: %s, trying GET", source, e)
            # If HEAD fails, fall back to GET and check headers
            try:
                with urllib.request.urlopen(source, timeout=10) as response:
                    return response.headers.get("Content-Type", "").lower()
            except Exception as e2:
                logger.warning("Failed to detect content-type for %s: %s", source, e2)
                return ""

    def _get_name_from_url(self, url: str) -> str:
        """Extract filename from URL path."""
        parsed = urlparse(url)
        path = parsed.path
        if path and "/" in path:
            return path.split("/")[-1] or "attachment"
        return "attachment"

    def fetch(self, source: str) -> Attachment:
        """Fetch URL content or prepare URL reference for LiteLLM.

        For images and documents, returns URL reference without downloading
        (LiteLLM can fetch them directly).
        For text/HTML, downloads and converts to text.

        Args:
            source: URL to fetch

        Returns:
            Attachment object with appropriate content type

        Raises:
            ValueError: If fetch fails
        """
        try:
            # Get content type without downloading full content
            content_type = self._get_content_type(source)
            name = self._get_name_from_url(source)

            # Determine content type and handle accordingly
            if content_type.startswith("image/"):
                # Image - let LiteLLM fetch it
                mime_type = content_type.split(";")[0].strip()
                return Attachment(
                    name=name,
                    content=None,
                    content_type=AttachmentContentType.IMAGE,
                    mime_type=mime_type,
                    source_url=source,
                )

            elif content_type.startswith("audio/"):
                # Audio - let LiteLLM fetch it
                mime_type = content_type.split(";")[0].strip()
                return Attachment(
                    name=name,
                    content=None,
                    content_type=AttachmentContentType.AUDIO,
                    mime_type=mime_type,
                    source_url=source,
                )

            elif content_type in (
                "application/pdf",
                "application/vnd.ms-excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/msword",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ):
                # Document - let LiteLLM fetch it
                mime_type = content_type.split(";")[0].strip()
                return Attachment(
                    name=name,
                    content=None,
                    content_type=AttachmentContentType.DOCUMENT,
                    mime_type=mime_type,
                    source_url=source,
                )

            else:
                # Text content - download and decode
                with urllib.request.urlopen(source, timeout=30) as response:
                    content = response.read().decode("utf-8")

                # If HTML, convert to markdown
                if "text/html" in content_type:
                    try:
                        import html2text

                        h = html2text.HTML2Text()
                        h.ignore_links = False
                        h.ignore_images = False
                        h.body_width = 0  # Don't wrap lines
                        content = h.handle(content)
                    except ImportError:
                        # Fall back to raw HTML if html2text not available
                        pass

                mime_type = content_type.split(";")[0].strip() if content_type else "text/plain"
                return Attachment(
                    name=name,
                    content=content,
                    content_type=AttachmentContentType.TEXT,
                    mime_type=mime_type,
                    source_url=None,
                )

        except Exception as e:
            raise ValueError(f"Failed to fetch URL '{source}': {e}")
