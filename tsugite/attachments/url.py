"""Generic URL handler for HTTP/HTTPS attachments."""

import urllib.request

from tsugite.attachments.base import AttachmentHandler


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

    def fetch(self, source: str) -> str:
        """Fetch URL content and convert HTML to markdown if needed.

        Args:
            source: URL to fetch

        Returns:
            Content as plain text (HTML converted to markdown if applicable)

        Raises:
            ValueError: If fetch fails
        """
        try:
            # Fetch URL - get both content and headers
            with urllib.request.urlopen(source, timeout=30) as response:
                content_type = response.headers.get("Content-Type", "").lower()
                content = response.read().decode("utf-8")

            # If HTML, convert to markdown
            if "text/html" in content_type:
                try:
                    import html2text

                    h = html2text.HTML2Text()
                    h.ignore_links = False
                    h.ignore_images = False
                    h.body_width = 0  # Don't wrap lines
                    return h.handle(content)
                except ImportError:
                    # Fall back to raw HTML if html2text not available
                    # Note: html2text is optional for better readability
                    return content
            else:
                # Plain text, JSON, XML, etc
                return content

        except Exception as e:
            raise ValueError(f"Failed to fetch URL '{source}': {e}")
