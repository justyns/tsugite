"""File handler for local file attachments."""

import base64
import logging
import mimetypes
from pathlib import Path

from tsugite.attachments.base import Attachment, AttachmentContentType, AttachmentHandler

logger = logging.getLogger(__name__)


class FileHandler(AttachmentHandler):
    """Handler for local file references."""

    # Map of file extensions to content types
    BINARY_EXTENSIONS = {
        # Images
        ".jpg": ("image/jpeg", AttachmentContentType.IMAGE),
        ".jpeg": ("image/jpeg", AttachmentContentType.IMAGE),
        ".png": ("image/png", AttachmentContentType.IMAGE),
        ".gif": ("image/gif", AttachmentContentType.IMAGE),
        ".webp": ("image/webp", AttachmentContentType.IMAGE),
        ".svg": ("image/svg+xml", AttachmentContentType.IMAGE),
        ".bmp": ("image/bmp", AttachmentContentType.IMAGE),
        ".tiff": ("image/tiff", AttachmentContentType.IMAGE),
        ".tif": ("image/tiff", AttachmentContentType.IMAGE),
        # Audio
        ".mp3": ("audio/mpeg", AttachmentContentType.AUDIO),
        ".wav": ("audio/wav", AttachmentContentType.AUDIO),
        ".ogg": ("audio/ogg", AttachmentContentType.AUDIO),
        ".m4a": ("audio/mp4", AttachmentContentType.AUDIO),
        ".flac": ("audio/flac", AttachmentContentType.AUDIO),
        # Documents
        ".pdf": ("application/pdf", AttachmentContentType.DOCUMENT),
        ".doc": ("application/msword", AttachmentContentType.DOCUMENT),
        ".docx": (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            AttachmentContentType.DOCUMENT,
        ),
        ".xls": ("application/vnd.ms-excel", AttachmentContentType.DOCUMENT),
        ".xlsx": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", AttachmentContentType.DOCUMENT),
        ".ppt": ("application/vnd.ms-powerpoint", AttachmentContentType.DOCUMENT),
        ".pptx": (
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            AttachmentContentType.DOCUMENT,
        ),
    }

    def can_handle(self, source: str) -> bool:
        """Check if source is a file path.

        Args:
            source: Source string

        Returns:
            True if source looks like a file path and exists
        """
        # Don't handle URLs or inline markers
        if source.lower() in ("inline", "text"):
            return False
        if source.startswith("http://") or source.startswith("https://"):
            return False
        if source.startswith("youtube:"):
            return False

        # Check if it's a valid file path
        try:
            path = Path(source).expanduser()
            return path.exists() and path.is_file()
        except (OSError, RuntimeError):
            return False

    def _detect_content_type(self, path: Path) -> tuple[str, AttachmentContentType]:
        """Detect MIME type and content type from file.

        Args:
            path: Path to file

        Returns:
            Tuple of (mime_type, content_type)
        """
        # Check by extension first
        ext = path.suffix.lower()
        if ext in self.BINARY_EXTENSIONS:
            return self.BINARY_EXTENSIONS[ext]

        # Fall back to mimetypes module
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type:
            if mime_type.startswith("image/"):
                return (mime_type, AttachmentContentType.IMAGE)
            elif mime_type.startswith("audio/"):
                return (mime_type, AttachmentContentType.AUDIO)
            elif mime_type.startswith("application/"):
                return (mime_type, AttachmentContentType.DOCUMENT)

        # Default to text
        return ("text/plain", AttachmentContentType.TEXT)

    def fetch(self, source: str) -> Attachment:
        """Read file content.

        Args:
            source: File path

        Returns:
            Attachment object with content and metadata

        Raises:
            ValueError: If file cannot be read
        """
        try:
            path = Path(source).expanduser()
            mime_type, content_type = self._detect_content_type(path)

            if content_type == AttachmentContentType.TEXT:
                # Read as text
                content = path.read_text(encoding="utf-8")

                from tsugite.events.helpers import emit_file_read_event

                emit_file_read_event(str(path), content, "attachment")

                return Attachment(
                    name=path.name,
                    content=content,
                    content_type=AttachmentContentType.TEXT,
                    mime_type=mime_type,
                    source_url=None,
                )
            else:
                # Read as binary and base64 encode
                binary_content = path.read_bytes()
                encoded_content = base64.b64encode(binary_content).decode("utf-8")

                from tsugite.events.helpers import emit_file_read_event

                emit_file_read_event(str(path), f"[Binary file: {len(binary_content)} bytes]", "attachment")

                return Attachment(
                    name=path.name,
                    content=encoded_content,
                    content_type=content_type,
                    mime_type=mime_type,
                    source_url=None,
                )

        except Exception as e:
            logger.warning("Failed to read file '%s': %s", source, e)
            raise ValueError(f"Failed to read file '{source}': {e}") from e
