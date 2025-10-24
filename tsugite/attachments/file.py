"""File handler for local file attachments."""

from pathlib import Path

from tsugite.attachments.base import AttachmentHandler


class FileHandler(AttachmentHandler):
    """Handler for local file references."""

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

    def fetch(self, source: str) -> str:
        """Read file content.

        Args:
            source: File path

        Returns:
            File content as string

        Raises:
            ValueError: If file cannot be read
        """
        try:
            path = Path(source).expanduser()
            return path.read_text(encoding="utf-8")
        except Exception as e:
            raise ValueError(f"Failed to read file '{source}': {e}")
