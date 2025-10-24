"""Base class for attachment handlers."""

from abc import ABC, abstractmethod


class AttachmentHandler(ABC):
    """Base class for attachment handlers."""

    @abstractmethod
    def can_handle(self, source: str) -> bool:
        """Check if this handler can process the source.

        Args:
            source: Source string (URL, file path, etc.)

        Returns:
            True if this handler can process the source
        """
        pass

    @abstractmethod
    def fetch(self, source: str) -> str:
        """Fetch and return content for this source.

        Args:
            source: Source string to fetch

        Returns:
            Content as string

        Raises:
            ValueError: If fetch fails
        """
        pass
