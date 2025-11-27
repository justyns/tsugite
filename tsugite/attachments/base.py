"""Base class for attachment handlers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union


class AttachmentContentType(Enum):
    """Type of attachment content."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"


@dataclass
class Attachment:
    """Represents an attachment with its content and metadata.

    Attributes:
        name: Name/identifier for the attachment
        content: Content as string (text) or bytes (binary), or None for URL-only attachments
        content_type: Type of content (TEXT, IMAGE, AUDIO, DOCUMENT)
        mime_type: MIME type of the content (e.g., 'image/jpeg', 'application/pdf')
        source_url: Optional URL source (for remote attachments that don't need downloading)
    """

    name: str
    content: Optional[Union[str, bytes]]
    content_type: AttachmentContentType
    mime_type: str
    source_url: Optional[str] = None


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
    def fetch(self, source: str) -> Attachment:
        """Fetch and return content for this source.

        Args:
            source: Source string to fetch

        Returns:
            Attachment object with content and metadata

        Raises:
            ValueError: If fetch fails
        """
        pass
