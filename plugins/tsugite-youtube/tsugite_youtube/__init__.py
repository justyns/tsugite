"""Tsugite plugin: YouTube transcript attachment handler."""

import re
from typing import Optional

from tsugite.attachments.base import Attachment, AttachmentContentType, AttachmentHandler


class YouTubeHandler(AttachmentHandler):
    """Handler for YouTube video transcripts."""

    def can_handle(self, source: str) -> bool:
        patterns = [
            r"youtube\.com/watch",
            r"youtu\.be/",
            r"^youtube:",
        ]
        return any(re.search(pattern, source) for pattern in patterns)

    def fetch(self, source: str) -> Attachment:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise ValueError("youtube-transcript-api not installed.")

        video_id = self._extract_video_id(source)
        if not video_id:
            raise ValueError(f"Could not extract video ID from: {source}")

        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)  # pylint: disable=no-member

            lines = []
            for entry in transcript:
                timestamp = self._format_timestamp(entry["start"])
                lines.append(f"[{timestamp}] {entry['text']}")

            return Attachment(
                name=f"youtube:{video_id}",
                content="\n".join(lines),
                content_type=AttachmentContentType.TEXT,
                mime_type="text/plain",
                source_url=source,
            )
        except Exception as e:
            raise ValueError(f"Failed to fetch YouTube transcript for {video_id}: {e}")

    def _extract_video_id(self, source: str) -> Optional[str]:
        if source.startswith("youtube:"):
            return source[8:]

        match = re.search(r"youtu\.be/([a-zA-Z0-9_-]+)", source)
        if match:
            return match.group(1)

        match = re.search(r"[?&]v=([a-zA-Z0-9_-]+)", source)
        if match:
            return match.group(1)

        return None

    def _format_timestamp(self, seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


def create_handler(config):
    """Entry point factory for the tsugite.attachments group."""
    return YouTubeHandler()
