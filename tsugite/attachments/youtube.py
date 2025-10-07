"""YouTube transcript handler for attachments."""

import re
from typing import Optional

from tsugite.attachments.base import AttachmentHandler


class YouTubeHandler(AttachmentHandler):
    """Handler for YouTube video transcripts."""

    def can_handle(self, source: str) -> bool:
        """Check if source is a YouTube URL.

        Args:
            source: Source string

        Returns:
            True if source is a YouTube URL or youtube: prefix
        """
        patterns = [
            r"youtube\.com/watch",
            r"youtu\.be/",
            r"^youtube:",
        ]
        return any(re.search(pattern, source) for pattern in patterns)

    def fetch(self, source: str) -> str:
        """Fetch YouTube transcript.

        Args:
            source: YouTube URL or youtube:VIDEO_ID

        Returns:
            Transcript as formatted text

        Raises:
            ValueError: If transcript cannot be fetched
        """
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise ValueError("youtube-transcript-api not installed. Install with: uv add youtube-transcript-api")

        # Extract video ID
        video_id = self._extract_video_id(source)
        if not video_id:
            raise ValueError(f"Could not extract video ID from: {source}")

        try:
            # Fetch transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id)

            # Format as text
            lines = []
            for entry in transcript:
                timestamp = self._format_timestamp(entry["start"])
                text = entry["text"]
                lines.append(f"[{timestamp}] {text}")

            return "\n".join(lines)
        except Exception as e:
            raise ValueError(f"Failed to fetch YouTube transcript for {video_id}: {e}")

    def _extract_video_id(self, source: str) -> Optional[str]:
        """Extract video ID from YouTube URL or youtube: prefix.

        Args:
            source: YouTube URL or youtube:VIDEO_ID

        Returns:
            Video ID or None if not found
        """
        # Handle youtube:VIDEO_ID format
        if source.startswith("youtube:"):
            return source[8:]  # Remove "youtube:" prefix

        # Handle youtu.be/VIDEO_ID format
        match = re.search(r"youtu\.be/([a-zA-Z0-9_-]+)", source)
        if match:
            return match.group(1)

        # Handle youtube.com/watch?v=VIDEO_ID format
        match = re.search(r"[?&]v=([a-zA-Z0-9_-]+)", source)
        if match:
            return match.group(1)

        return None

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as MM:SS timestamp.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted timestamp string
        """
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
