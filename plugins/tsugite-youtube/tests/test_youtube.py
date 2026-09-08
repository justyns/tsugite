"""Tests for the tsugite-youtube attachment handler."""

from unittest.mock import patch

from tsugite_youtube import YouTubeHandler


def test_can_handle_youtube_sources():
    handler = YouTubeHandler()
    assert handler.can_handle("https://youtube.com/watch?v=abc123")
    assert handler.can_handle("https://youtu.be/abc123")
    assert handler.can_handle("youtube:abc123")
    assert not handler.can_handle("https://example.com/page")


def test_resolves_via_attachment_registry():
    """The handler is discoverable through the tsugite.attachments registry."""
    from tsugite.attachments import get_handler

    handler = get_handler("youtube:abc123")
    assert isinstance(handler, YouTubeHandler)


class _StubFetchedTranscript:
    def __init__(self, raw):
        self._raw = raw

    def to_raw_data(self):
        return self._raw


class _StubTranscriptApi:
    """The youtube-transcript-api 1.x surface: an instance with `fetch` and nothing else."""

    def fetch(self, video_id, languages=("en",), preserve_formatting=False):
        return _StubFetchedTranscript(
            [
                {"start": 0.0, "duration": 5.0, "text": "Hello world"},
                {"start": 5.0, "duration": 5.0, "text": "This is a test"},
            ]
        )


def test_fetch_formats_transcript():
    handler = YouTubeHandler()
    with patch("youtube_transcript_api.YouTubeTranscriptApi", _StubTranscriptApi):
        result = handler.fetch("https://youtube.com/watch?v=test123")

    assert result.name == "youtube:test123"
    assert "[00:00] Hello world" in result.content
    assert "[00:05] This is a test" in result.content
