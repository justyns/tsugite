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


def test_fetch_formats_transcript():
    handler = YouTubeHandler()
    mock_transcript = [
        {"start": 0.0, "text": "Hello world"},
        {"start": 5.0, "text": "This is a test"},
    ]
    with patch("youtube_transcript_api.YouTubeTranscriptApi") as mock_api:
        mock_api.get_transcript.return_value = mock_transcript
        result = handler.fetch("https://youtube.com/watch?v=test123")

    assert result.name == "youtube:test123"
    assert "[00:00] Hello world" in result.content
    assert "[00:05] This is a test" in result.content
