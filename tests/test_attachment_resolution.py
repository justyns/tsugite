"""Tests for attachment resolution (aliases, URLs, files)."""

from unittest.mock import patch

import pytest

from tsugite.attachments import add_attachment
from tsugite.utils import resolve_attachments


class TestAttachmentResolution:
    """Test resolving attachment references to content."""

    def test_resolve_inline_attachment(self, tmp_path, monkeypatch):
        """Test resolving an inline attachment."""
        monkeypatch.setenv("HOME", str(tmp_path))

        # Add inline attachment
        add_attachment("mycode", source="inline", content="def hello(): pass")

        # Resolve it
        results = resolve_attachments(["mycode"])
        assert len(results) == 1
        name, content = results[0]
        assert name == "mycode"
        assert content == "def hello(): pass"

    def test_resolve_file_attachment(self, tmp_path, monkeypatch):
        """Test resolving a file attachment."""
        monkeypatch.setenv("HOME", str(tmp_path))

        # Create a file
        test_file = tmp_path / "test.txt"
        test_file.write_text("File content")

        # Register as attachment
        add_attachment("myfile", source=str(test_file))

        # Resolve it
        results = resolve_attachments(["myfile"])
        assert len(results) == 1
        name, content = results[0]
        assert name == "myfile"
        assert content == "File content"

    def test_resolve_url_attachment(self, tmp_path, monkeypatch):
        """Test resolving a URL attachment."""
        monkeypatch.setenv("HOME", str(tmp_path))

        # Register URL attachment
        add_attachment("myurl", source="https://example.com/doc.md")

        mock_content = "URL content here"

        # Mock URL fetching
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = mock_urlopen.return_value.__enter__.return_value
            mock_response.headers.get.return_value = "text/plain"
            mock_response.read.return_value = mock_content.encode()

            results = resolve_attachments(["myurl"])
            assert len(results) == 1
            name, content = results[0]
            assert name == "myurl"
            assert content == mock_content

    def test_resolve_multiple_attachments(self, tmp_path, monkeypatch):
        """Test resolving multiple attachments."""
        monkeypatch.setenv("HOME", str(tmp_path))

        # Add inline attachment
        add_attachment("alias1", source="inline", content="Alias content")

        # Create and register file
        test_file = tmp_path / "file.txt"
        test_file.write_text("File content")
        add_attachment("file1", source=str(test_file))

        # Register URL
        add_attachment("url1", source="https://example.com/doc.md")

        # Mock URL fetch
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = mock_urlopen.return_value.__enter__.return_value
            mock_response.headers.get.return_value = "text/plain"
            mock_response.read.return_value = b"URL content"

            results = resolve_attachments(["alias1", "file1", "url1"])

            assert len(results) == 3
            assert results[0] == ("alias1", "Alias content")
            assert results[1] == ("file1", "File content")
            assert results[2] == ("url1", "URL content")

    def test_resolve_nonexistent_attachment_error(self, tmp_path, monkeypatch):
        """Test error when attachment doesn't exist."""
        monkeypatch.setenv("HOME", str(tmp_path))

        with pytest.raises(ValueError, match="Attachment not found"):
            resolve_attachments(["nonexistent"])

    def test_resolve_empty_list(self, tmp_path, monkeypatch):
        """Test resolving empty list."""
        monkeypatch.setenv("HOME", str(tmp_path))

        results = resolve_attachments([])
        assert results == []

    def test_caching_works(self, tmp_path, monkeypatch):
        """Test that caching works for file attachments."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

        # Create file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content")

        # Register and resolve
        add_attachment("myfile", source=str(test_file))
        results1 = resolve_attachments(["myfile"])
        assert results1[0][1] == "Original content"

        # Modify file
        test_file.write_text("Modified content")

        # Resolve again - should get cached version
        results2 = resolve_attachments(["myfile"])
        assert results2[0][1] == "Original content"  # Still cached

        # Resolve with refresh_cache - should get new version
        results3 = resolve_attachments(["myfile"], refresh_cache=True)
        assert results3[0][1] == "Modified content"

    def test_youtube_handler_integration(self, tmp_path, monkeypatch):
        """Test YouTube handler integration."""
        monkeypatch.setenv("HOME", str(tmp_path))

        # Register YouTube URL
        add_attachment("tutorial", source="https://youtube.com/watch?v=test123")

        # Mock YouTube API
        mock_transcript = [
            {"start": 0.0, "text": "Hello world"},
            {"start": 5.0, "text": "This is a test"},
        ]

        with patch("youtube_transcript_api.YouTubeTranscriptApi") as mock_api:
            mock_api.get_transcript.return_value = mock_transcript

            results = resolve_attachments(["tutorial"])
            assert len(results) == 1
            name, content = results[0]
            assert name == "tutorial"
            assert "[00:00] Hello world" in content
            assert "[00:05] This is a test" in content

    def test_cache_metadata_structure(self, tmp_path, monkeypatch):
        """Test that cache metadata has expected structure for show command."""
        from tsugite.cache import get_cache_key, list_cache

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

        # Create and register file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        add_attachment("myfile", source=str(test_file))

        # Resolve to create cache
        resolve_attachments(["myfile"])

        # Get cache metadata
        cache_entries = list_cache()
        cache_key = get_cache_key(str(test_file))

        # Verify metadata structure matches what show command expects
        assert cache_key in cache_entries
        cache_info = cache_entries[cache_key]
        assert "source" in cache_info
        assert "cached_at" in cache_info  # This is what show command uses
        assert "size" in cache_info
        assert cache_info["source"] == str(test_file)
        assert isinstance(cache_info["size"], int)
        assert cache_info["size"] > 0
