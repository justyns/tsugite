"""Tests for attachment management, resolution, and agent integration."""

import json
from unittest.mock import patch

import pytest

from tsugite.attachments import (
    add_attachment,
    get_attachment,
    list_attachments,
    remove_attachment,
    search_attachments,
)
from tsugite.md_agents import parse_agent
from tsugite.utils import resolve_attachments


class TestAttachmentStorage:
    """Test attachment storage and retrieval."""

    def test_add_and_get_inline_attachment(self, tmp_path, monkeypatch):
        """Test adding and retrieving an inline attachment."""
        monkeypatch.setenv("HOME", str(tmp_path))

        content = "Test attachment content"
        add_attachment("test", source="inline", content=content)

        result = get_attachment("test")
        assert result is not None
        source, retrieved_content = result
        assert source == "inline"
        assert retrieved_content == content

    def test_add_and_get_file_reference(self, tmp_path, monkeypatch):
        """Test adding and retrieving a file reference."""
        monkeypatch.setenv("HOME", str(tmp_path))

        file_path = tmp_path / "test.txt"
        file_path.write_text("File content")

        add_attachment("test", source=str(file_path))

        result = get_attachment("test")
        assert result is not None
        source, content = result
        assert source == str(file_path)
        assert content is None  # Not stored inline

    def test_get_nonexistent_attachment(self, tmp_path, monkeypatch):
        """Test getting an attachment that doesn't exist."""
        monkeypatch.setenv("HOME", str(tmp_path))

        result = get_attachment("nonexistent")
        assert result is None

    def test_update_existing_attachment(self, tmp_path, monkeypatch):
        """Test updating an existing attachment."""
        monkeypatch.setenv("HOME", str(tmp_path))

        add_attachment("test", source="inline", content="Original content")
        add_attachment("test", source="inline", content="Updated content")

        result = get_attachment("test")
        assert result is not None
        source, content = result
        assert content == "Updated content"
        assert source == "inline"

    def test_list_attachments(self, tmp_path, monkeypatch):
        """Test listing all attachments."""
        monkeypatch.setenv("HOME", str(tmp_path))

        add_attachment("att1", source="inline", content="Content 1")
        add_attachment("att2", source="inline", content="Content 2")
        add_attachment("att3", source="/path/to/file")

        attachments = list_attachments()
        assert len(attachments) == 3
        assert "att1" in attachments
        assert "att2" in attachments
        assert "att3" in attachments

    def test_remove_attachment(self, tmp_path, monkeypatch):
        """Test removing an attachment."""
        monkeypatch.setenv("HOME", str(tmp_path))

        add_attachment("test", source="inline", content="Content")

        result = remove_attachment("test")
        assert result is True
        assert get_attachment("test") is None

    def test_remove_nonexistent_attachment(self, tmp_path, monkeypatch):
        """Test removing an attachment that doesn't exist."""
        monkeypatch.setenv("HOME", str(tmp_path))

        result = remove_attachment("nonexistent")
        assert result is False

    def test_search_attachments_by_alias(self, tmp_path, monkeypatch):
        """Test searching attachments by alias."""
        monkeypatch.setenv("HOME", str(tmp_path))

        add_attachment("python_code", source="inline", content="def foo(): pass")
        add_attachment("python_docs", source="inline", content="Documentation")
        add_attachment("java_code", source="inline", content="class Foo {}")

        results = search_attachments("python")
        assert len(results) == 2
        assert "python_code" in results
        assert "python_docs" in results
        assert "java_code" not in results

    def test_search_attachments_by_source(self, tmp_path, monkeypatch):
        """Test searching attachments by source."""
        monkeypatch.setenv("HOME", str(tmp_path))

        add_attachment("code1", source="https://example.com/api.md")
        add_attachment("code2", source="docs/readme.md")
        add_attachment("code3", source="https://example.com/guide.md")

        results = search_attachments("example.com")
        assert len(results) == 2
        assert "code1" in results
        assert "code3" in results
        assert "code2" not in results

    def test_search_case_insensitive(self, tmp_path, monkeypatch):
        """Test that search is case-insensitive."""
        monkeypatch.setenv("HOME", str(tmp_path))

        add_attachment("MyCode", source="MyFile.py")

        results = search_attachments("mycode")
        assert len(results) == 1
        assert "MyCode" in results

        results = search_attachments("myfile")
        assert len(results) == 1

    def test_empty_alias_error(self, tmp_path, monkeypatch):
        """Test that empty alias raises error."""
        monkeypatch.setenv("HOME", str(tmp_path))

        with pytest.raises(ValueError, match="alias cannot be empty"):
            add_attachment("", source="inline", content="Content")

        with pytest.raises(ValueError, match="alias cannot be empty"):
            add_attachment("   ", source="inline", content="Content")

    def test_inline_without_content_error(self, tmp_path, monkeypatch):
        """Test that inline attachment without content raises error."""
        monkeypatch.setenv("HOME", str(tmp_path))

        with pytest.raises(ValueError, match="Inline attachments require content"):
            add_attachment("test", source="inline")

    def test_attachments_json_format(self, tmp_path, monkeypatch):
        """Test that attachments.json has correct structure."""
        from tsugite.attachments import get_attachments_path

        monkeypatch.setenv("HOME", str(tmp_path))

        add_attachment("inline_test", source="inline", content="Content")
        add_attachment("file_test", source="/path/to/file")

        attachments_path = get_attachments_path()
        assert attachments_path.exists()

        with open(attachments_path) as f:
            data = json.load(f)

        assert "attachments" in data

        assert data["attachments"]["inline_test"]["content"] == "Content"
        assert data["attachments"]["inline_test"]["source"] == "inline"

        assert "content" not in data["attachments"]["file_test"]
        assert data["attachments"]["file_test"]["source"] == "/path/to/file"

        for key in ("inline_test", "file_test"):
            assert "created" in data["attachments"][key]
            assert "updated" in data["attachments"][key]


class TestAttachmentResolution:
    """Test resolving attachment references to content."""

    def test_resolve_inline_attachment(self, tmp_path, monkeypatch):
        """Test resolving an inline attachment."""
        monkeypatch.setenv("HOME", str(tmp_path))

        add_attachment("mycode", source="inline", content="def hello(): pass")

        results = resolve_attachments(["mycode"])
        assert len(results) == 1
        assert results[0].name == "mycode"
        assert results[0].content == "def hello(): pass"

    def test_resolve_file_attachment(self, tmp_path, monkeypatch):
        """Test resolving a file attachment."""
        monkeypatch.setenv("HOME", str(tmp_path))

        test_file = tmp_path / "test.txt"
        test_file.write_text("File content")

        add_attachment("myfile", source=str(test_file))

        results = resolve_attachments(["myfile"])
        assert len(results) == 1
        assert results[0].name == "test.txt"
        assert results[0].content == "File content"

    def test_resolve_url_attachment(self, tmp_path, monkeypatch):
        """Test resolving a URL attachment."""
        monkeypatch.setenv("HOME", str(tmp_path))

        add_attachment("myurl", source="https://example.com/doc.md")

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = mock_urlopen.return_value.__enter__.return_value
            mock_response.headers.get.return_value = "text/plain"
            mock_response.read.return_value = b"URL content here"

            results = resolve_attachments(["myurl"])
            assert len(results) == 1
            assert results[0].name == "doc.md"
            assert results[0].content == "URL content here"

    def test_resolve_multiple_attachments(self, tmp_path, monkeypatch):
        """Test resolving multiple attachments."""
        monkeypatch.setenv("HOME", str(tmp_path))

        add_attachment("alias1", source="inline", content="Alias content")

        test_file = tmp_path / "file.txt"
        test_file.write_text("File content")
        add_attachment("file1", source=str(test_file))

        add_attachment("url1", source="https://example.com/doc.md")

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = mock_urlopen.return_value.__enter__.return_value
            mock_response.headers.get.return_value = "text/plain"
            mock_response.read.return_value = b"URL content"

            results = resolve_attachments(["alias1", "file1", "url1"])

            assert len(results) == 3
            assert results[0].name == "alias1"
            assert results[0].content == "Alias content"
            assert results[1].name == "file.txt"
            assert results[1].content == "File content"
            assert results[2].name == "doc.md"
            assert results[2].content == "URL content"

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

        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content")

        add_attachment("myfile", source=str(test_file))
        results1 = resolve_attachments(["myfile"])
        assert results1[0].content == "Original content"

        test_file.write_text("Modified content")

        results2 = resolve_attachments(["myfile"])
        assert results2[0].content == "Original content"  # Still cached

        results3 = resolve_attachments(["myfile"], refresh_cache=True)
        assert results3[0].content == "Modified content"

    def test_youtube_handler_integration(self, tmp_path, monkeypatch):
        """Test YouTube handler integration."""
        monkeypatch.setenv("HOME", str(tmp_path))

        add_attachment("tutorial", source="https://youtube.com/watch?v=test123")

        mock_transcript = [
            {"start": 0.0, "text": "Hello world"},
            {"start": 5.0, "text": "This is a test"},
        ]

        with patch("youtube_transcript_api.YouTubeTranscriptApi") as mock_api:
            mock_api.get_transcript.return_value = mock_transcript

            results = resolve_attachments(["tutorial"])
            assert len(results) == 1
            assert results[0].name == "youtube:test123"
            assert "[00:00] Hello world" in results[0].content
            assert "[00:05] This is a test" in results[0].content

    def test_cache_metadata_structure(self, tmp_path, monkeypatch):
        """Test that cache metadata has expected structure."""
        from tsugite.cache import get_cache_key, list_cache

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        add_attachment("myfile", source=str(test_file))

        resolve_attachments(["myfile"])

        cache_entries = list_cache()
        cache_key = get_cache_key(str(test_file))

        assert cache_key in cache_entries
        cache_info = cache_entries[cache_key]
        assert "source" in cache_info
        assert "cached_at" in cache_info
        assert "size" in cache_info
        assert cache_info["source"] == str(test_file)
        assert cache_info["size"] > 0


class TestAgentAttachments:
    """Test attachment support in agent definitions."""

    def test_agent_with_attachments_field(self):
        """Test parsing agent with attachments field."""
        agent_text = """---
name: test_agent
model: openai:gpt-4o-mini
tools: []
attachments:
  - coding-standards
  - api-docs
---

Task: {{ user_prompt }}
"""
        agent = parse_agent(agent_text)
        assert agent.config.attachments == ["coding-standards", "api-docs"]

    def test_agent_without_attachments_field(self):
        """Test parsing agent without attachments field."""
        agent_text = """---
name: test_agent
model: openai:gpt-4o-mini
tools: []
---

Task: {{ user_prompt }}
"""
        agent = parse_agent(agent_text)
        assert agent.config.attachments == []

    def test_agent_with_empty_attachments(self):
        """Test parsing agent with empty attachments list."""
        agent_text = """---
name: test_agent
model: openai:gpt-4o-mini
tools: []
attachments: []
---

Task: {{ user_prompt }}
"""
        agent = parse_agent(agent_text)
        assert agent.config.attachments == []

    def test_agent_with_single_attachment(self):
        """Test parsing agent with single attachment."""
        agent_text = """---
name: test_agent
model: openai:gpt-4o-mini
tools: []
attachments:
  - style-guide
---

Task: {{ user_prompt }}
"""
        agent = parse_agent(agent_text)
        assert agent.config.attachments == ["style-guide"]

    def test_agent_attachments_in_agent_info(self, tmp_path, monkeypatch):
        """Test that attachments appear in get_agent_info."""
        from tsugite.agent_runner import get_agent_info

        monkeypatch.setenv("HOME", str(tmp_path))

        agent_text = """---
name: test_agent
model: openai:gpt-4o-mini
tools: []
attachments:
  - coding-standards
  - security-guide
---

Task: {{ user_prompt }}
"""
        agent_file = tmp_path / "test_agent.md"
        agent_file.write_text(agent_text)

        agent_info = get_agent_info(agent_file)

        assert "attachments" in agent_info
        assert agent_info["attachments"] == ["coding-standards", "security-guide"]

    def test_agent_attachments_resolve(self, tmp_path, monkeypatch):
        """Test full integration: agent attachments resolve correctly."""
        monkeypatch.setenv("HOME", str(tmp_path))

        add_attachment("style-guide", source="inline", content="Use tabs for indentation")

        agent_text = """---
name: code_reviewer
model: openai:gpt-4o-mini
tools: []
attachments:
  - style-guide
---

You are a code reviewer.

Task: {{ user_prompt }}
"""
        agent = parse_agent(agent_text)

        resolved = resolve_attachments(agent.config.attachments)

        assert len(resolved) == 1
        assert resolved[0].name == "style-guide"
        assert resolved[0].content == "Use tabs for indentation"
