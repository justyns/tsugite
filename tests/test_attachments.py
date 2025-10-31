"""Tests for attachment management system."""

import json

import pytest

from tsugite.attachments import (
    add_attachment,
    get_attachment,
    list_attachments,
    remove_attachment,
    search_attachments,
)


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

        # Add initial inline attachment
        add_attachment("test", source="inline", content="Original content")

        # Update it
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

        # Remove it
        result = remove_attachment("test")
        assert result is True

        # Verify it's gone
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

        # Add inline attachment
        add_attachment("inline_test", source="inline", content="Content")
        # Add file reference
        add_attachment("file_test", source="/path/to/file")

        attachments_path = get_attachments_path()
        assert attachments_path.exists()

        with open(attachments_path) as f:
            data = json.load(f)

        assert "attachments" in data

        # Inline attachment should have content
        assert "inline_test" in data["attachments"]
        assert "content" in data["attachments"]["inline_test"]
        assert data["attachments"]["inline_test"]["content"] == "Content"
        assert data["attachments"]["inline_test"]["source"] == "inline"

        # File reference should NOT have content
        assert "file_test" in data["attachments"]
        assert "content" not in data["attachments"]["file_test"]
        assert data["attachments"]["file_test"]["source"] == "/path/to/file"

        # Both should have timestamps
        assert "created" in data["attachments"]["inline_test"]
        assert "updated" in data["attachments"]["inline_test"]
        assert "created" in data["attachments"]["file_test"]
        assert "updated" in data["attachments"]["file_test"]
