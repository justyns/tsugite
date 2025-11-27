"""Tests for attachment deduplication."""

from tsugite.attachments.base import Attachment, AttachmentContentType
from tsugite.cli.helpers import deduplicate_attachments


def make_text_attachment(name: str, content: str) -> Attachment:
    """Helper to create text attachments for tests."""
    return Attachment(
        name=name,
        content=content,
        content_type=AttachmentContentType.TEXT,
        mime_type="text/plain",
    )


class TestDeduplicateAttachments:
    """Test cases for attachment deduplication."""

    def test_no_duplicates(self):
        """Test that unique attachments are preserved."""
        attachments = [
            make_text_attachment("file1.txt", "content1"),
            make_text_attachment("file2.txt", "content2"),
            make_text_attachment("file3.txt", "content3"),
        ]
        result = deduplicate_attachments(attachments)
        assert len(result) == 3
        assert result[0].name == "file1.txt"
        assert result[1].name == "file2.txt"
        assert result[2].name == "file3.txt"

    def test_exact_duplicate_names(self):
        """Test deduplication of attachments with exact same name."""
        attachments = [
            make_text_attachment("file.txt", "content"),
            make_text_attachment("file.txt", "content"),
        ]
        result = deduplicate_attachments(attachments)
        assert len(result) == 1
        assert result[0].name == "file.txt (also: file.txt)"
        assert result[0].content == "content"

    def test_content_based_deduplication(self):
        """Test deduplication by content hash (renamed files)."""
        attachments = [
            make_text_attachment("original.txt", "same content"),
            make_text_attachment("renamed.txt", "same content"),
            make_text_attachment("another.txt", "same content"),
        ]
        result = deduplicate_attachments(attachments)
        assert len(result) == 1
        assert result[0].name == "original.txt (also: renamed.txt, another.txt)"
        assert result[0].content == "same content"

    def test_symlink_deduplication(self, tmp_path):
        """Test that symlinks are deduplicated to their target."""
        # Create original file
        original = tmp_path / "original.txt"
        original.write_text("content")

        # Create symlink
        symlink = tmp_path / "symlink.txt"
        symlink.symlink_to(original)

        attachments = [
            make_text_attachment(str(original), "content"),
            make_text_attachment(str(symlink), "content"),
        ]
        result = deduplicate_attachments(attachments)
        assert len(result) == 1
        # Either name could be first depending on resolution, but should have alias
        assert "(also:" in result[0].name
        assert result[0].content == "content"

    def test_relative_vs_absolute_paths(self, tmp_path):
        """Test that relative and absolute paths to same file are deduplicated."""
        # Create file
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")

        # Get both relative and absolute paths
        abs_path = str(file_path.resolve())

        attachments = [
            make_text_attachment(abs_path, "content"),
            make_text_attachment(str(file_path), "content"),
        ]
        result = deduplicate_attachments(attachments)
        # Should deduplicate to 1 if both resolve to same path
        assert len(result) == 1
        assert result[0].content == "content"

    def test_mixed_duplicates(self):
        """Test deduplication with mix of path and content duplicates."""
        attachments = [
            make_text_attachment("file1.txt", "unique1"),
            make_text_attachment("file2.txt", "duplicate"),
            make_text_attachment("file3.txt", "duplicate"),  # Content duplicate
            make_text_attachment("file4.txt", "unique2"),
            make_text_attachment("file2.txt", "duplicate"),  # Exact name duplicate
        ]
        result = deduplicate_attachments(attachments)
        assert len(result) == 3  # unique1, duplicate (with aliases), unique2

        # Find the deduplicated entry
        duplicate_entry = next(r for r in result if r.content == "duplicate")
        assert "(also:" in duplicate_entry.name
        assert "file3.txt" in duplicate_entry.name

    def test_preserves_order(self):
        """Test that order of first occurrence is preserved."""
        attachments = [
            make_text_attachment("third.txt", "content3"),
            make_text_attachment("first.txt", "content1"),
            make_text_attachment("second.txt", "content2"),
        ]
        result = deduplicate_attachments(attachments)
        assert len(result) == 3
        assert result[0].name == "third.txt"
        assert result[1].name == "first.txt"
        assert result[2].name == "second.txt"

    def test_non_path_attachments(self):
        """Test that non-path attachments (URLs, inline) work correctly."""
        attachments = [
            make_text_attachment("https://example.com/doc", "web content"),
            make_text_attachment("inline://data", "inline content"),
            make_text_attachment("https://example.com/doc", "web content"),  # Duplicate URL
        ]
        result = deduplicate_attachments(attachments)
        # URL duplicates should be caught by content hash
        assert len(result) == 2

    def test_nonexistent_path(self):
        """Test handling of paths that don't exist."""
        attachments = [
            make_text_attachment("/nonexistent/file.txt", "content"),
            make_text_attachment("/another/nonexistent.txt", "content"),
        ]
        result = deduplicate_attachments(attachments)
        # Should deduplicate by content hash since paths don't resolve
        assert len(result) == 1
        assert "(also:" in result[0].name

    def test_empty_list(self):
        """Test deduplication of empty list."""
        result = deduplicate_attachments([])
        assert result == []

    def test_single_attachment(self):
        """Test deduplication of single attachment."""
        attachments = [make_text_attachment("file.txt", "content")]
        result = deduplicate_attachments(attachments)
        assert len(result) == 1
        assert result[0].name == "file.txt"
        assert result[0].content == "content"

    def test_multiple_symlinks_to_same_file(self, tmp_path):
        """Test multiple symlinks all pointing to same file."""
        # Create original
        original = tmp_path / "original.txt"
        original.write_text("content")

        # Create multiple symlinks
        link1 = tmp_path / "link1.txt"
        link2 = tmp_path / "link2.txt"
        link3 = tmp_path / "link3.txt"
        link1.symlink_to(original)
        link2.symlink_to(original)
        link3.symlink_to(original)

        attachments = [
            make_text_attachment(str(original), "content"),
            make_text_attachment(str(link1), "content"),
            make_text_attachment(str(link2), "content"),
            make_text_attachment(str(link3), "content"),
        ]
        result = deduplicate_attachments(attachments)
        assert len(result) == 1
        # Should have all three links as aliases
        assert "(also:" in result[0].name
        assert "link1.txt" in result[0].name
        assert "link2.txt" in result[0].name
        assert "link3.txt" in result[0].name

    def test_chain_of_symlinks(self, tmp_path):
        """Test chain of symlinks (symlink to symlink)."""
        # Create original
        original = tmp_path / "original.txt"
        original.write_text("content")

        # Create chain: link1 -> original, link2 -> link1
        link1 = tmp_path / "link1.txt"
        link2 = tmp_path / "link2.txt"
        link1.symlink_to(original)
        link2.symlink_to(link1)

        attachments = [
            make_text_attachment(str(original), "content"),
            make_text_attachment(str(link1), "content"),
            make_text_attachment(str(link2), "content"),
        ]
        result = deduplicate_attachments(attachments)
        # All should resolve to the same canonical path
        assert len(result) == 1
        assert "(also:" in result[0].name

    def test_unicode_content(self):
        """Test deduplication with unicode content."""
        attachments = [
            make_text_attachment("file1.txt", "Hello 世界 🌍"),
            make_text_attachment("file2.txt", "Hello 世界 🌍"),
        ]
        result = deduplicate_attachments(attachments)
        assert len(result) == 1
        assert result[0].content == "Hello 世界 🌍"
        assert "(also:" in result[0].name

    def test_large_content(self):
        """Test deduplication with large content."""
        large_content = "x" * 1000000  # 1MB
        attachments = [
            make_text_attachment("large1.txt", large_content),
            make_text_attachment("large2.txt", large_content),
        ]
        result = deduplicate_attachments(attachments)
        assert len(result) == 1
        assert len(result[0].content) == 1000000

    def test_mixed_sources_integration(self):
        """Test deduplication across different sources (simulating real usage)."""
        # Simulates: agent attachment, CLI attachment, file reference
        attachments = [
            make_text_attachment("agent_config.json", '{"key": "value"}'),
            make_text_attachment("cli_config.json", '{"key": "value"}'),
            make_text_attachment("@inline_config.json", '{"key": "value"}'),
            make_text_attachment("unique.txt", "unique content"),
        ]
        result = deduplicate_attachments(attachments)
        assert len(result) == 2  # config (with aliases) and unique
        config_entry = next(r for r in result if "config" in r.name)
        assert "(also:" in config_entry.name

    def test_url_only_attachments(self):
        """Test deduplication of URL-only attachments (no content downloaded)."""
        # URL-only attachments use source_url for deduplication
        attachments = [
            Attachment(
                name="image1.png",
                content=None,
                content_type=AttachmentContentType.IMAGE,
                mime_type="image/png",
                source_url="https://example.com/image.png",
            ),
            Attachment(
                name="image2.png",
                content=None,
                content_type=AttachmentContentType.IMAGE,
                mime_type="image/png",
                source_url="https://example.com/image.png",  # Same URL
            ),
        ]
        result = deduplicate_attachments(attachments)
        assert len(result) == 1
        assert "(also:" in result[0].name

    def test_mixed_text_and_binary_attachments(self):
        """Test deduplication with mixed content types."""
        attachments = [
            make_text_attachment("readme.txt", "text content"),
            Attachment(
                name="image.png",
                content="base64data",
                content_type=AttachmentContentType.IMAGE,
                mime_type="image/png",
            ),
            make_text_attachment("readme2.txt", "text content"),  # Duplicate text
        ]
        result = deduplicate_attachments(attachments)
        assert len(result) == 2  # text (with alias) and image
