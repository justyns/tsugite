"""Tests for attachment deduplication."""

from tsugite.cli.helpers import deduplicate_attachments


class TestDeduplicateAttachments:
    """Test cases for attachment deduplication."""

    def test_no_duplicates(self):
        """Test that unique attachments are preserved."""
        attachments = [
            ("file1.txt", "content1"),
            ("file2.txt", "content2"),
            ("file3.txt", "content3"),
        ]
        result = deduplicate_attachments(attachments)
        assert len(result) == 3
        assert result == attachments

    def test_exact_duplicate_names(self):
        """Test deduplication of attachments with exact same name."""
        attachments = [
            ("file.txt", "content"),
            ("file.txt", "content"),
        ]
        result = deduplicate_attachments(attachments)
        assert len(result) == 1
        assert result[0] == ("file.txt (also: file.txt)", "content")

    def test_content_based_deduplication(self):
        """Test deduplication by content hash (renamed files)."""
        attachments = [
            ("original.txt", "same content"),
            ("renamed.txt", "same content"),
            ("another.txt", "same content"),
        ]
        result = deduplicate_attachments(attachments)
        assert len(result) == 1
        assert result[0][0] == "original.txt (also: renamed.txt, another.txt)"
        assert result[0][1] == "same content"

    def test_symlink_deduplication(self, tmp_path):
        """Test that symlinks are deduplicated to their target."""
        # Create original file
        original = tmp_path / "original.txt"
        original.write_text("content")

        # Create symlink
        symlink = tmp_path / "symlink.txt"
        symlink.symlink_to(original)

        attachments = [
            (str(original), "content"),
            (str(symlink), "content"),
        ]
        result = deduplicate_attachments(attachments)
        assert len(result) == 1
        # Either name could be first depending on resolution, but should have alias
        assert "(also:" in result[0][0]
        assert result[0][1] == "content"

    def test_relative_vs_absolute_paths(self, tmp_path):
        """Test that relative and absolute paths to same file are deduplicated."""
        # Create file
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")

        # Get both relative and absolute paths
        abs_path = str(file_path.resolve())

        attachments = [
            (abs_path, "content"),
            (str(file_path), "content"),  # Could be relative depending on cwd
        ]
        result = deduplicate_attachments(attachments)
        # Should deduplicate to 1 if both resolve to same path
        assert len(result) == 1
        assert result[0][1] == "content"

    def test_mixed_duplicates(self):
        """Test deduplication with mix of path and content duplicates."""
        attachments = [
            ("file1.txt", "unique1"),
            ("file2.txt", "duplicate"),
            ("file3.txt", "duplicate"),  # Content duplicate
            ("file4.txt", "unique2"),
            ("file2.txt", "duplicate"),  # Exact name duplicate
        ]
        result = deduplicate_attachments(attachments)
        assert len(result) == 3  # unique1, duplicate (with aliases), unique2

        # Find the deduplicated entry
        duplicate_entry = next(r for r in result if "duplicate" in r[1])
        assert "(also:" in duplicate_entry[0]
        assert "file3.txt" in duplicate_entry[0]

    def test_preserves_order(self):
        """Test that order of first occurrence is preserved."""
        attachments = [
            ("third.txt", "content3"),
            ("first.txt", "content1"),
            ("second.txt", "content2"),
        ]
        result = deduplicate_attachments(attachments)
        assert len(result) == 3
        assert result[0][0] == "third.txt"
        assert result[1][0] == "first.txt"
        assert result[2][0] == "second.txt"

    def test_non_path_attachments(self):
        """Test that non-path attachments (URLs, inline) work correctly."""
        attachments = [
            ("https://example.com/doc", "web content"),
            ("inline://data", "inline content"),
            ("https://example.com/doc", "web content"),  # Duplicate URL
        ]
        result = deduplicate_attachments(attachments)
        # URL duplicates should be caught by content hash
        assert len(result) == 2

    def test_nonexistent_path(self):
        """Test handling of paths that don't exist."""
        attachments = [
            ("/nonexistent/file.txt", "content"),
            ("/another/nonexistent.txt", "content"),
        ]
        result = deduplicate_attachments(attachments)
        # Should deduplicate by content hash since paths don't resolve
        assert len(result) == 1
        assert "(also:" in result[0][0]

    def test_empty_list(self):
        """Test deduplication of empty list."""
        result = deduplicate_attachments([])
        assert result == []

    def test_single_attachment(self):
        """Test deduplication of single attachment."""
        attachments = [("file.txt", "content")]
        result = deduplicate_attachments(attachments)
        assert len(result) == 1
        assert result[0] == ("file.txt", "content")

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
            (str(original), "content"),
            (str(link1), "content"),
            (str(link2), "content"),
            (str(link3), "content"),
        ]
        result = deduplicate_attachments(attachments)
        assert len(result) == 1
        # Should have all three links as aliases
        assert "(also:" in result[0][0]
        assert "link1.txt" in result[0][0]
        assert "link2.txt" in result[0][0]
        assert "link3.txt" in result[0][0]

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
            (str(original), "content"),
            (str(link1), "content"),
            (str(link2), "content"),
        ]
        result = deduplicate_attachments(attachments)
        # All should resolve to the same canonical path
        assert len(result) == 1
        assert "(also:" in result[0][0]

    def test_unicode_content(self):
        """Test deduplication with unicode content."""
        attachments = [
            ("file1.txt", "Hello 世界 🌍"),
            ("file2.txt", "Hello 世界 🌍"),
        ]
        result = deduplicate_attachments(attachments)
        assert len(result) == 1
        assert result[0][1] == "Hello 世界 🌍"
        assert "(also:" in result[0][0]

    def test_large_content(self):
        """Test deduplication with large content."""
        large_content = "x" * 1000000  # 1MB
        attachments = [
            ("large1.txt", large_content),
            ("large2.txt", large_content),
        ]
        result = deduplicate_attachments(attachments)
        assert len(result) == 1
        assert len(result[0][1]) == 1000000

    def test_mixed_sources_integration(self):
        """Test deduplication across different sources (simulating real usage)."""
        # Simulates: agent attachment, CLI attachment, file reference
        attachments = [
            ("agent_config.json", '{"key": "value"}'),  # Agent attachment
            ("cli_config.json", '{"key": "value"}'),  # CLI attachment (same content)
            ("@inline_config.json", '{"key": "value"}'),  # File reference (same content)
            ("unique.txt", "unique content"),  # Unique attachment
        ]
        result = deduplicate_attachments(attachments)
        assert len(result) == 2  # config (with aliases) and unique
        config_entry = next(r for r in result if "config" in r[0])
        assert "(also:" in config_entry[0]
