"""Tests for file reference expansion (@filename syntax)."""

from pathlib import Path

import pytest

from tsugite.attachments.base import AttachmentContentType
from tsugite.utils import expand_file_references


class TestExpandFileReferences:
    """Test cases for @filename expansion in prompts."""

    def test_single_file_reference(self, tmp_path):
        """Test basic @filename expansion."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!")

        prompt = "Analyze @test.txt"
        updated_prompt, attachments = expand_file_references(prompt, tmp_path)

        # @test.txt should be replaced with just test.txt in the prompt
        assert updated_prompt == "Analyze test.txt"
        assert "@test.txt" not in updated_prompt
        # File contents returned as Attachment objects
        assert len(attachments) == 1
        assert attachments[0].name == "test.txt"
        assert attachments[0].content == "Hello, world!"
        assert attachments[0].content_type == AttachmentContentType.TEXT

    def test_multiple_file_references(self, tmp_path):
        """Test multiple @filename expansions in one prompt."""
        # Create test files
        file1 = tmp_path / "file1.py"
        file1.write_text("def foo(): pass")

        file2 = tmp_path / "file2.md"
        file2.write_text("# Documentation")

        prompt = "Compare @file1.py and @file2.md"
        updated_prompt, attachments = expand_file_references(prompt, tmp_path)

        # @filenames should be replaced with just filenames
        assert updated_prompt == "Compare file1.py and file2.md"
        assert "@file1.py" not in updated_prompt
        assert "@file2.md" not in updated_prompt
        # Both file contents returned as Attachment objects
        assert len(attachments) == 2
        assert attachments[0].name == "file1.py"
        assert attachments[0].content == "def foo(): pass"
        assert attachments[1].name == "file2.md"
        assert attachments[1].content == "# Documentation"

    def test_quoted_path_with_spaces(self, tmp_path):
        """Test @"filename with spaces.txt" syntax."""
        # Create file with spaces in name
        test_file = tmp_path / "my file.txt"
        test_file.write_text("Content with spaces")

        prompt = 'Review @"my file.txt"'
        updated_prompt, attachments = expand_file_references(prompt, tmp_path)

        assert updated_prompt == "Review my file.txt"
        assert len(attachments) == 1
        assert attachments[0].name == "my file.txt"
        assert attachments[0].content == "Content with spaces"

    def test_relative_path(self, tmp_path):
        """Test relative path resolution."""
        # Create subdirectory and file
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        test_file = subdir / "test.py"
        test_file.write_text("print('hello')")

        prompt = "Analyze @subdir/test.py"
        updated_prompt, attachments = expand_file_references(prompt, tmp_path)

        assert updated_prompt == "Analyze subdir/test.py"
        assert len(attachments) == 1
        assert attachments[0].name == "subdir/test.py"
        assert attachments[0].content == "print('hello')"

    def test_absolute_path(self, tmp_path):
        """Test absolute path resolution."""
        test_file = tmp_path / "absolute.txt"
        test_file.write_text("Absolute content")

        abs_path = str(test_file)
        prompt = f"Check @{abs_path}"
        updated_prompt, attachments = expand_file_references(prompt, Path("/"))

        assert updated_prompt == f"Check {abs_path}"
        assert len(attachments) == 1
        assert attachments[0].name == abs_path
        assert attachments[0].content == "Absolute content"

    def test_file_not_found_error(self, tmp_path):
        """Test error when referenced file doesn't exist."""
        prompt = "Analyze @nonexistent.txt"

        with pytest.raises(ValueError, match="File not found: nonexistent.txt"):
            expand_file_references(prompt, tmp_path)

    def test_directory_not_file_error(self, tmp_path):
        """Test error when reference points to a directory."""
        subdir = tmp_path / "mydir"
        subdir.mkdir()

        prompt = "@mydir"

        with pytest.raises(ValueError, match="Path is not a file: mydir"):
            expand_file_references(prompt, tmp_path)

    def test_no_references_unchanged(self, tmp_path):
        """Test that prompts without @ are unchanged."""
        prompt = "This is a normal prompt without file references"
        updated_prompt, attachments = expand_file_references(prompt, tmp_path)

        assert updated_prompt == prompt
        assert attachments == []

    def test_unicode_content(self, tmp_path):
        """Test files with unicode content."""
        test_file = tmp_path / "unicode.txt"
        test_file.write_text("Hello 世界 🌍", encoding="utf-8")

        prompt = "Translate @unicode.txt"
        updated_prompt, attachments = expand_file_references(prompt, tmp_path)

        assert updated_prompt == "Translate unicode.txt"
        assert len(attachments) == 1
        assert attachments[0].name == "unicode.txt"
        assert attachments[0].content == "Hello 世界 🌍"

    def test_binary_file_error(self, tmp_path):
        """Test error on binary file (non-text)."""
        binary_file = tmp_path / "binary.bin"
        binary_file.write_bytes(b"\x00\x01\x02\xff\xfe")

        prompt = "@binary.bin"

        with pytest.raises(ValueError, match="not a text file or has encoding issues"):
            expand_file_references(prompt, tmp_path)

    def test_multiple_references_same_file(self, tmp_path):
        """Test referencing the same file multiple times."""
        test_file = tmp_path / "repeated.txt"
        test_file.write_text("Repeated content")

        prompt = "Compare @repeated.txt with @repeated.txt"
        updated_prompt, attachments = expand_file_references(prompt, tmp_path)

        # Should expand both occurrences
        assert updated_prompt == "Compare repeated.txt with repeated.txt"
        assert len(attachments) == 2
        assert attachments[0].name == "repeated.txt"
        assert attachments[0].content == "Repeated content"
        assert attachments[1].name == "repeated.txt"
        assert attachments[1].content == "Repeated content"

    def test_empty_file(self, tmp_path):
        """Test expansion of empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        prompt = "Check @empty.txt"
        updated_prompt, attachments = expand_file_references(prompt, tmp_path)

        assert updated_prompt == "Check empty.txt"
        assert len(attachments) == 1
        assert attachments[0].name == "empty.txt"
        assert attachments[0].content == ""

    @pytest.mark.parametrize(
        "filename,content,prompt,expected_prompt",
        [
            (
                "code.py",
                "def main(): pass",
                "Please review the code in @code.py and suggest improvements",
                "Please review the code in code.py and suggest improvements",
            ),
            ("start.txt", "Start content", "@start.txt is the file to analyze", "start.txt is the file to analyze"),
            ("end.txt", "End content", "Analyze this file: @end.txt", "Analyze this file: end.txt"),
            ("my.config.json", '{"key": "value"}', "Parse @my.config.json", "Parse my.config.json"),
            ("large.txt", "Line {}\n".format("x" * 100) * 1000, "@large.txt", "large.txt"),
        ],
        ids=["mixed_content", "at_start", "at_end", "dots_in_name", "large_file"],
    )
    def test_reference_position_and_content(self, tmp_path, filename, content, prompt, expected_prompt):
        (tmp_path / filename).write_text(content)

        updated_prompt, attachments = expand_file_references(prompt, tmp_path)

        assert updated_prompt == expected_prompt
        assert len(attachments) == 1
        assert attachments[0].name == filename
        assert attachments[0].content == content

    def test_at_symbol_with_special_chars_not_expanded(self, tmp_path):
        """Test that @ followed by special characters is not treated as file reference."""
        # Prompts with @ followed by special chars should pass through unchanged
        prompt = "Email me @#$%^&*() for details"
        updated_prompt, attachments = expand_file_references(prompt, tmp_path)

        # @ followed by special chars should not be expanded
        assert "@#$%^&*()" in updated_prompt
        assert attachments == []

    def test_at_mention_requires_file_to_exist(self, tmp_path):
        """Test that @word attempts file expansion (even if it looks like a mention).

        This is expected behavior - @word is treated as a file reference.
        Users should avoid patterns like @username in prompts if they don't want expansion,
        or the file must exist for the prompt to work.
        """
        # This will fail because @user123 looks like a file reference
        prompt = "Contact @user123"
        with pytest.raises(ValueError, match="File not found: user123"):
            expand_file_references(prompt, tmp_path)
