"""Tests for file reference expansion (@filename syntax)."""

from pathlib import Path

import pytest

from tsugite.utils import expand_file_references


class TestExpandFileReferences:
    """Test cases for @filename expansion in prompts."""

    def test_single_file_reference(self, tmp_path):
        """Test basic @filename expansion."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!")

        prompt = "Analyze @test.txt"
        expanded, files = expand_file_references(prompt, tmp_path)

        # File contents should be prepended
        assert expanded.startswith("<File: test.txt>")
        assert "Hello, world!" in expanded
        assert "</File: test.txt>" in expanded
        # @test.txt should be replaced with just test.txt in the prompt
        assert "Analyze test.txt" in expanded
        assert "@test.txt" not in expanded
        assert files == ["test.txt"]

    def test_multiple_file_references(self, tmp_path):
        """Test multiple @filename expansions in one prompt."""
        # Create test files
        file1 = tmp_path / "file1.py"
        file1.write_text("def foo(): pass")

        file2 = tmp_path / "file2.md"
        file2.write_text("# Documentation")

        prompt = "Compare @file1.py and @file2.md"
        expanded, files = expand_file_references(prompt, tmp_path)

        # Both file contents should be prepended
        assert "def foo(): pass" in expanded
        assert "# Documentation" in expanded
        assert "<File: file1.py>" in expanded
        assert "<File: file2.md>" in expanded
        # @filenames should be replaced with just filenames
        assert "Compare file1.py and file2.md" in expanded
        assert "@file1.py" not in expanded
        assert "@file2.md" not in expanded
        assert set(files) == {"file1.py", "file2.md"}

    def test_quoted_path_with_spaces(self, tmp_path):
        """Test @"filename with spaces.txt" syntax."""
        # Create file with spaces in name
        test_file = tmp_path / "my file.txt"
        test_file.write_text("Content with spaces")

        prompt = 'Review @"my file.txt"'
        expanded, files = expand_file_references(prompt, tmp_path)

        assert "Content with spaces" in expanded
        assert "<File: my file.txt>" in expanded
        assert files == ["my file.txt"]

    def test_relative_path(self, tmp_path):
        """Test relative path resolution."""
        # Create subdirectory and file
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        test_file = subdir / "test.py"
        test_file.write_text("print('hello')")

        prompt = "Analyze @subdir/test.py"
        expanded, files = expand_file_references(prompt, tmp_path)

        assert "print('hello')" in expanded
        assert "<File: subdir/test.py>" in expanded
        assert files == ["subdir/test.py"]

    def test_absolute_path(self, tmp_path):
        """Test absolute path resolution."""
        test_file = tmp_path / "absolute.txt"
        test_file.write_text("Absolute content")

        abs_path = str(test_file)
        prompt = f"Check @{abs_path}"
        expanded, files = expand_file_references(prompt, Path("/"))

        assert "Absolute content" in expanded
        assert abs_path in files

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
        expanded, files = expand_file_references(prompt, tmp_path)

        assert expanded == prompt
        assert files == []

    def test_mixed_content(self, tmp_path):
        """Test prompt with text before and after @filename."""
        test_file = tmp_path / "code.py"
        test_file.write_text("def main(): pass")

        prompt = "Please review the code in @code.py and suggest improvements"
        expanded, files = expand_file_references(prompt, tmp_path)

        # File contents prepended
        assert expanded.startswith("<File: code.py>")
        assert "def main(): pass" in expanded
        # Prompt with @code.py replaced by code.py
        assert "Please review the code in code.py and suggest improvements" in expanded
        assert "@code.py" not in expanded
        assert files == ["code.py"]

    def test_unicode_content(self, tmp_path):
        """Test files with unicode content."""
        test_file = tmp_path / "unicode.txt"
        test_file.write_text("Hello 世界 🌍", encoding="utf-8")

        prompt = "Translate @unicode.txt"
        expanded, files = expand_file_references(prompt, tmp_path)

        assert "Hello 世界 🌍" in expanded
        assert files == ["unicode.txt"]

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
        expanded, files = expand_file_references(prompt, tmp_path)

        # Should expand both occurrences
        assert expanded.count("Repeated content") == 2
        assert files == ["repeated.txt", "repeated.txt"]

    def test_file_at_start_of_prompt(self, tmp_path):
        """Test @filename at the beginning of prompt."""
        test_file = tmp_path / "start.txt"
        test_file.write_text("Start content")

        prompt = "@start.txt is the file to analyze"
        expanded, files = expand_file_references(prompt, tmp_path)

        assert expanded.startswith("<File: start.txt>")
        assert "Start content" in expanded
        assert files == ["start.txt"]

    def test_file_at_end_of_prompt(self, tmp_path):
        """Test @filename at the end of prompt."""
        test_file = tmp_path / "end.txt"
        test_file.write_text("End content")

        prompt = "Analyze this file: @end.txt"
        expanded, files = expand_file_references(prompt, tmp_path)

        # File contents prepended
        assert expanded.startswith("<File: end.txt>")
        assert "End content" in expanded
        # Prompt with @end.txt replaced by end.txt
        assert "Analyze this file: end.txt" in expanded
        assert files == ["end.txt"]

    def test_complex_path_with_dots(self, tmp_path):
        """Test path with dots and special characters."""
        test_file = tmp_path / "my.config.json"
        test_file.write_text('{"key": "value"}')

        prompt = "Parse @my.config.json"
        expanded, files = expand_file_references(prompt, tmp_path)

        assert '{"key": "value"}' in expanded
        assert files == ["my.config.json"]

    def test_empty_file(self, tmp_path):
        """Test expansion of empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        prompt = "Check @empty.txt"
        expanded, files = expand_file_references(prompt, tmp_path)

        assert "<File: empty.txt>" in expanded
        assert "</File: empty.txt>" in expanded
        assert files == ["empty.txt"]

    def test_large_file(self, tmp_path):
        """Test expansion of larger file."""
        test_file = tmp_path / "large.txt"
        content = "Line {}\n".format("x" * 100) * 1000  # ~100KB
        test_file.write_text(content)

        prompt = "@large.txt"
        expanded, files = expand_file_references(prompt, tmp_path)

        assert content in expanded
        assert files == ["large.txt"]

    def test_at_symbol_with_special_chars_not_expanded(self, tmp_path):
        """Test that @ followed by special characters is not treated as file reference."""
        # Prompts with @ followed by special chars should pass through unchanged
        prompt = "Email me @#$%^&*() for details"
        expanded, files = expand_file_references(prompt, tmp_path)

        # @ followed by special chars should not be expanded
        assert "@#$%^&*()" in expanded
        assert files == []

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
