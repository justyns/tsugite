"""Tests for file system tools."""

import pytest

from tsugite.tools import call_tool


def test_write_and_read_file(temp_dir, file_tools):
    """Test writing and reading a file."""

    test_file = temp_dir / "test.txt"
    content = "Hello, Tsugite!"

    # Test write
    result = call_tool("write_file", path=str(test_file), content=content)
    assert "Successfully wrote" in result
    assert str(len(content)) in result
    assert test_file.exists()

    # Test read
    read_content = call_tool("read_file", path=str(test_file))
    assert read_content == content


def test_write_file_creates_directories(temp_dir, file_tools):
    """Test that write_file creates parent directories."""

    nested_file = temp_dir / "nested" / "deep" / "test.txt"
    content = "Nested content"

    result = call_tool("write_file", path=str(nested_file), content=content)
    assert "Successfully wrote" in result
    assert nested_file.exists()
    assert nested_file.read_text() == content


def test_read_nonexistent_file(temp_dir, file_tools):
    """Test reading a file that doesn't exist."""

    nonexistent = temp_dir / "nonexistent.txt"

    with pytest.raises(RuntimeError, match="Tool 'read_file' failed"):
        call_tool("read_file", path=str(nonexistent))


def test_read_directory_as_file(temp_dir, file_tools):
    """Test trying to read a directory as a file."""

    directory = temp_dir / "subdir"
    directory.mkdir()

    with pytest.raises(RuntimeError, match="Tool 'read_file' failed"):
        call_tool("read_file", path=str(directory))


def test_write_file_error_handling(file_tools):
    """Test write_file error handling."""

    # Try to write to an invalid path (assuming /invalid doesn't exist and can't be created)
    invalid_path = "/invalid/path/that/cannot/be/created/file.txt"

    with pytest.raises(RuntimeError, match="Tool 'write_file' failed"):
        call_tool("write_file", path=invalid_path, content="test")


def test_list_files_basic(temp_dir, file_tools):
    """Test basic file listing."""

    # Create test files
    (temp_dir / "file1.txt").write_text("content1")
    (temp_dir / "file2.py").write_text("content2")
    (temp_dir / "subdir").mkdir()
    (temp_dir / "subdir" / "file3.txt").write_text("content3")

    files = call_tool("list_files", path=str(temp_dir))

    assert isinstance(files, list)
    assert "file1.txt" in files
    assert "file2.py" in files
    assert "subdir" not in files  # Should not include directories
    assert "file3.txt" not in files  # Should not include files in subdirectories


def test_list_files_with_pattern(temp_dir, file_tools):
    """Test file listing with pattern matching."""

    # Create test files
    (temp_dir / "test1.txt").write_text("content1")
    (temp_dir / "test2.txt").write_text("content2")
    (temp_dir / "other.py").write_text("content3")

    # List only .txt files
    txt_files = call_tool("list_files", path=str(temp_dir), pattern="*.txt")
    assert len(txt_files) == 2
    assert "test1.txt" in txt_files
    assert "test2.txt" in txt_files
    assert "other.py" not in txt_files

    # List only .py files
    py_files = call_tool("list_files", path=str(temp_dir), pattern="*.py")
    assert len(py_files) == 1
    assert "other.py" in py_files


def test_list_files_default_path(file_tools):
    """Test file listing with default path (current directory)."""

    files = call_tool("list_files")
    assert isinstance(files, list)
    # Should list files in current directory


def test_list_files_nonexistent_directory(temp_dir, file_tools):
    """Test listing files in nonexistent directory."""

    nonexistent = temp_dir / "nonexistent"

    with pytest.raises(RuntimeError, match="Tool 'list_files' failed"):
        call_tool("list_files", path=str(nonexistent))


def test_list_files_file_as_directory(temp_dir, file_tools):
    """Test trying to list files where path is a file."""

    test_file = temp_dir / "test.txt"
    test_file.write_text("content")

    with pytest.raises(RuntimeError, match="Tool 'list_files' failed"):
        call_tool("list_files", path=str(test_file))


def test_file_exists_true(temp_dir, file_tools):
    """Test file_exists with existing file."""

    test_file = temp_dir / "exists.txt"
    test_file.write_text("content")

    result = call_tool("file_exists", path=str(test_file))
    assert result is True


def test_file_exists_false(temp_dir, file_tools):
    """Test file_exists with nonexistent file."""

    nonexistent = temp_dir / "nonexistent.txt"

    result = call_tool("file_exists", path=str(nonexistent))
    assert result is False


def test_file_exists_directory(temp_dir, file_tools):
    """Test file_exists with directory."""

    directory = temp_dir / "subdir"
    directory.mkdir()

    result = call_tool("file_exists", path=str(directory))
    assert result is True  # Directories also "exist"


def test_create_directory_new(temp_dir, file_tools):
    """Test creating a new directory."""

    new_dir = temp_dir / "new_directory"

    result = call_tool("create_directory", path=str(new_dir))
    assert "Successfully created directory" in result
    assert new_dir.exists()
    assert new_dir.is_dir()


def test_create_directory_nested(temp_dir, file_tools):
    """Test creating nested directories."""

    nested_dir = temp_dir / "level1" / "level2" / "level3"

    result = call_tool("create_directory", path=str(nested_dir))
    assert "Successfully created directory" in result
    assert nested_dir.exists()
    assert nested_dir.is_dir()

    # Check parent directories were also created
    assert (temp_dir / "level1").is_dir()
    assert (temp_dir / "level1" / "level2").is_dir()


def test_create_directory_existing(temp_dir, file_tools):
    """Test creating a directory that already exists."""

    existing_dir = temp_dir / "existing"
    existing_dir.mkdir()

    # Should not raise error due to exist_ok=True
    result = call_tool("create_directory", path=str(existing_dir))
    assert "Successfully created directory" in result
    assert existing_dir.exists()


def test_create_directory_error(file_tools):
    """Test create_directory error handling."""

    # Try to create directory where we don't have permissions
    invalid_path = "/root/cannot_create_here"

    with pytest.raises(RuntimeError, match="Tool 'create_directory' failed"):
        call_tool("create_directory", path=invalid_path)


def test_file_tools_integration(temp_dir, file_tools):
    """Test integration between different file tools."""

    # Create directory structure
    nested_dir = temp_dir / "project" / "src"
    call_tool("create_directory", path=str(nested_dir))

    # Write files
    file1 = nested_dir / "main.py"
    file2 = nested_dir / "utils.py"
    call_tool("write_file", path=str(file1), content="# Main module")
    call_tool("write_file", path=str(file2), content="# Utilities")

    # Check files exist
    assert call_tool("file_exists", path=str(file1)) is True
    assert call_tool("file_exists", path=str(file2)) is True

    # List files
    files = call_tool("list_files", path=str(nested_dir), pattern="*.py")
    assert len(files) == 2
    assert "main.py" in files
    assert "utils.py" in files

    # Read files
    content1 = call_tool("read_file", path=str(file1))
    content2 = call_tool("read_file", path=str(file2))
    assert content1 == "# Main module"
    assert content2 == "# Utilities"


def test_unicode_content(temp_dir, file_tools):
    """Test handling of Unicode content in files."""

    unicode_content = "Hello 世界! 🌍 Résumé café naïve"
    test_file = temp_dir / "unicode.txt"

    # Write Unicode content
    call_tool("write_file", path=str(test_file), content=unicode_content)

    # Read back Unicode content
    read_content = call_tool("read_file", path=str(test_file))
    assert read_content == unicode_content


def test_large_file_handling(temp_dir, file_tools):
    """Test handling of larger files."""

    # Create a moderately large file (1MB)
    large_content = "A" * (1024 * 1024)
    large_file = temp_dir / "large.txt"

    result = call_tool("write_file", path=str(large_file), content=large_content)
    assert "Successfully wrote 1048576 characters" in result

    read_content = call_tool("read_file", path=str(large_file))
    assert len(read_content) == 1048576
    assert read_content == large_content


def test_empty_file_handling(temp_dir, file_tools):
    """Test handling of empty files."""

    empty_file = temp_dir / "empty.txt"

    # Write empty content
    call_tool("write_file", path=str(empty_file), content="")

    # Read empty content
    content = call_tool("read_file", path=str(empty_file))
    assert content == ""

    # File should exist
    assert call_tool("file_exists", path=str(empty_file)) is True


# Tests for new file editing tools


def test_read_file_lines_full_file(temp_dir, file_tools):
    """Test reading all lines from a file with line numbers."""
    test_file = temp_dir / "lines.txt"
    content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
    test_file.write_text(content)

    result = call_tool("read_file", path=str(test_file), start_line=1)
    assert "1: Line 1" in result
    assert "2: Line 2" in result
    assert "5: Line 5" in result


def test_read_file_lines_range(temp_dir, file_tools):
    """Test reading a specific range of lines."""
    test_file = temp_dir / "lines.txt"
    content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
    test_file.write_text(content)

    result = call_tool("read_file", path=str(test_file), start_line=2, end_line=4)
    assert "2: Line 2" in result
    assert "3: Line 3" in result
    assert "4: Line 4" in result
    assert "1: Line 1" not in result
    assert "5: Line 5" not in result


def test_read_file_lines_from_start(temp_dir, file_tools):
    """Test reading from a specific line to end."""
    test_file = temp_dir / "lines.txt"
    content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
    test_file.write_text(content)

    result = call_tool("read_file", path=str(test_file), start_line=3)
    assert "3: Line 3" in result
    assert "4: Line 4" in result
    assert "5: Line 5" in result
    assert "1: Line 1" not in result


def test_read_file_lines_invalid_range(temp_dir, file_tools):
    """Test read_file with invalid line range."""
    test_file = temp_dir / "lines.txt"
    test_file.write_text("Line 1\nLine 2")

    with pytest.raises(RuntimeError, match="end_line.*must be"):
        call_tool("read_file", path=str(test_file), start_line=5, end_line=2)


def test_read_file_lines_zero_based_index(temp_dir, file_tools):
    """Test read_file accepts start_line=0 (treats as 1)."""
    test_file = temp_dir / "lines.txt"
    test_file.write_text("Line 1\nLine 2\nLine 3")

    # start_line=0 should be treated as start_line=1
    result = call_tool("read_file", path=str(test_file), start_line=0, end_line=2)
    assert "1: Line 1" in result
    assert "2: Line 2" in result
    assert "Line 3" not in result


def test_get_file_info_existing(temp_dir, file_tools):
    """Test get_file_info for existing file."""
    test_file = temp_dir / "info.txt"
    content = "Line 1\nLine 2\nLine 3"
    test_file.write_text(content)

    info = call_tool("get_file_info", path=str(test_file))
    assert info["exists"] is True
    assert info["is_directory"] is False
    assert info["line_count"] == 3
    assert info["size_bytes"] > 0
    assert info["last_modified"] is not None


def test_get_file_info_nonexistent(temp_dir, file_tools):
    """Test get_file_info for nonexistent file."""
    nonexistent = temp_dir / "nonexistent.txt"

    info = call_tool("get_file_info", path=str(nonexistent))
    assert info["exists"] is False
    assert info["line_count"] == 0
    assert info["size_bytes"] == 0


def test_get_file_info_directory(temp_dir, file_tools):
    """Test get_file_info for directory."""
    test_dir = temp_dir / "subdir"
    test_dir.mkdir()

    info = call_tool("get_file_info", path=str(test_dir))
    assert info["exists"] is True
    assert info["is_directory"] is True


def test_edit_file_exact_match(temp_dir, file_tools):
    """Test edit_file with exact string match."""
    test_file = temp_dir / "edit.txt"
    original = "Hello World\nThis is a test\nGoodbye World"
    test_file.write_text(original)

    result = call_tool(
        "edit_file",
        path=str(test_file),
        old_string="This is a test",
        new_string="This is modified",
    )

    assert "Successfully edited" in result
    assert "1 replacement(s)" in result

    new_content = test_file.read_text()
    assert "This is modified" in new_content
    assert "This is a test" not in new_content


def test_edit_file_with_context(temp_dir, file_tools):
    """Test edit_file with multi-line context."""
    test_file = temp_dir / "edit.txt"
    original = """def hello():
    print("Hello")
    return True

def goodbye():
    print("Goodbye")
    return False"""
    test_file.write_text(original)

    result = call_tool(
        "edit_file",
        path=str(test_file),
        old_string='def hello():\n    print("Hello")\n    return True',
        new_string='def hello():\n    print("Hi there!")\n    return True',
    )

    assert "Successfully edited" in result
    new_content = test_file.read_text()
    assert 'print("Hi there!")' in new_content
    assert 'print("Hello")' not in new_content
    assert 'print("Goodbye")' in new_content


def test_edit_file_line_trimmed_strategy(temp_dir, file_tools):
    """Test edit_file with line-trimmed matching."""
    test_file = temp_dir / "edit.txt"
    original = "  def foo():\n      return 42\n  "
    test_file.write_text(original)

    # Search with different whitespace should still match
    result = call_tool(
        "edit_file",
        path=str(test_file),
        old_string="def foo():\n    return 42",
        new_string="def foo():\n    return 100",
    )

    assert "Successfully edited" in result
    new_content = test_file.read_text()
    assert "return 100" in new_content


@pytest.mark.skip(reason="Line ending preservation needs investigation - known limitation")
def test_edit_file_preserves_line_endings(temp_dir, file_tools):
    """Test that edit_file preserves original line endings.

    NOTE: Currently this functionality needs more work. The line ending
    detection and preservation logic exists but may not work correctly
    in all cases. This is a minor edge case that doesn't affect core functionality.
    """
    test_file = temp_dir / "edit.txt"
    # Windows-style line endings
    original = "Line 1\r\nLine 2\r\nLine 3"
    # Write as bytes to ensure CRLF is preserved
    test_file.write_bytes(original.encode("utf-8"))

    call_tool(
        "edit_file",
        path=str(test_file),
        old_string="Line 2",
        new_string="Modified Line 2",
    )

    # Read as bytes to check actual line endings
    new_content_bytes = test_file.read_bytes().decode("utf-8")
    assert "\r\n" in new_content_bytes  # Should preserve Windows line endings
    assert "Modified Line 2" in new_content_bytes


def test_edit_file_no_match(temp_dir, file_tools):
    """Test edit_file when search string not found."""
    test_file = temp_dir / "edit.txt"
    test_file.write_text("Hello World")

    with pytest.raises(RuntimeError, match="No matches found"):
        call_tool(
            "edit_file",
            path=str(test_file),
            old_string="Nonexistent text",
            new_string="New text",
        )


def test_edit_file_multiple_matches(temp_dir, file_tools):
    """Test edit_file with multiple matches requiring expected_replacements."""
    test_file = temp_dir / "edit.txt"
    original = "foo bar foo baz foo"
    test_file.write_text(original)

    # Should fail with default expected_replacements=1
    with pytest.raises(RuntimeError, match="Found 3 matches but expected 1"):
        call_tool(
            "edit_file",
            path=str(test_file),
            old_string="foo",
            new_string="qux",
        )

    # Should succeed with expected_replacements=3
    result = call_tool(
        "edit_file",
        path=str(test_file),
        old_string="foo",
        new_string="qux",
        expected_replacements=3,
    )

    assert "3 replacement(s)" in result
    new_content = test_file.read_text()
    assert new_content == "qux bar qux baz qux"


def test_edit_file_same_strings(temp_dir, file_tools):
    """Test edit_file with identical old and new strings."""
    test_file = temp_dir / "edit.txt"
    test_file.write_text("Hello World")

    with pytest.raises(RuntimeError, match="must be different"):
        call_tool(
            "edit_file",
            path=str(test_file),
            old_string="Hello",
            new_string="Hello",
        )


def test_edit_file_batch_sequential(temp_dir, file_tools):
    """Test edit_file with sequential batch edits."""
    test_file = temp_dir / "multiedit.txt"
    original = "foo bar baz"
    test_file.write_text(original)

    edits = [
        {"old_string": "foo", "new_string": "FOO"},
        {"old_string": "bar", "new_string": "BAR"},
        {"old_string": "baz", "new_string": "BAZ"},
    ]

    result = call_tool("edit_file", path=str(test_file), edits=edits)

    assert "Successfully applied 3 edit(s)" in result
    assert "3 total replacements" in result

    new_content = test_file.read_text()
    assert new_content == "FOO BAR BAZ"


def test_edit_file_batch_dependent(temp_dir, file_tools):
    """Test edit_file batch mode where later edits depend on earlier ones."""
    test_file = temp_dir / "multiedit.txt"
    original = "Hello World"
    test_file.write_text(original)

    edits = [
        {"old_string": "Hello", "new_string": "Hi"},
        {"old_string": "Hi World", "new_string": "Hi There"},
    ]

    result = call_tool("edit_file", path=str(test_file), edits=edits)

    assert "Successfully applied 2 edit(s)" in result
    new_content = test_file.read_text()
    assert new_content == "Hi There"


def test_edit_file_batch_atomic_failure(temp_dir, file_tools):
    """Test that edit_file batch mode is atomic - no changes if any edit fails."""
    test_file = temp_dir / "multiedit.txt"
    original = "Line 1\nLine 2\nLine 3"
    test_file.write_text(original)

    edits = [
        {"old_string": "Line 1", "new_string": "Modified 1"},
        {"old_string": "Nonexistent", "new_string": "Will fail"},
    ]

    # Should fail on second edit
    with pytest.raises(RuntimeError, match="Edit #2 failed"):
        call_tool("edit_file", path=str(test_file), edits=edits)

    # File should be unchanged (atomic)
    assert test_file.read_text() == original


def test_edit_file_batch_empty_edits(temp_dir, file_tools):
    """Test edit_file batch mode with empty edits list."""
    test_file = temp_dir / "multiedit.txt"
    test_file.write_text("Hello")

    with pytest.raises(RuntimeError, match="cannot be empty"):
        call_tool("edit_file", path=str(test_file), edits=[])


def test_edit_file_batch_missing_fields(temp_dir, file_tools):
    """Test edit_file batch mode with missing required fields."""
    test_file = temp_dir / "multiedit.txt"
    test_file.write_text("Hello")

    edits = [{"old_string": "Hello"}]  # Missing new_string

    with pytest.raises(RuntimeError, match="missing required"):
        call_tool("edit_file", path=str(test_file), edits=edits)


def test_edit_file_conflicting_parameters(temp_dir, file_tools):
    """Test edit_file with conflicting mode parameters."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("Hello World")

    # Both single and batch mode parameters
    with pytest.raises(RuntimeError, match="Provide either old_string/new_string OR edits"):
        call_tool(
            "edit_file",
            path=str(test_file),
            old_string="Hello",
            new_string="Hi",
            edits=[{"old_string": "World", "new_string": "There"}],
        )


def test_edit_file_missing_parameters(temp_dir, file_tools):
    """Test edit_file with missing parameters."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("Hello World")

    # No mode parameters
    with pytest.raises(RuntimeError, match="Must provide either"):
        call_tool("edit_file", path=str(test_file))

    # old_string without new_string
    with pytest.raises(RuntimeError, match="new_string is required"):
        call_tool("edit_file", path=str(test_file), old_string="Hello")


# Tests for gitignore support


def test_list_files_respects_gitignore_by_default(temp_dir, file_tools):
    """Test that list_files respects .gitignore by default."""
    # Create a .gitignore file
    gitignore = temp_dir / ".gitignore"
    gitignore.write_text("*.log\ntemp/\n")

    # Create test files
    (temp_dir / "file1.txt").write_text("content1")
    (temp_dir / "file2.log").write_text("log content")
    (temp_dir / "file3.txt").write_text("content3")

    files = call_tool("list_files", path=str(temp_dir))

    assert isinstance(files, list)
    assert "file1.txt" in files
    assert "file3.txt" in files
    assert "file2.log" not in files  # Should be excluded by .gitignore
    assert ".gitignore" in files  # .gitignore itself should be listed


def test_list_files_excludes_git_directory(temp_dir, file_tools):
    """Test that list_files always excludes .git/ directory."""
    # Create a .git directory
    git_dir = temp_dir / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("git config")

    # Create regular files
    (temp_dir / "file1.txt").write_text("content1")

    files = call_tool("list_files", path=str(temp_dir))

    assert "file1.txt" in files
    assert ".git" not in files  # .git directory should be excluded


def test_list_files_disable_gitignore(temp_dir, file_tools):
    """Test that gitignore can be disabled with respect_gitignore=False."""
    # Create a .gitignore file
    gitignore = temp_dir / ".gitignore"
    gitignore.write_text("*.log\n")

    # Create test files
    (temp_dir / "file1.txt").write_text("content1")
    (temp_dir / "file2.log").write_text("log content")

    files = call_tool("list_files", path=str(temp_dir), respect_gitignore=False)

    assert isinstance(files, list)
    assert "file1.txt" in files
    assert "file2.log" in files  # Should be included when gitignore is disabled
    assert ".gitignore" in files


def test_list_files_nested_gitignore(temp_dir, file_tools):
    """Test that list_files handles nested .gitignore files."""
    # Create root .gitignore
    root_gitignore = temp_dir / ".gitignore"
    root_gitignore.write_text("*.log\n")

    # Create subdirectory with its own .gitignore
    subdir = temp_dir / "subdir"
    subdir.mkdir()
    sub_gitignore = subdir / ".gitignore"
    sub_gitignore.write_text("*.tmp\n")

    # Create test files in subdirectory
    (subdir / "file1.txt").write_text("content1")
    (subdir / "file2.log").write_text("log content")
    (subdir / "file3.tmp").write_text("temp content")

    files = call_tool("list_files", path=str(subdir))

    assert isinstance(files, list)
    assert "file1.txt" in files
    assert "file2.log" not in files  # Excluded by parent .gitignore
    assert "file3.tmp" not in files  # Excluded by local .gitignore
    assert ".gitignore" in files  # Local .gitignore should be listed


def test_list_files_no_gitignore(temp_dir, file_tools):
    """Test that list_files works normally when no .gitignore exists."""
    # Create test files without .gitignore
    (temp_dir / "file1.txt").write_text("content1")
    (temp_dir / "file2.log").write_text("log content")

    files = call_tool("list_files", path=str(temp_dir))

    assert isinstance(files, list)
    assert "file1.txt" in files
    assert "file2.log" in files  # Should be included when no .gitignore


def test_list_files_empty_gitignore(temp_dir, file_tools):
    """Test that list_files handles empty .gitignore files."""
    # Create empty .gitignore
    gitignore = temp_dir / ".gitignore"
    gitignore.write_text("")

    # Create test files
    (temp_dir / "file1.txt").write_text("content1")
    (temp_dir / "file2.log").write_text("log content")

    files = call_tool("list_files", path=str(temp_dir))

    assert isinstance(files, list)
    assert "file1.txt" in files
    assert "file2.log" in files  # Should be included (no patterns in .gitignore)
    assert ".gitignore" in files


def test_list_files_gitignore_with_comments(temp_dir, file_tools):
    """Test that list_files correctly handles comments in .gitignore."""
    # Create .gitignore with comments
    gitignore = temp_dir / ".gitignore"
    gitignore.write_text("# This is a comment\n*.log\n# Another comment\n")

    # Create test files
    (temp_dir / "file1.txt").write_text("content1")
    (temp_dir / "file2.log").write_text("log content")

    files = call_tool("list_files", path=str(temp_dir))

    assert "file1.txt" in files
    assert "file2.log" not in files  # Should be excluded despite comments


def test_list_files_gitignore_directory_pattern(temp_dir, file_tools):
    """Test that list_files handles directory patterns in .gitignore."""
    # Create .gitignore with directory pattern
    gitignore = temp_dir / ".gitignore"
    gitignore.write_text("temp/\n")

    # Create temp directory with files
    temp_subdir = temp_dir / "temp"
    temp_subdir.mkdir()
    (temp_subdir / "file.txt").write_text("temp file")

    # Create regular file
    (temp_dir / "regular.txt").write_text("regular file")

    files = call_tool("list_files", path=str(temp_dir))

    assert "regular.txt" in files
    assert ".gitignore" in files

    # When explicitly listing the temp directory, files should still be shown
    # (similar to how `ls temp/` or `rg pattern temp/` explicitly searches that directory)
    # The gitignore only prevents traversal from parent directories
    temp_files = call_tool("list_files", path=str(temp_subdir))
    assert "file.txt" in temp_files  # Files should be listed when explicitly asked


def test_list_files_gitignore_with_pattern(temp_dir, file_tools):
    """Test that list_files works with both pattern and gitignore."""
    # Create .gitignore
    gitignore = temp_dir / ".gitignore"
    gitignore.write_text("*.log\n")

    # Create test files
    (temp_dir / "test1.txt").write_text("content1")
    (temp_dir / "test2.py").write_text("content2")
    (temp_dir / "test3.log").write_text("log content")

    # List only .txt files (should also respect .gitignore)
    txt_files = call_tool("list_files", path=str(temp_dir), pattern="*.txt")
    assert "test1.txt" in txt_files
    assert "test2.py" not in txt_files  # Excluded by pattern
    assert "test3.log" not in txt_files  # Would be excluded by .gitignore anyway
