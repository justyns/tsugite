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
