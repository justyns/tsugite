"""Tests for AutoContextHandler."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from tsugite.attachments.auto_context import AutoContextHandler


class TestAutoContextHandler:
    """Test AutoContextHandler functionality."""

    def test_can_handle_auto_context(self):
        """Test that handler recognizes auto-context markers."""
        handler = AutoContextHandler()
        assert handler.can_handle("auto-context")
        assert handler.can_handle("auto")
        assert handler.can_handle("auto:something")

    def test_cannot_handle_other_sources(self):
        """Test that handler rejects non-auto-context sources."""
        handler = AutoContextHandler()
        assert not handler.can_handle("file.txt")
        assert not handler.can_handle("inline")
        assert not handler.can_handle("http://example.com")

    def test_fetch_with_no_files_found(self, tmp_path, monkeypatch):
        """Test fetch_multiple when no context files are found."""
        # Change to temp directory with no context files
        monkeypatch.chdir(tmp_path)

        handler = AutoContextHandler(context_files=["CONTEXT.md", "AGENTS.md"])
        result = handler.fetch_multiple("auto-context")

        # Should return empty list, not raise an error
        assert result == []

    def test_fetch_discovers_single_file(self, tmp_path, monkeypatch):
        """Test fetch_multiple when one context file exists."""
        # Create a context file
        context_file = tmp_path / "CONTEXT.md"
        context_file.write_text("# Project Context\nThis is context.")

        monkeypatch.chdir(tmp_path)

        handler = AutoContextHandler(context_files=["CONTEXT.md", "AGENTS.md"])
        result = handler.fetch_multiple("auto-context")

        assert len(result) == 1
        name, content = result[0]
        assert name == "CONTEXT.md"
        assert "# Project Context" in content
        assert "This is context." in content

    def test_fetch_discovers_multiple_files(self, tmp_path, monkeypatch):
        """Test fetch_multiple when multiple context files exist."""
        # Create multiple context files
        (tmp_path / "CONTEXT.md").write_text("Context content")
        (tmp_path / "AGENTS.md").write_text("Agents content")
        (tmp_path / "CLAUDE.md").write_text("Claude content")

        monkeypatch.chdir(tmp_path)

        handler = AutoContextHandler(context_files=["CONTEXT.md", "AGENTS.md", "CLAUDE.md"])
        result = handler.fetch_multiple("auto-context")

        assert len(result) == 3
        names = [name for name, _ in result]
        contents = {name: content for name, content in result}

        assert "CONTEXT.md" in names
        assert "AGENTS.md" in names
        assert "CLAUDE.md" in names
        assert "Context content" in contents["CONTEXT.md"]
        assert "Agents content" in contents["AGENTS.md"]
        assert "Claude content" in contents["CLAUDE.md"]

    def test_fetch_in_tsugite_directory(self, tmp_path, monkeypatch):
        """Test fetch_multiple finds files in .tsugite directory."""
        # Create .tsugite directory with context
        tsugite_dir = tmp_path / ".tsugite"
        tsugite_dir.mkdir()
        (tsugite_dir / "CONTEXT.md").write_text("Tsugite context")

        monkeypatch.chdir(tmp_path)

        handler = AutoContextHandler(context_files=[".tsugite/CONTEXT.md"])
        result = handler.fetch_multiple("auto-context")

        assert len(result) == 1
        name, content = result[0]
        assert name == ".tsugite/CONTEXT.md"
        assert "Tsugite context" in content

    def test_fetch_prefers_closer_files(self, tmp_path, monkeypatch):
        """Test that files in current dir are preferred over parent dirs."""
        # Create parent directory with context
        parent_dir = tmp_path / "parent"
        parent_dir.mkdir()
        (parent_dir / "CONTEXT.md").write_text("Parent context")

        # Create child directory with different context
        child_dir = parent_dir / "child"
        child_dir.mkdir()
        (child_dir / "CONTEXT.md").write_text("Child context")

        monkeypatch.chdir(child_dir)

        handler = AutoContextHandler(context_files=["CONTEXT.md"])
        result = handler.fetch_multiple("auto-context")

        assert len(result) == 1
        name, content = result[0]
        assert name == "CONTEXT.md"
        # Should find child version, not parent
        assert "Child context" in content
        assert "Parent context" not in content

    @patch("tsugite.attachments.auto_context.subprocess.run")
    def test_find_git_root_success(self, mock_run, tmp_path):
        """Test finding git root successfully."""
        mock_run.return_value = MagicMock(returncode=0, stdout="/path/to/repo\n", stderr="")

        handler = AutoContextHandler()
        git_root = handler._find_git_root(tmp_path)

        assert git_root == Path("/path/to/repo")
        mock_run.assert_called_once()

    @patch("tsugite.attachments.auto_context.subprocess.run")
    def test_find_git_root_not_in_repo(self, mock_run, tmp_path):
        """Test finding git root when not in a git repository."""
        mock_run.return_value = MagicMock(returncode=128, stdout="", stderr="")

        handler = AutoContextHandler()
        git_root = handler._find_git_root(tmp_path)

        assert git_root is None

    def test_get_search_directories_with_git_root(self, tmp_path):
        """Test getting search directories when in a git repo."""
        # Create directory structure
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        subdir = repo_root / "src" / "module"
        subdir.mkdir(parents=True)

        handler = AutoContextHandler()
        dirs = handler._get_search_directories(subdir, repo_root)

        # Should walk from subdir up to repo root
        assert len(dirs) >= 3
        assert dirs[0] == subdir
        assert repo_root in dirs

    def test_get_search_directories_without_git_root(self, tmp_path):
        """Test getting search directories when not in a git repo."""
        # Create deep directory structure
        deep_dir = tmp_path / "a" / "b" / "c"
        deep_dir.mkdir(parents=True)

        handler = AutoContextHandler()
        dirs = handler._get_search_directories(deep_dir, None)

        # Should walk all the way up
        assert dirs[0] == deep_dir
        assert len(dirs) >= 4

    def test_fetch_handles_unreadable_file(self, tmp_path, monkeypatch):
        """Test fetch_multiple handles files that cannot be read."""
        # Create a file and then mock it to raise an error
        context_file = tmp_path / "CONTEXT.md"
        context_file.write_text("Test")

        monkeypatch.chdir(tmp_path)

        handler = AutoContextHandler(context_files=["CONTEXT.md"])

        # Mock read_text to raise an exception
        with patch.object(Path, "read_text", side_effect=PermissionError("Access denied")):
            result = handler.fetch_multiple("auto-context")

            # Should include error attachment but not fail
            assert len(result) == 1
            name, content = result[0]
            assert name == "CONTEXT.md"
            assert "Error reading" in content
            assert "Access denied" in content

    def test_fetch_loads_config_when_no_context_files(self, tmp_path, monkeypatch):
        """Test that fetch_multiple loads config when context_files is None."""
        monkeypatch.chdir(tmp_path)

        # Create config files
        (tmp_path / "CLAUDE.md").write_text("Config content")

        # Create a mock config
        mock_cfg = MagicMock()
        mock_cfg.auto_context_files = ["CLAUDE.md"]
        mock_cfg.auto_context_include_global = False

        # Mock load_config at the import location in auto_context module
        with patch("tsugite.config.load_config", return_value=mock_cfg):
            handler = AutoContextHandler(context_files=None)
            result = handler.fetch_multiple("auto-context")

            assert len(result) == 1
            name, content = result[0]
            assert name == "CLAUDE.md"
            assert "Config content" in content

    def test_get_global_context_file_exists(self, tmp_path):
        """Test finding global context file when it exists."""
        # Create a fake global context file
        global_context = tmp_path / "CONTEXT.md"
        global_context.write_text("Global context content")

        handler = AutoContextHandler()

        # Mock get_xdg_config_path to return our test path
        with patch("tsugite.xdg.get_xdg_config_path", return_value=global_context):
            result = handler._get_global_context_file()

            assert result is not None
            assert result[0] == global_context
            assert "Global Context" in result[1]

    def test_get_global_context_file_not_exists(self, tmp_path):
        """Test when global context file doesn't exist."""
        non_existent = tmp_path / "CONTEXT.md"

        handler = AutoContextHandler()

        with patch("tsugite.xdg.get_xdg_config_path", return_value=non_existent):
            result = handler._get_global_context_file()

            assert result is None

    def test_fetch_includes_global_context(self, tmp_path, monkeypatch):
        """Test that fetch includes global context file when enabled."""
        monkeypatch.chdir(tmp_path)

        # Create project context
        (tmp_path / "AGENTS.md").write_text("Project agents")

        # Create global context
        global_context = tmp_path / "global" / "CONTEXT.md"
        global_context.parent.mkdir()
        global_context.write_text("Global context")

        # Mock config
        mock_cfg = MagicMock()
        mock_cfg.auto_context_files = ["AGENTS.md"]
        mock_cfg.auto_context_include_global = True

        with patch("tsugite.config.load_config", return_value=mock_cfg):
            with patch("tsugite.xdg.get_xdg_config_path", return_value=global_context):
                handler = AutoContextHandler(context_files=None)
                result = handler.fetch_multiple("auto-context")

                # Should include both project and global context
                assert len(result) == 2
                names = [name for name, _ in result]
                contents = {name: content for name, content in result}
                assert "AGENTS.md" in names
                assert "Global Context (~/.config/tsugite/CONTEXT.md)" in names
                assert "Project agents" in contents["AGENTS.md"]
                assert "Global context" in contents["Global Context (~/.config/tsugite/CONTEXT.md)"]

    def test_fetch_excludes_global_context_when_disabled(self, tmp_path, monkeypatch):
        """Test that fetch_multiple excludes global context when disabled."""
        monkeypatch.chdir(tmp_path)

        # Create project context
        (tmp_path / "AGENTS.md").write_text("Project agents")

        # Create global context
        global_context = tmp_path / "global" / "CONTEXT.md"
        global_context.parent.mkdir()
        global_context.write_text("Global context")

        # Mock config with global disabled
        mock_cfg = MagicMock()
        mock_cfg.auto_context_files = ["AGENTS.md"]
        mock_cfg.auto_context_include_global = False

        with patch("tsugite.config.load_config", return_value=mock_cfg):
            with patch("tsugite.xdg.get_xdg_config_path", return_value=global_context):
                handler = AutoContextHandler(context_files=None)
                result = handler.fetch_multiple("auto-context")

                # Should only include project context
                assert len(result) == 1
                name, content = result[0]
                assert name == "AGENTS.md"
                assert "Project agents" in content
                assert "Global context" not in content

    def test_fetch_multiple_returns_list_of_tuples(self, tmp_path, monkeypatch):
        """Test that fetch_multiple returns list of (name, content) tuples."""
        # Create multiple context files
        (tmp_path / "CONTEXT.md").write_text("Context content")
        (tmp_path / "AGENTS.md").write_text("Agents content")

        monkeypatch.chdir(tmp_path)

        handler = AutoContextHandler(context_files=["CONTEXT.md", "AGENTS.md"])
        result = handler.fetch_multiple("auto-context")

        # Should return list of tuples
        assert isinstance(result, list)
        assert len(result) == 2

        # Check structure
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

        # Check names and contents
        names = [item[0] for item in result]
        contents = [item[1] for item in result]

        assert "CONTEXT.md" in names
        assert "AGENTS.md" in names
        assert "Context content" in contents
        assert "Agents content" in contents

    def test_fetch_multiple_with_single_file(self, tmp_path, monkeypatch):
        """Test fetch_multiple with only one file found."""
        (tmp_path / "CONTEXT.md").write_text("Single file")

        monkeypatch.chdir(tmp_path)

        handler = AutoContextHandler(context_files=["CONTEXT.md", "AGENTS.md"])
        result = handler.fetch_multiple("auto-context")

        assert len(result) == 1
        assert result[0] == ("CONTEXT.md", "Single file")

    def test_fetch_multiple_with_no_files(self, tmp_path, monkeypatch):
        """Test fetch_multiple when no files are found."""
        monkeypatch.chdir(tmp_path)

        handler = AutoContextHandler(context_files=["CONTEXT.md"])
        result = handler.fetch_multiple("auto-context")

        # Should return empty list
        assert result == []

    def test_fetch_multiple_preserves_file_names(self, tmp_path, monkeypatch):
        """Test that fetch_multiple uses correct display names."""
        # Create files including .tsugite directory
        tsugite_dir = tmp_path / ".tsugite"
        tsugite_dir.mkdir()
        (tsugite_dir / "CONTEXT.md").write_text("Tsugite context")
        (tmp_path / "AGENTS.md").write_text("Agents content")

        monkeypatch.chdir(tmp_path)

        handler = AutoContextHandler(context_files=[".tsugite/CONTEXT.md", "AGENTS.md"])
        result = handler.fetch_multiple("auto-context")

        names = [item[0] for item in result]
        assert ".tsugite/CONTEXT.md" in names
        assert "AGENTS.md" in names

    def test_fetch_multiple_with_global_context(self, tmp_path, monkeypatch):
        """Test that fetch_multiple includes global context with proper name."""
        monkeypatch.chdir(tmp_path)

        # Create project context
        (tmp_path / "AGENTS.md").write_text("Project agents")

        # Create global context
        global_context = tmp_path / "global" / "CONTEXT.md"
        global_context.parent.mkdir()
        global_context.write_text("Global context")

        # Mock config
        mock_cfg = MagicMock()
        mock_cfg.auto_context_files = ["AGENTS.md"]
        mock_cfg.auto_context_include_global = True

        with patch("tsugite.config.load_config", return_value=mock_cfg):
            with patch("tsugite.xdg.get_xdg_config_path", return_value=global_context):
                handler = AutoContextHandler(context_files=None)
                result = handler.fetch_multiple("auto-context")

                # Should have two attachments
                assert len(result) == 2

                names = [item[0] for item in result]
                contents = [item[1] for item in result]

                assert "AGENTS.md" in names
                assert "Global Context (~/.config/tsugite/CONTEXT.md)" in names
                assert "Project agents" in contents
                assert "Global context" in contents

    def test_fetch_multiple_handles_read_error_gracefully(self, tmp_path, monkeypatch):
        """Test that fetch_multiple handles unreadable files with error content."""
        context_file = tmp_path / "CONTEXT.md"
        context_file.write_text("Test")

        monkeypatch.chdir(tmp_path)

        handler = AutoContextHandler(context_files=["CONTEXT.md"])

        # Mock read_text to raise an exception
        with patch.object(Path, "read_text", side_effect=PermissionError("Access denied")):
            result = handler.fetch_multiple("auto-context")

            # Should still return the file with error content
            assert len(result) == 1
            name, content = result[0]
            assert name == "CONTEXT.md"
            assert "Error reading CONTEXT.md" in content
            assert "Access denied" in content

    def test_fetch_multiple_orders_files_correctly(self, tmp_path, monkeypatch):
        """Test that fetch_multiple returns files in discovery order."""
        # Create files in specific order
        (tmp_path / "CONTEXT.md").write_text("First")
        (tmp_path / "AGENTS.md").write_text("Second")
        (tmp_path / "CLAUDE.md").write_text("Third")

        monkeypatch.chdir(tmp_path)

        handler = AutoContextHandler(context_files=["CONTEXT.md", "AGENTS.md", "CLAUDE.md"])
        result = handler.fetch_multiple("auto-context")

        # Check order matches config order
        names = [item[0] for item in result]
        assert names == ["CONTEXT.md", "AGENTS.md", "CLAUDE.md"]
