"""Auto-context handler for discovering project context files."""

import subprocess
from pathlib import Path
from typing import List, Optional

from tsugite.attachments.base import AttachmentHandler


class AutoContextHandler(AttachmentHandler):
    """Handler for auto-discovering context files.

    Searches from current directory up to git root for specific context files
    like CONTEXT.md, AGENTS.md, CLAUDE.md and concatenates them.
    """

    def __init__(self, context_files: Optional[List[str]] = None):
        """Initialize handler.

        Args:
            context_files: List of filenames to search for.
                          If None, will use config at fetch time.
        """
        self.context_files = context_files

    def can_handle(self, source: str) -> bool:
        """Check if source is auto-context marker.

        Args:
            source: Source string

        Returns:
            True if source is "auto-context" or starts with "auto:"
        """
        return source in ("auto-context", "auto") or source.startswith("auto:")

    def fetch(self, source: str) -> str:
        """Discover and concatenate context files.

        DEPRECATED: This method concatenates all files into one string.
        Use fetch_multiple() instead for separate attachments.

        Args:
            source: Auto-context marker

        Returns:
            Concatenated content from discovered files

        Raises:
            ValueError: If discovery fails
        """
        try:
            files = self._discover_context_files()
            if not files:
                return ""
            return self._concatenate_files(files)
        except Exception as e:
            raise ValueError(f"Failed to fetch auto-context: {e}")

    def fetch_multiple(self, source: str) -> List[tuple[str, str]]:
        """Discover context files and return as separate attachments.

        Args:
            source: Auto-context marker

        Returns:
            List of (attachment_name, content) tuples, one per discovered file

        Raises:
            ValueError: If discovery fails
        """
        try:
            found_files = self._discover_context_files()

            result = []
            for file_path, relative_name in found_files:
                try:
                    content = file_path.read_text(encoding="utf-8")
                    # Use the relative name as the attachment name
                    result.append((relative_name, content))
                except Exception as e:
                    # Add error as attachment content for visibility
                    error_content = f"# Error reading {relative_name}\n\n{str(e)}"
                    result.append((relative_name, error_content))

            return result
        except Exception as e:
            raise ValueError(f"Failed to fetch auto-context: {e}")

    def _discover_context_files(self) -> List[tuple[Path, str]]:
        """Discover all context files from project and global locations.

        Returns:
            List of (file_path, display_name) tuples for found files
        """
        # Load context files from config if not set at init
        context_files = self.context_files
        include_global = True
        if context_files is None:
            from tsugite.config import load_config

            config = load_config()
            context_files = config.auto_context_files
            include_global = config.auto_context_include_global

        cwd = Path.cwd()
        git_root = self._find_git_root(cwd)

        # Search from cwd up to git root (or just cwd if not in git repo)
        search_dirs = self._get_search_directories(cwd, git_root)

        # Find all project context files
        found_files = self._discover_files(search_dirs, context_files)

        # Add global context file if enabled
        if include_global:
            global_file = self._get_global_context_file()
            if global_file:
                found_files.append(global_file)

        return found_files

    def _find_git_root(self, start_dir: Path) -> Optional[Path]:
        """Find git repository root.

        Args:
            start_dir: Directory to start searching from

        Returns:
            Path to git root, or None if not in a git repository
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"], cwd=start_dir, capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                return Path(result.stdout.strip())
            return None
        except (subprocess.SubprocessError, FileNotFoundError):
            return None

    def _get_search_directories(self, cwd: Path, git_root: Optional[Path]) -> List[Path]:
        """Get list of directories to search for context files.

        Walks from cwd up to git root (inclusive).

        Args:
            cwd: Current working directory
            git_root: Git repository root, or None

        Returns:
            List of directories to search, from most specific to most general
        """
        dirs = []
        current = cwd

        while True:
            dirs.append(current)

            # Stop at git root if we have one
            if git_root and current == git_root:
                break

            # Stop at filesystem root
            parent = current.parent
            if parent == current:
                break

            current = parent

        return dirs

    def _discover_files(self, search_dirs: List[Path], context_files: List[str]) -> List[tuple[Path, str]]:
        """Discover context files in search directories.

        Args:
            search_dirs: Directories to search
            context_files: List of filenames to search for

        Returns:
            List of (file_path, relative_name) tuples for found files
        """
        found = []
        seen_names = set()

        # Search in order (most specific directory first)
        for directory in search_dirs:
            for filename in context_files:
                # Skip if we've already found a file with this name
                if filename in seen_names:
                    continue

                file_path = directory / filename
                if file_path.exists() and file_path.is_file():
                    found.append((file_path, filename))
                    seen_names.add(filename)

        return found

    def _get_global_context_file(self) -> Optional[tuple[Path, str]]:
        """Get global context file from user's config directory.

        Returns:
            Tuple of (file_path, "Global Context") if file exists, None otherwise
        """
        from tsugite.xdg import get_xdg_config_path

        global_context_path = get_xdg_config_path("CONTEXT.md")
        if global_context_path.exists() and global_context_path.is_file():
            return (global_context_path, "Global Context (~/.config/tsugite/CONTEXT.md)")
        return None

    def _concatenate_files(self, files: List[tuple[Path, str]]) -> str:
        """Concatenate multiple files with headers.

        Args:
            files: List of (file_path, relative_name) tuples

        Returns:
            Concatenated content with headers
        """
        sections = []

        for file_path, relative_name in files:
            try:
                content = file_path.read_text(encoding="utf-8")
                # Add header and content
                sections.append(f"# Auto-Context: {relative_name}\n")
                sections.append(f"# Source: {file_path}\n\n")
                sections.append(content)
                sections.append("\n")
            except Exception as e:
                # Log warning but continue with other files
                sections.append(f"# Warning: Failed to read {relative_name}: {e}\n\n")

        return "".join(sections)
