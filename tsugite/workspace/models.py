"""Convention-based workspace models."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

WORKSPACE_FILES = ["SOUL.md", "USER.md", "MEMORY.md", "IDENTITY.md", "AGENTS.md"]
DEFAULT_MEMORY_INJECT_DAYS = 3
DEFAULT_COMPACTION_THRESHOLD = 0.8


class WorkspaceNotFoundError(ValueError):
    """Raised when workspace cannot be found."""

    pass


@dataclass
class Workspace:
    """Convention-based workspace.

    Workspaces no longer require config.yaml - everything is discovered by convention.
    """

    name: str
    path: Path

    @property
    def session_path(self) -> Path:
        """Path to active session file."""
        return self.path / "session.jsonl"

    @property
    def sessions_dir(self) -> Path:
        """Path to archived sessions directory."""
        return self.path / "sessions"

    @property
    def memory_dir(self) -> Path:
        """Path to memory files directory."""
        return self.path / "memory"

    @property
    def skills_dir(self) -> Path:
        """Path to workspace-specific skills directory."""
        return self.path / "skills"

    @property
    def agents_dir(self) -> Path:
        """Path to workspace-specific agents directory."""
        return self.path / "agents"

    def get_workspace_files(self) -> List[Path]:
        """Auto-discover workspace identity files (SOUL.md, USER.md, etc.).

        Returns:
            List of existing workspace files in conventional order
        """
        files = []
        for filename in WORKSPACE_FILES:
            path = self.path / filename
            if path.exists() and path.is_file():
                files.append(path)
        return files

    def get_memory_files(self, days: int = DEFAULT_MEMORY_INJECT_DAYS) -> List[Path]:
        """Get recent memory files matching YYYY-MM-DD.md pattern.

        Args:
            days: Number of days to look back

        Returns:
            List of memory files from the last N days, sorted chronologically
        """
        if not self.memory_dir.exists():
            return []

        cutoff = datetime.now() - timedelta(days=days)
        files = []

        for path in sorted(self.memory_dir.glob("*.md")):
            try:
                date_str = path.stem
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if file_date >= cutoff:
                    files.append(path)
            except ValueError:
                continue

        return files

    @classmethod
    def load(cls, path: Path) -> "Workspace":
        """Load workspace from path (no config.yaml required).

        Args:
            path: Path to workspace directory

        Returns:
            Loaded workspace

        Raises:
            WorkspaceNotFoundError: If path does not exist or is not a directory
        """
        if not path.exists():
            raise WorkspaceNotFoundError(f"Workspace path not found: {path}")
        if not path.is_dir():
            raise WorkspaceNotFoundError(f"Workspace path is not a directory: {path}")
        return cls(name=path.name, path=path)

    @classmethod
    def create(
        cls,
        path: Path,
        soul_template: Optional[str] = None,
        user_name: Optional[str] = None,
        init_git: bool = False,
    ) -> "Workspace":
        """Create new workspace with optional soul template.

        Args:
            path: Path to create workspace at
            soul_template: Optional soul template name
            user_name: Optional user name for template rendering
            init_git: Whether to initialize git repository

        Returns:
            Created workspace
        """
        path.mkdir(parents=True, exist_ok=True)
        (path / "memory").mkdir(exist_ok=True)
        (path / "sessions").mkdir(exist_ok=True)
        (path / "skills").mkdir(exist_ok=True)
        (path / "agents").mkdir(exist_ok=True)

        # Create .gitkeep files for directories that should be tracked (sessions is gitignored)
        (path / "memory" / ".gitkeep").touch()
        (path / "skills" / ".gitkeep").touch()
        (path / "agents" / ".gitkeep").touch()

        # Create .gitignore for workspace
        gitignore_content = """# Tsugite workspace files
session.jsonl
sessions/
*.log

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# Virtual environments
venv/
.venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
"""
        (path / ".gitignore").write_text(gitignore_content)

        if soul_template:
            from .templates import load_soul_template

            soul_content = load_soul_template(soul_template, user_name=user_name)
            (path / "SOUL.md").write_text(soul_content)

        if user_name:
            (path / "USER.md").write_text(f"# {user_name}\n\n")

        # Initialize git repository if requested
        if init_git:
            import subprocess

            try:
                # Initialize git
                subprocess.run(
                    ["git", "init"],
                    cwd=path,
                    check=True,
                    capture_output=True,
                )

                # Add all files
                subprocess.run(
                    ["git", "add", "."],
                    cwd=path,
                    check=True,
                    capture_output=True,
                )

                # Create initial commit
                subprocess.run(
                    ["git", "commit", "-m", "Initialize tsugite workspace"],
                    cwd=path,
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError:
                print("Warning: Failed to initialize git repository.")

        return cls(name=path.name, path=path)


__all__ = [
    "Workspace",
    "WorkspaceNotFoundError",
    "WORKSPACE_FILES",
    "DEFAULT_MEMORY_INJECT_DAYS",
    "DEFAULT_COMPACTION_THRESHOLD",
]
