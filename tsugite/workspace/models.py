"""Convention-based workspace models."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

WORKSPACE_FILES = ["PERSONA.md", "SOUL.md", "USER.md", "MEMORY.md", "IDENTITY.md", "AGENTS.md"]
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
        """Auto-discover workspace identity files (PERSONA.md, SOUL.md, USER.md, etc.).

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

    def needs_onboarding(self) -> bool:
        """Check if workspace needs onboarding (unfilled identity).

        Returns:
            True if IDENTITY.md doesn't exist or has no name filled in
        """
        identity_path = self.path / "IDENTITY.md"
        if not identity_path.exists():
            return True

        content = identity_path.read_text()
        # Check if name field is empty - template has "- **Name:**" with nothing meaningful after
        for line in content.splitlines():
            if line.startswith("- **Name:**"):
                name_value = line.replace("- **Name:**", "").strip()
                return not name_value
        return True

    @staticmethod
    def get_workspace_dirs() -> List[Path]:
        """Get workspace search directories."""
        from tsugite.config import get_xdg_data_path

        return [
            get_xdg_data_path("workspaces"),
            Path.cwd() / ".tsugite" / "workspaces",
        ]

    @staticmethod
    def list_workspaces() -> List[str]:
        """List available workspaces."""
        workspaces = set()
        for workspace_dir in Workspace.get_workspace_dirs():
            if not workspace_dir.exists():
                continue
            for path in workspace_dir.iterdir():
                if path.is_dir():
                    workspaces.add(path.name)
        return sorted(workspaces)

    @staticmethod
    def find_workspace_path(name: str) -> Optional[Path]:
        """Find workspace directory by name."""
        for workspace_dir in Workspace.get_workspace_dirs():
            path = workspace_dir / name
            if path.exists() and path.is_dir():
                return path
        return None

    @classmethod
    def load_by_name(cls, name: str) -> "Workspace":
        """Load workspace by name.

        Args:
            name: Workspace name

        Returns:
            Loaded workspace

        Raises:
            WorkspaceNotFoundError: If workspace not found
        """
        path = cls.find_workspace_path(name)
        if not path:
            available = cls.list_workspaces()
            raise WorkspaceNotFoundError(f"Workspace '{name}' not found. Available: {', '.join(available) or 'none'}")
        return cls.load(path)

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
        persona_template: Optional[str] = None,
        user_name: Optional[str] = None,
        init_git: bool = False,
    ) -> "Workspace":
        """Create new workspace with optional persona template.

        Args:
            path: Path to create workspace at
            persona_template: Optional persona template name
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

        # Copy AGENTS.md template (operating instructions)
        templates_dir = Path(__file__).parent.parent / "templates"
        agents_template = templates_dir / "AGENTS.md"
        if agents_template.exists():
            import shutil

            shutil.copy(agents_template, path / "AGENTS.md")

        # Copy USER.md template (user profile)
        user_template = templates_dir / "USER.md"
        if user_template.exists():
            user_content = user_template.read_text()
            if user_name:
                user_content = user_content.replace("- **Name:**", f"- **Name:** {user_name}")
            (path / "USER.md").write_text(user_content)

        # Copy MEMORY.md template (long-term memory)
        memory_template = templates_dir / "MEMORY.md"
        if memory_template.exists():
            shutil.copy(memory_template, path / "MEMORY.md")

        # Copy IDENTITY.md template (agent identity)
        identity_template = templates_dir / "IDENTITY.md"
        if identity_template.exists():
            shutil.copy(identity_template, path / "IDENTITY.md")

        if persona_template:
            from .templates import load_persona_template

            persona_content = load_persona_template(persona_template, user_name=user_name)
            (path / "PERSONA.md").write_text(persona_content)

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
