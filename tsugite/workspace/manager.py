"""Workspace discovery and management."""

from pathlib import Path
from typing import List, Optional

from .models import Workspace, WorkspaceNotFoundError


class WorkspaceManager:
    """Manages workspace discovery and loading (convention-based, no config.yaml required)."""

    def __init__(self):
        """Initialize workspace manager with default search paths."""
        self.workspace_dirs = [
            Path.home() / ".tsugite" / "workspaces",
            Path.cwd() / ".tsugite" / "workspaces",
        ]

    def list_workspaces(self) -> List[str]:
        """List available workspaces.

        Returns:
            List of workspace names (any directory is a potential workspace)
        """
        workspaces = set()

        for workspace_dir in self.workspace_dirs:
            if not workspace_dir.exists():
                continue

            for path in workspace_dir.iterdir():
                if path.is_dir():
                    workspaces.add(path.name)

        return sorted(workspaces)

    def find_workspace_path(self, name: str) -> Optional[Path]:
        """Find workspace directory by name.

        Args:
            name: Workspace name

        Returns:
            Path to workspace directory, or None if not found
        """
        for workspace_dir in self.workspace_dirs:
            path = workspace_dir / name
            if path.exists() and path.is_dir():
                return path

        return None

    def load_workspace(self, name: str) -> Workspace:
        """Load workspace by name.

        Args:
            name: Workspace name

        Returns:
            Loaded workspace

        Raises:
            WorkspaceNotFoundError: If workspace not found
        """
        path = self.find_workspace_path(name)
        if not path:
            available = self.list_workspaces()
            raise WorkspaceNotFoundError(f"Workspace '{name}' not found. Available: {', '.join(available) or 'none'}")

        return Workspace.load(path)

    def create_workspace(
        self,
        name: str,
        path: Optional[Path] = None,
        soul_template: Optional[str] = None,
        user_name: Optional[str] = None,
        init_git: bool = False,
    ) -> Workspace:
        """Create new workspace.

        Args:
            name: Workspace name
            path: Optional custom path (defaults to ~/.tsugite/workspaces/{name})
            soul_template: Optional soul template name
            user_name: Optional user name for template rendering
            init_git: Whether to initialize git repository

        Returns:
            Created workspace

        Raises:
            ValueError: If workspace already exists
        """
        if path is None:
            path = Path.home() / ".tsugite" / "workspaces" / name

        if path.exists():
            raise ValueError(f"Workspace already exists: {path}")

        return Workspace.create(path, soul_template=soul_template, user_name=user_name, init_git=init_git)


__all__ = ["WorkspaceManager"]
