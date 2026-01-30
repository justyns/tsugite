"""Tests for workspace CWD behavior."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from tsugite.cli.helpers import PathContext, workspace_directory_context


class TestPathContext:
    """Tests for PathContext dataclass."""

    def test_path_context_creation(self):
        """Test PathContext can be created with all fields."""
        ctx = PathContext(
            invoked_from=Path("/home/user/projects"),
            workspace_dir=Path("/home/user/.tsugite/workspaces/test"),
            effective_cwd=Path("/home/user/.tsugite/workspaces/test"),
        )
        assert ctx.invoked_from == Path("/home/user/projects")
        assert ctx.workspace_dir == Path("/home/user/.tsugite/workspaces/test")
        assert ctx.effective_cwd == Path("/home/user/.tsugite/workspaces/test")

    def test_path_context_none_workspace(self):
        """Test PathContext with None workspace_dir."""
        ctx = PathContext(
            invoked_from=Path("/home/user/projects"),
            workspace_dir=None,
            effective_cwd=Path("/home/user/projects"),
        )
        assert ctx.invoked_from == Path("/home/user/projects")
        assert ctx.workspace_dir is None
        assert ctx.effective_cwd == Path("/home/user/projects")


class TestWorkspaceDirectoryContext:
    """Tests for workspace_directory_context context manager."""

    @pytest.fixture
    def console(self):
        """Provide a Rich console for tests."""
        return Console()

    @pytest.fixture
    def mock_workspace(self, tmp_path):
        """Create a mock workspace object."""
        workspace_path = tmp_path / "workspace"
        workspace_path.mkdir()
        workspace = MagicMock()
        workspace.path = workspace_path
        return workspace

    def test_no_workspace_no_root(self, console):
        """Test behavior with no workspace and no root - CWD unchanged."""
        original_cwd = Path.cwd()

        with workspace_directory_context(None, None, console) as ctx:
            assert ctx.invoked_from == original_cwd
            assert ctx.workspace_dir is None
            assert ctx.effective_cwd == original_cwd
            assert Path.cwd() == original_cwd

    def test_workspace_only_changes_to_workspace(self, console, mock_workspace):
        """Test that --workspace alone sets CWD to workspace.path."""
        original_cwd = Path.cwd()

        with workspace_directory_context(mock_workspace, None, console) as ctx:
            assert ctx.invoked_from == original_cwd
            assert ctx.workspace_dir == mock_workspace.path
            assert ctx.effective_cwd == mock_workspace.path
            assert Path.cwd() == mock_workspace.path

        # Verify CWD restored
        assert Path.cwd() == original_cwd

    def test_root_only_changes_to_root(self, console, tmp_path):
        """Test that --root alone sets CWD to root path."""
        original_cwd = Path.cwd()
        root_path = tmp_path / "root"
        root_path.mkdir()

        with workspace_directory_context(None, str(root_path), console) as ctx:
            assert ctx.invoked_from == original_cwd
            assert ctx.workspace_dir is None
            assert ctx.effective_cwd == root_path
            assert Path.cwd() == root_path

        # Verify CWD restored
        assert Path.cwd() == original_cwd

    def test_workspace_and_root_uses_root(self, console, mock_workspace, tmp_path):
        """Test that --workspace + --root uses root for CWD but tracks workspace."""
        original_cwd = Path.cwd()
        root_path = tmp_path / "root"
        root_path.mkdir()

        with workspace_directory_context(mock_workspace, str(root_path), console) as ctx:
            assert ctx.invoked_from == original_cwd
            assert ctx.workspace_dir == mock_workspace.path
            assert ctx.effective_cwd == root_path
            assert Path.cwd() == root_path

        # Verify CWD restored
        assert Path.cwd() == original_cwd

    def test_nonexistent_workspace_raises_exit(self, console):
        """Test that nonexistent workspace path raises typer.Exit."""
        import typer

        workspace = MagicMock()
        workspace.path = Path("/nonexistent/workspace/path")

        with pytest.raises(typer.Exit):
            with workspace_directory_context(workspace, None, console):
                pass

    def test_nonexistent_root_raises_exit(self, console):
        """Test that nonexistent root path raises typer.Exit."""
        import typer

        with pytest.raises(typer.Exit):
            with workspace_directory_context(None, "/nonexistent/root/path", console):
                pass


class TestContextVariablesInTemplates:
    """Tests for path context variables in template rendering."""

    def test_cwd_helper_in_renderer(self):
        """Test that cwd() helper function works in renderer."""
        from tsugite.renderer import AgentRenderer, cwd

        # Test the helper function directly
        assert cwd() == str(Path.cwd())

        # Test via renderer globals
        renderer = AgentRenderer()
        result = renderer.render("CWD: {{ cwd() }}", {})
        assert result == f"CWD: {Path.cwd()}"

    def test_context_variables_available_in_prepare(self):
        """Test that CWD, WORKSPACE_DIR, INVOKED_FROM are in template context."""
        from unittest.mock import MagicMock

        from tsugite.agent_preparation import AgentPreparer
        from tsugite.cli.helpers import PathContext

        # Create mock agent
        agent = MagicMock()
        agent.config.prefetch = None
        agent.config.text_mode = False
        agent.config.tools = []
        agent.config.instructions = ""
        agent.config.auto_load_skills = []
        agent.content = "Test content"

        # Create path context
        path_context = PathContext(
            invoked_from=Path("/home/user/projects"),
            workspace_dir=Path("/home/user/workspace"),
            effective_cwd=Path("/home/user/workspace"),
        )

        preparer = AgentPreparer()

        # Mock the steps that require external resources
        with (
            patch("tsugite.utils.is_interactive", return_value=False),
            patch("tsugite.agent_runner.runner.get_default_instructions", return_value=""),
            patch("tsugite.agent_runner.runner._combine_instructions", return_value=""),
            patch("tsugite.tools.expand_tool_specs", return_value=[]),
            patch("tsugite.core.tools.create_tool_from_tsugite"),
            patch("tsugite.core.agent.build_system_prompt", return_value="System prompt"),
        ):
            prepared = preparer.prepare(
                agent=agent,
                prompt="Test prompt",
                path_context=path_context,
            )

        # Verify context variables
        assert prepared.context["CWD"] == str(Path.cwd())
        assert prepared.context["INVOKED_FROM"] == "/home/user/projects"
        assert prepared.context["WORKSPACE_DIR"] == "/home/user/workspace"

    def test_context_variables_none_without_path_context(self):
        """Test that WORKSPACE_DIR and INVOKED_FROM are None without path_context."""
        from unittest.mock import MagicMock

        from tsugite.agent_preparation import AgentPreparer

        # Create mock agent
        agent = MagicMock()
        agent.config.prefetch = None
        agent.config.text_mode = False
        agent.config.tools = []
        agent.config.instructions = ""
        agent.config.auto_load_skills = []
        agent.content = "Test content"

        preparer = AgentPreparer()

        with (
            patch("tsugite.utils.is_interactive", return_value=False),
            patch("tsugite.agent_runner.runner.get_default_instructions", return_value=""),
            patch("tsugite.agent_runner.runner._combine_instructions", return_value=""),
            patch("tsugite.tools.expand_tool_specs", return_value=[]),
            patch("tsugite.core.tools.create_tool_from_tsugite"),
            patch("tsugite.core.agent.build_system_prompt", return_value="System prompt"),
        ):
            prepared = preparer.prepare(
                agent=agent,
                prompt="Test prompt",
                path_context=None,
            )

        assert prepared.context["INVOKED_FROM"] is None
        assert prepared.context["WORKSPACE_DIR"] is None


class TestEnvironmentBlockInSystemPrompt:
    """Tests for environment block added to system prompt."""

    def test_environment_block_added_when_cwd_differs(self):
        """Test that environment block is added when INVOKED_FROM != CWD."""
        from unittest.mock import MagicMock

        from tsugite.agent_preparation import AgentPreparer
        from tsugite.cli.helpers import PathContext

        # Create mock agent
        agent = MagicMock()
        agent.config.prefetch = None
        agent.config.text_mode = False
        agent.config.tools = []
        agent.config.instructions = ""
        agent.config.auto_load_skills = []
        agent.content = "Test content"

        # Create path context where invoked_from differs from CWD
        path_context = PathContext(
            invoked_from=Path("/home/user/projects"),
            workspace_dir=Path("/home/user/workspace"),
            effective_cwd=Path("/home/user/workspace"),
        )

        preparer = AgentPreparer()

        base_system_prompt = "Base system prompt"

        with (
            patch("tsugite.utils.is_interactive", return_value=False),
            patch("tsugite.agent_runner.runner.get_default_instructions", return_value=""),
            patch("tsugite.agent_runner.runner._combine_instructions", return_value=""),
            patch("tsugite.tools.expand_tool_specs", return_value=[]),
            patch("tsugite.core.tools.create_tool_from_tsugite"),
            patch("tsugite.core.agent.build_system_prompt", return_value=base_system_prompt),
        ):
            prepared = preparer.prepare(
                agent=agent,
                prompt="Test prompt",
                path_context=path_context,
            )

        # Verify environment block is in system message
        assert "## Environment" in prepared.system_message
        assert "Invoked from:" in prepared.system_message
        assert "/home/user/projects" in prepared.system_message

    def test_no_environment_block_when_same_directory(self):
        """Test that no environment block when invoked_from == CWD."""
        from unittest.mock import MagicMock

        from tsugite.agent_preparation import AgentPreparer
        from tsugite.cli.helpers import PathContext

        cwd = Path.cwd()

        # Create mock agent
        agent = MagicMock()
        agent.config.prefetch = None
        agent.config.text_mode = False
        agent.config.tools = []
        agent.config.instructions = ""
        agent.config.auto_load_skills = []
        agent.content = "Test content"

        # Create path context where invoked_from equals CWD
        path_context = PathContext(
            invoked_from=cwd,
            workspace_dir=None,
            effective_cwd=cwd,
        )

        preparer = AgentPreparer()

        base_system_prompt = "Base system prompt"

        with (
            patch("tsugite.utils.is_interactive", return_value=False),
            patch("tsugite.agent_runner.runner.get_default_instructions", return_value=""),
            patch("tsugite.agent_runner.runner._combine_instructions", return_value=""),
            patch("tsugite.tools.expand_tool_specs", return_value=[]),
            patch("tsugite.core.tools.create_tool_from_tsugite"),
            patch("tsugite.core.agent.build_system_prompt", return_value=base_system_prompt),
        ):
            prepared = preparer.prepare(
                agent=agent,
                prompt="Test prompt",
                path_context=path_context,
            )

        # Verify environment block is NOT in system message
        assert "## Environment" not in prepared.system_message


class TestExecutorPathVariables:
    """Tests for path variables injected into executor namespace."""

    @pytest.mark.asyncio
    async def test_executor_injects_path_variables(self):
        """Test that LocalExecutor injects WORKSPACE_DIR and INVOKED_FROM."""
        from tsugite.cli.helpers import PathContext
        from tsugite.core.executor import LocalExecutor

        path_context = PathContext(
            invoked_from=Path("/home/user/projects"),
            workspace_dir=Path("/home/user/workspace"),
            effective_cwd=Path("/home/user/workspace"),
        )

        executor = LocalExecutor(path_context=path_context)

        assert executor.namespace["WORKSPACE_DIR"] == "/home/user/workspace"
        assert executor.namespace["INVOKED_FROM"] == "/home/user/projects"

    @pytest.mark.asyncio
    async def test_executor_path_variables_none_without_context(self):
        """Test that path variables are None without path_context."""
        from tsugite.core.executor import LocalExecutor

        executor = LocalExecutor(path_context=None)

        assert executor.namespace["WORKSPACE_DIR"] is None
        assert executor.namespace["INVOKED_FROM"] is None

    @pytest.mark.asyncio
    async def test_executor_path_variables_accessible_in_code(self):
        """Test that path variables can be accessed in executed code."""
        from tsugite.cli.helpers import PathContext
        from tsugite.core.executor import LocalExecutor

        path_context = PathContext(
            invoked_from=Path("/home/user/projects"),
            workspace_dir=Path("/home/user/workspace"),
            effective_cwd=Path("/home/user/workspace"),
        )

        executor = LocalExecutor(path_context=path_context)

        # Execute code that accesses the variables
        result = await executor.execute("print(f'Workspace: {WORKSPACE_DIR}')")
        assert result.error is None
        assert "Workspace: /home/user/workspace" in result.stdout

        result = await executor.execute("print(f'Invoked from: {INVOKED_FROM}')")
        assert result.error is None
        assert "Invoked from: /home/user/projects" in result.stdout
