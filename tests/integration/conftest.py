"""Shared fixtures for integration tests.

These tests hit a real LLM API and execute real tools.
They are excluded from default pytest runs and meant to be run manually.
"""

import os

import pytest

from tsugite.agent_runner.runner import run_agent
from tsugite.options import ExecutionOptions

INTEGRATION_MODEL = os.environ.get("TSUGITE_TEST_MODEL", "openai:gpt-4o-mini")


def pytest_configure(config):
    """Force serial execution -- integration tests use monkeypatch.chdir and real LLM calls."""
    if hasattr(config.option, "numprocesses"):
        config.option.numprocesses = 0
    if hasattr(config.option, "dist"):
        config.option.dist = "no"


def pytest_collection_modifyitems(config, items):
    """Skip all integration tests if no API key is available."""
    api_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]
    if not any(os.environ.get(k) for k in api_keys):
        skip = pytest.mark.skip(reason="No API key set")
        for item in items:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _register_all_tools(reset_tool_registry):
    """Re-register all tools after the parent conftest clears the registry.

    We must explicitly call tool(fn) because Python caches module imports,
    so _ensure_tools_loaded() won't re-execute @tool decorators after _tools is cleared.
    """
    from tsugite.tools import tool
    from tsugite.tools.agents import list_agents, list_available_agents, spawn_agent
    from tsugite.tools.fs import (
        create_directory,
        edit_file,
        file_exists,
        get_file_info,
        list_files,
        read_file,
        write_file,
    )
    from tsugite.tools.interactive import ask_user, ask_user_batch, return_value, send_message
    from tsugite.tools.shell import get_system_info, run

    for fn in [
        read_file,
        write_file,
        list_files,
        file_exists,
        create_directory,
        get_file_info,
        edit_file,
        run,
        get_system_info,
        return_value,
        send_message,
        ask_user,
        ask_user_batch,
        spawn_agent,
        list_agents,
        list_available_agents,
    ]:
        tool(fn)


@pytest.fixture(autouse=True)
def _isolate_data_dirs(tmp_path, monkeypatch):
    """Point XDG dirs at tmp_path so subprocess `tsu run` cannot read or
    write the user's real config, workspace, history, or secrets.

    Without this, `tsu run` falls back to the user's configured
    default_workspace (e.g. ~/.local/share/tsugite/workspaces/<name>) and
    silently pulls in their USER.md / MEMORY.md / AGENTS.md as attachments,
    and may write history/session state there. Tests must not touch real
    user data.
    """
    isolated = tmp_path / "xdg"
    isolated.mkdir(exist_ok=True)
    monkeypatch.setenv("XDG_DATA_HOME", str(isolated / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(isolated / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(isolated / "cache"))


@pytest.fixture(autouse=True)
def _chdir_to_workspace(work_dir, monkeypatch):
    monkeypatch.chdir(work_dir)


@pytest.fixture
def model():
    return INTEGRATION_MODEL


@pytest.fixture
def work_dir(tmp_path):
    """Isolated working directory for agent file operations."""
    d = tmp_path / "workspace"
    d.mkdir()
    return d


def run_integration_agent(agent_path, prompt, **kwargs):
    """Run an agent with integration-test defaults."""
    return run_agent(
        agent_path=agent_path,
        prompt=prompt,
        exec_options=ExecutionOptions(return_token_usage=False),
        **kwargs,
    )


@pytest.fixture
def agent_file(tmp_path):
    """Factory to create agent .md files with the configured model."""

    def _create(name="test", tools=None, max_turns=5, extra_frontmatter="", body=None):
        tools = tools or ["read_file", "write_file", "list_files", "run"]
        body = body or (
            "You are working in directory: {{ cwd() }}\n\n"
            "Task: {{ user_prompt }}\n\n"
            "Complete the task using available tools. Call return_value() with your result."
        )
        content = f"""---
name: {name}
extends: none
model: {INTEGRATION_MODEL}
max_turns: {max_turns}
tools: {tools}
{extra_frontmatter}---

{body}
"""
        path = tmp_path / f"{name}.md"
        path.write_text(content)
        return path

    return _create
