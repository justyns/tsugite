"""Tests for interactive mode context flag."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from tsugite.agent_runner import run_agent_async, run_multistep_agent_async
from tsugite.md_agents import AgentConfig


@pytest.fixture
def mock_agent_runner():
    """Create a mock TsugiteAgent that captures the rendered prompt."""
    captured_prompts = []

    def create_mock(return_value="Test complete"):
        async def mock_run(prompt, return_full_result=False, stream=False):
            captured_prompts.append(prompt)
            return return_value

        mock_instance = MagicMock()
        mock_instance.run = MagicMock(side_effect=mock_run)
        return mock_instance, captured_prompts

    return create_mock


@pytest.mark.asyncio
async def test_is_interactive_flag_true_in_tty(temp_dir, monkeypatch, mock_agent_runner):
    """Test that is_interactive flag is True when running in a TTY."""
    # Mock TTY check to return True
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    # Create a simple test agent that checks is_interactive
    agent_content = """---
name: test_interactive_flag
extends: none
model: openai:gpt-4o-mini
max_turns: 1
tools: []
---

# Test Interactive Flag

Interactive mode: {{ is_interactive }}

Task: {{ user_prompt }}
"""

    agent_file = temp_dir / "test_agent.md"
    agent_file.write_text(agent_content)

    # Use fixture to create mock
    mock_instance, captured_prompts = mock_agent_runner()

    with patch("tsugite.agent_runner.runner.TsugiteAgent", return_value=mock_instance):
        await run_agent_async(agent_file, "test prompt")

    # Verify is_interactive was True in the rendered prompt
    assert len(captured_prompts) > 0
    assert "Interactive mode: True" in captured_prompts[0]


@pytest.mark.asyncio
async def test_is_interactive_flag_false_in_non_tty(temp_dir, monkeypatch, mock_agent_runner):
    """Test that is_interactive flag is False when not running in a TTY."""
    # Mock TTY check to return False
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    # Create a simple test agent that checks is_interactive
    agent_content = """---
name: test_interactive_flag
extends: none
model: openai:gpt-4o-mini
max_turns: 1
tools: []
---

# Test Interactive Flag

Interactive mode: {{ is_interactive }}

Task: {{ user_prompt }}
"""

    agent_file = temp_dir / "test_agent.md"
    agent_file.write_text(agent_content)

    # Use fixture to create mock
    mock_instance, captured_prompts = mock_agent_runner()

    with patch("tsugite.agent_runner.runner.TsugiteAgent", return_value=mock_instance):
        await run_agent_async(agent_file, "test prompt")

    # Verify is_interactive was False in the rendered prompt
    assert len(captured_prompts) > 0
    assert "Interactive mode: False" in captured_prompts[0]


@pytest.mark.asyncio
async def test_multistep_agent_receives_interactive_flag(temp_dir, monkeypatch, mock_agent_runner):
    """Test that multi-step agents receive the is_interactive flag."""
    # Mock TTY check
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    # Create a multi-step test agent
    agent_content = """---
name: test_multistep_interactive
extends: none
model: openai:gpt-4o-mini
max_turns: 1
tools: []
---

# Preamble

This is the preamble content.

<!-- tsu:step name="check_interactive" assign="result" -->

Check interactive mode: {{ is_interactive }}

Task: {{ user_prompt }}
"""

    agent_file = temp_dir / "test_multistep.md"
    agent_file.write_text(agent_content)

    # Use fixture to create mock
    mock_instance, captured_prompts = mock_agent_runner("Step complete")

    with patch("tsugite.agent_runner.runner.TsugiteAgent", return_value=mock_instance):
        await run_multistep_agent_async(agent_file, "test prompt")

    # Verify at least one prompt was captured
    assert len(captured_prompts) > 0

    # Verify is_interactive was True in the rendered prompt
    assert any("Check interactive mode: True" in p for p in captured_prompts)


@pytest.mark.asyncio
async def test_interactive_flag_available_in_templates(temp_dir, monkeypatch, mock_agent_runner):
    """Test that is_interactive can be used in template conditionals."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    agent_content = """---
name: test_conditional
extends: none
model: openai:gpt-4o-mini
max_turns: 1
tools: []
---

# Conditional Interactive Check

{% if is_interactive %}
You can use ask_user tool.
{% else %}
Running in headless mode, use defaults.
{% endif %}

Task: {{ user_prompt }}
"""

    agent_file = temp_dir / "test_conditional.md"
    agent_file.write_text(agent_content)

    mock_instance, captured_prompts = mock_agent_runner()

    with patch("tsugite.agent_runner.runner.TsugiteAgent", return_value=mock_instance):
        await run_agent_async(agent_file, "test prompt")

    assert len(captured_prompts) > 0
    assert "You can use ask_user tool." in captured_prompts[0]
    assert "Running in headless mode" not in captured_prompts[0]


# --- Helpers for tool injection tests ---


def _make_agent_config(tools=None):
    return AgentConfig(
        name="test_agent",
        description="Test agent",
        model="openai:gpt-4o-mini",
        max_turns=1,
        tools=tools or ["write_file"],
        prefetch=[],
        instructions="",
        mcp_servers={},
        extends=None,
    )


def _prepare_and_capture_tools(agent_config, context=None):
    """Run AgentPreparer.prepare() and return the list of tool names created."""
    captured_tools = []

    def mock_create_tool(tool_name):
        captured_tools.append(tool_name)
        from tsugite.core.tools import Tool

        return Tool(name=tool_name, description=f"Mock {tool_name}", function=lambda: None, parameters={})

    with patch("tsugite.core.tools.create_tool_from_tsugite", side_effect=mock_create_tool):
        from pathlib import Path

        from tsugite.agent_preparation import AgentPreparer
        from tsugite.md_agents import Agent

        agent = Agent(content="Test", config=agent_config, file_path=Path("<test>"))
        preparer = AgentPreparer()
        preparer.prepare(agent=agent, prompt="Test prompt", context=context or {})

    return captured_tools


# --- Tool injection tests ---


def test_ask_user_tool_not_available_in_headless(monkeypatch, file_tools, interactive_tools):
    """Test that ask_user tool is filtered out in non-interactive mode."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    config = _make_agent_config(tools=["ask_user", "write_file"])
    tools = _prepare_and_capture_tools(config)

    assert "ask_user" not in tools
    assert "write_file" in tools


def test_ask_user_tool_available_in_interactive(monkeypatch, file_tools, interactive_tools):
    """Test that ask_user tool is available in interactive mode."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    config = _make_agent_config(tools=["ask_user", "write_file"])
    tools = _prepare_and_capture_tools(config)

    assert "ask_user" in tools
    assert "write_file" in tools


class TestDaemonModeInteractiveTools:
    """Test that ask_user tools are auto-injected in daemon mode."""

    def test_daemon_mode_injects_interactive_tools(self, monkeypatch, file_tools, interactive_tools):
        """In daemon mode (is_daemon=True), ask_user tools should be auto-injected."""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

        config = _make_agent_config(tools=["write_file"])
        tools = _prepare_and_capture_tools(config, context={"is_daemon": True})

        assert "ask_user" in tools
        assert "ask_user_batch" in tools
        assert "write_file" in tools

    def test_daemon_mode_no_duplicate_if_already_listed(self, monkeypatch, file_tools, interactive_tools):
        """If agent already lists ask_user, daemon mode shouldn't duplicate it."""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

        config = _make_agent_config(tools=["ask_user", "write_file"])
        tools = _prepare_and_capture_tools(config, context={"is_daemon": True})

        assert tools.count("ask_user") == 1

    def test_headless_no_daemon_removes_ask_user(self, monkeypatch, file_tools, interactive_tools):
        """Without daemon mode and without TTY, ask_user should be removed."""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

        config = _make_agent_config(tools=["ask_user", "write_file"])
        tools = _prepare_and_capture_tools(config, context={})

        assert "ask_user" not in tools
        assert "write_file" in tools

    def test_headless_no_daemon_doesnt_inject(self, monkeypatch, file_tools, interactive_tools):
        """Without daemon mode and without TTY, ask_user should NOT be auto-injected."""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

        config = _make_agent_config(tools=["write_file"])
        tools = _prepare_and_capture_tools(config, context={})

        assert "ask_user" not in tools
        assert "ask_user_batch" not in tools

    def test_interactive_tty_injects_ask_user(self, monkeypatch, file_tools, interactive_tools):
        """With TTY available, ask_user should be auto-injected even if not listed."""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

        config = _make_agent_config(tools=["write_file"])
        tools = _prepare_and_capture_tools(config, context={})

        assert "ask_user" in tools
        assert "ask_user_batch" in tools

    def test_interaction_backend_injects_ask_user(self, monkeypatch, file_tools, interactive_tools):
        """When an interaction backend is set, ask_user should be auto-injected."""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

        from tsugite.interaction import NonInteractiveBackend, set_interaction_backend

        set_interaction_backend(NonInteractiveBackend())
        try:
            config = _make_agent_config(tools=["write_file"])
            tools = _prepare_and_capture_tools(config, context={})

            assert "ask_user" in tools
            assert "ask_user_batch" in tools
        finally:
            set_interaction_backend(None)
