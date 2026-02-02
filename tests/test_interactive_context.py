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
    # Mock TTY check
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    # Create an agent that uses conditional logic
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

    # Use fixture to create mock
    mock_instance, captured_prompts = mock_agent_runner()

    with patch("tsugite.agent_runner.runner.TsugiteAgent", return_value=mock_instance):
        await run_agent_async(agent_file, "test prompt")

    # Verify the conditional worked
    assert len(captured_prompts) > 0
    assert "You can use ask_user tool." in captured_prompts[0]
    assert "Running in headless mode" not in captured_prompts[0]


def test_ask_user_tool_not_available_in_headless(monkeypatch, file_tools, interactive_tools):
    """Test that ask_user tool is filtered out in non-interactive mode."""
    # Mock TTY check to return False (headless)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    # Create an agent config that includes ask_user
    agent_config = AgentConfig(
        name="test_headless",
        description="Test agent",
        model="openai:gpt-4o-mini",
        max_turns=1,
        tools=["ask_user", "write_file"],  # ask_user should be filtered out
        prefetch=[],
        permissions_profile="default",
        context_budget={"tokens": 8000, "priority": ["system", "task"]},
        instructions="",
        mcp_servers={},
        extends=None,
    )

    # Capture the tools list
    captured_tools = []

    def mock_create_tool(tool_name):
        captured_tools.append(tool_name)
        from tsugite.core.tools import Tool

        return Tool(name=tool_name, description=f"Mock {tool_name}", function=lambda: None, parameters={})

    with patch("tsugite.core.tools.create_tool_from_tsugite", side_effect=mock_create_tool):
        from pathlib import Path

        from tsugite.agent_preparation import AgentPreparer
        from tsugite.md_agents import Agent

        # Create agent and use AgentPreparer to trigger tool creation
        agent = Agent(content="Test", config=agent_config, file_path=Path("<test>"))
        preparer = AgentPreparer()
        preparer.prepare(agent=agent, prompt="Test prompt", context={})

    # Verify ask_user was filtered out
    assert len(captured_tools) > 0
    assert "ask_user" not in captured_tools
    assert "write_file" in captured_tools  # Other tools should still be there


def test_ask_user_tool_available_in_interactive(monkeypatch, file_tools, interactive_tools):
    """Test that ask_user tool is available in interactive mode."""
    # Mock TTY check to return True (interactive)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    # Create an agent config that includes ask_user
    agent_config = AgentConfig(
        name="test_interactive",
        description="Test agent",
        model="openai:gpt-4o-mini",
        max_turns=1,
        tools=["ask_user", "write_file"],
        prefetch=[],
        permissions_profile="default",
        context_budget={"tokens": 8000, "priority": ["system", "task"]},
        instructions="",
        mcp_servers={},
        extends=None,
    )

    # Capture the tools list
    captured_tools = []

    def mock_create_tool(tool_name):
        captured_tools.append(tool_name)
        from tsugite.core.tools import Tool

        return Tool(name=tool_name, description=f"Mock {tool_name}", function=lambda: None, parameters={})

    with patch("tsugite.core.tools.create_tool_from_tsugite", side_effect=mock_create_tool):
        from pathlib import Path

        from tsugite.agent_preparation import AgentPreparer
        from tsugite.md_agents import Agent

        # Create agent and use AgentPreparer to trigger tool creation
        agent = Agent(content="Test", config=agent_config, file_path=Path("<test>"))
        preparer = AgentPreparer()
        preparer.prepare(agent=agent, prompt="Test prompt", context={})

    # Verify ask_user is still present in interactive mode
    assert len(captured_tools) > 0
    assert "ask_user" in captured_tools
    assert "write_file" in captured_tools
