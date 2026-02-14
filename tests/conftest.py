"""Test configuration and fixtures."""

import asyncio
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Generator

import pytest

# Suppress RuntimeWarnings from litellm's async cleanup
# These warnings occur during test teardown and are not actionable in our tests
warnings.filterwarnings("ignore", category=RuntimeWarning, module="litellm")
warnings.filterwarnings("ignore", message="coroutine.*was never awaited")

# Configure Python warnings module to suppress uncaught RuntimeWarnings at module level
if not sys.warnoptions:
    warnings.simplefilter("ignore", RuntimeWarning)


@pytest.fixture(autouse=True)
def event_loop_policy():
    """Ensure clean event loop for each test.

    This fixture prevents 'Event loop is closed' errors that can occur when
    run_multistep_agent() (and other sync wrappers) use asyncio.run() internally.
    asyncio.run() creates and closes its own event loop, which can interfere with
    pytest-asyncio's event loop management in strict mode.
    """
    # Get current policy or create new one
    policy = asyncio.get_event_loop_policy()

    yield

    # Clean up: close any lingering loops and reset policy
    try:
        # Suppress deprecation warning for get_event_loop() in cleanup context
        # This is safe here because we're only closing existing loops, not creating new ones
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            loop = policy.get_event_loop()
        if loop and not loop.is_closed():
            loop.close()
    except RuntimeError:
        # No current event loop, which is fine
        pass

    # Reset to ensure fresh state for next test
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())


@pytest.fixture(autouse=True)
def clear_agent_context():
    """Clear agent context state between tests.

    This prevents test pollution where one test sets current_agent
    and affects another test running in the same worker.
    """
    from tsugite.agent_runner.helpers import clear_allowed_agents, clear_current_agent

    # Clear before test
    clear_current_agent()
    clear_allowed_agents()

    yield

    # Clear after test
    clear_current_agent()
    clear_allowed_agents()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_agent_content() -> str:
    """Sample agent markdown content for testing."""
    return """---
name: test_agent
extends: none
model: openai:gpt-4o-mini
max_turns: 5
tools: [read_file, write_file]
prefetch:
  - tool: search_memory
    args: { query: "test" }
    assign: memories
permissions_profile: test_safe
instructions: |
    Provide concise, evidence-backed responses and note any assumptions.
context_budget:
  tokens: 8000
  priority: [system, task]
---

# System
You are a test agent.

# Context
- Time: {{ now() }}
- Memories: {{ memories }}

# Task
{{ user_prompt }}

## Test Section
<!-- tsu:tool name=read_file args={"path": "test.txt"} assign=content -->
<!-- tsu:await output=result -->
"""


@pytest.fixture
def simple_agent_content() -> str:
    """Simple agent content without complex features."""
    return """---
name: simple_test_agent
extends: none
model: ollama:qwen2.5-coder:7b
max_turns: 3
tools: []
---

# Simple Test Agent

You are a simple test agent.

Task: {{ user_prompt }}
"""


@pytest.fixture
def spawn_agent_content() -> str:
    """Agent content that uses spawn_agent tool."""
    return """---
name: parent_agent
extends: none
model: ollama:qwen2.5-coder:7b
max_turns: 5
tools: [spawn_agent]
---

# Parent Agent

You are a parent agent that coordinates sub-agents.

Task: {{ user_prompt }}

Use the spawn_agent tool to delegate work to sub-agents.
"""


@pytest.fixture
def evaluator_agent_content() -> str:
    """Agent content for LLM evaluation tests."""
    return """---
name: evaluator_agent
extends: none
model: openai:gpt-4o-mini
max_turns: 3
tools: []
---

# LLM Evaluator Agent

You are an expert evaluator tasked with assessing AI agent performance.

{{ user_prompt }}
"""


@pytest.fixture
def sample_agent_file(temp_dir: Path, sample_agent_content: str) -> Path:
    """Create a sample agent file for testing."""
    agent_file = temp_dir / "test_agent.md"
    agent_file.write_text(sample_agent_content)
    return agent_file


def create_agent_file(temp_dir: Path, content: str, filename: str = "agent.md") -> Path:
    """Helper function to create agent files in tests."""
    agent_file = temp_dir / filename
    agent_file.write_text(content)
    return agent_file


@pytest.fixture
def mock_llm_evaluation_response() -> str:
    """Standard mock response for LLM evaluation tests."""
    return """{
    "score": 8.5,
    "feedback": "The output demonstrates good understanding of the task.",
    "reasoning": "Clear structure, accurate content, and appropriate tone.",
    "criteria_breakdown": {
        "accuracy": 9,
        "clarity": 8,
        "completeness": 8
    },
    "assessment": "Good quality output"
}"""


@pytest.fixture(autouse=True)
def reset_tool_registry():
    """Reset the tool registry before each test."""
    from tsugite.tools import _tools

    original_tools = _tools.copy()
    _tools.clear()
    yield
    _tools.clear()
    _tools.update(original_tools)


@pytest.fixture(autouse=True)
def isolate_config_files(tmp_path, monkeypatch):
    """Isolate config files for each test to prevent cross-test contamination.

    Uses XDG_CONFIG_HOME instead of HOME because Path.home() doesn't respect
    the HOME environment variable on Unix systems (it reads from password database).
    """
    test_config = tmp_path / "config"
    test_config.mkdir(exist_ok=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(test_config))

    yield


@pytest.fixture
def file_tools(reset_tool_registry):
    """Register file system tools for testing."""
    from tsugite.tools import tool
    from tsugite.tools.fs import (
        create_directory,
        edit_file,
        file_exists,
        get_file_info,
        list_files,
        read_file,
        write_file,
    )

    # Re-register the tools after registry reset
    tool(write_file)
    tool(read_file)
    tool(list_files)
    tool(file_exists)
    tool(create_directory)
    tool(get_file_info)
    tool(edit_file)


@pytest.fixture(autouse=True)
def reset_skill_manager():
    """Reset the skill manager before each test."""
    from tsugite.tools.skills import SkillManager, set_skill_manager

    original_manager = None
    try:
        from tsugite.tools.skills import _default_skill_manager

        original_manager = _default_skill_manager
    except (ImportError, AttributeError):
        pass

    # Create a fresh manager for each test
    set_skill_manager(SkillManager())
    yield
    # Reset to original or None
    set_skill_manager(original_manager)


@pytest.fixture(autouse=True)
def spawn_agent_tool(reset_tool_registry, request):
    """Register spawn_agent tool for testing.

    Autouse fixture because spawn_agent is automatically added to all agents
    in agent_preparation.py.

    Skips registration for test_tool_registry.py tests that need an empty registry.
    """
    # Skip for tool registry tests that need an empty registry
    if "test_tool_registry" in request.node.nodeid:
        return

    from tsugite.tools import tool
    from tsugite.tools.agents import spawn_agent

    # Re-register the tools after registry reset
    tool(spawn_agent)


@pytest.fixture
def interactive_tools(reset_tool_registry):
    """Register interactive tools for testing."""
    from tsugite.tools import tool
    from tsugite.tools.interactive import ask_user, ask_user_batch, final_answer, send_message

    # Re-register the tools after registry reset
    tool(ask_user)
    tool(ask_user_batch)
    tool(final_answer)
    tool(send_message)


@pytest.fixture
def agents_tools(reset_tool_registry):
    """Register agent delegation tools for testing."""
    from tsugite.tools import tool
    from tsugite.tools.agents import list_agents, spawn_agent

    # Re-register the tools after registry reset
    tool(list_agents)
    tool(spawn_agent)


@pytest.fixture
def http_tools(reset_tool_registry):
    """Register HTTP tools for testing."""
    from tsugite.tools import tool
    from tsugite.tools.http import fetch_text, web_search

    # Re-register the tools after registry reset
    tool(fetch_text)
    tool(web_search)


@pytest.fixture
def shell_tools(reset_tool_registry):
    """Register shell execution tools for testing."""
    from tsugite.tools import tool
    from tsugite.tools.shell import run

    # Re-register the tools after registry reset
    tool(run)


@pytest.fixture
def skill_tools(reset_tool_registry):
    """Register skill management tools for testing."""
    from tsugite.tools import tool
    from tsugite.tools.skills import (
        get_skills_for_template,
        list_available_skills,
        load_skill,
    )

    # Re-register the tools after registry reset
    tool(load_skill)
    tool(list_available_skills)
    tool(get_skills_for_template)


@pytest.fixture
def cli_runner():
    """Create a CLI test runner."""
    from typer.testing import CliRunner

    return CliRunner()


@pytest.fixture
def mock_animation_context():
    """Mock animation context manager for CLI tests."""
    from unittest.mock import MagicMock

    mock_context = MagicMock()
    mock_context.__enter__ = MagicMock()
    mock_context.__exit__ = MagicMock(return_value=None)
    return mock_context


@pytest.fixture
def temp_config_file(temp_dir):
    """Create a temporary config file path."""
    return temp_dir / "config.json"


@pytest.fixture
def agent_content_factory():
    """Factory for creating agent content with various configurations."""

    def _create_agent(
        name="test_agent",
        model="openai:gpt-4o-mini",
        tools=None,
        prefetch=None,
        instructions=None,
        content="# Task\n{{ user_prompt }}",
        extends="none",
    ):
        tools_str = str(tools) if tools is not None else "[]"
        frontmatter = f"""---
name: {name}
extends: {extends}
model: {model}
tools: {tools_str}"""

        if prefetch:
            frontmatter += f"\nprefetch: {prefetch}"
        if instructions:
            frontmatter += f"\ninstructions: {instructions}"

        frontmatter += "\n---\n\n"
        return frontmatter + content

    return _create_agent


@pytest.fixture
def mcp_config_factory():
    """Factory for creating MCP server configurations."""

    def _create_config(server_type="stdio", name="test-server", **kwargs):
        """Create an MCP server config dict.

        Args:
            server_type: "stdio" or "http"
            name: Server name
            **kwargs: Additional config fields (command, args, env, url, etc)
        """
        if server_type == "stdio":
            return {
                "command": kwargs.get("command", "npx"),
                "args": kwargs.get("args", ["-y", "test-server"]),
                "env": kwargs.get("env", {}),
            }
        elif server_type == "http":
            return {"url": kwargs.get("url", "http://localhost:8000/mcp"), "type": "http"}
        else:
            raise ValueError(f"Unknown server type: {server_type}")

    return _create_config


@pytest.fixture
def mock_litellm_response():
    """Create a mock LiteLLM response for agent tests.

    This fixture provides a factory function that creates mock LLM responses
    with customizable content and reasoning. Consolidates duplicate fixtures
    from test_agent.py and test_agent_ui_events.py.

    Returns:
        Factory function that creates mock response objects
    """
    from unittest.mock import MagicMock

    def _create_response(content: str, reasoning_content: str = None):
        """Factory to create mock responses with different content."""
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock(spec=[])
        response.choices[0].message.content = content

        # Add reasoning content if provided
        if reasoning_content:
            response.choices[0].message.reasoning_content = reasoning_content

        # Add token usage with spec to prevent auto-creation of attributes
        # This prevents MagicMock from auto-creating completion_tokens_details
        # which would trigger REASONING_TOKENS events in tests
        response.usage = MagicMock(spec=["total_tokens"])
        response.usage.total_tokens = 100

        return response

    return _create_response


def mock_agent_execution_result(
    response: str = "Test response",
    token_count: int = 100,
    cost: float = 0.005,
    step_count: int = 1,
    execution_steps: list = None,
    system_message: str = None,
    attachments: list = None,
):
    """Helper to create AgentExecutionResult for tests.

    Args:
        response: Agent's final response
        token_count: Total tokens used
        cost: Total cost in dollars
        step_count: Number of execution steps
        execution_steps: List of execution step details
        system_message: System prompt used
        attachments: List of (name, content) tuples

    Returns:
        AgentExecutionResult instance
    """
    from tsugite.agent_runner.models import AgentExecutionResult

    return AgentExecutionResult(
        response=response,
        token_count=token_count,
        cost=cost,
        step_count=step_count,
        execution_steps=execution_steps or [],
        system_message=system_message,
        attachments=attachments or [],
    )


@pytest.fixture
def mock_default_agent_prefetch():
    """Mock prefetch results for default builtin agent.

    The default agent has two prefetch tools:
    - list_agents: Returns available agent list
    - get_skills_for_template: Returns available skills

    Returns:
        List suitable for mock.side_effect
    """
    return [
        "agents/example.md",  # list_agents
        [],  # get_skills_for_template
    ]
