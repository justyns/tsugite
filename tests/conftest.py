"""Test configuration and fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator


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
model: openai:gpt-4o-mini
max_steps: 5
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
model: ollama:qwen2.5-coder:7b
max_steps: 3
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
model: ollama:qwen2.5-coder:7b
max_steps: 5
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
model: openai:gpt-4o-mini
max_steps: 3
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


@pytest.fixture
def file_tools(reset_tool_registry):
    """Register file system tools for testing."""
    from tsugite.tools.fs import (
        write_file,
        read_file,
        list_files,
        file_exists,
        create_directory,
    )
    from tsugite.tools import tool

    # Re-register the tools after registry reset
    tool(write_file)
    tool(read_file)
    tool(list_files)
    tool(file_exists)
    tool(create_directory)
