"""Tests for the agent markdown parser."""

from pathlib import Path

import pytest

from tsugite.md_agents import (
    Agent,
    AgentConfig,
    extract_directives,
    parse_agent_file,
    validate_agent,
)


def test_parse_valid_agent_file(sample_agent_file):
    """Test parsing a valid agent file."""
    agent = parse_agent_file(sample_agent_file)

    assert isinstance(agent, Agent)
    assert agent.name == "test_agent"
    assert agent.config.model == "openai:gpt-4o-mini"
    assert agent.config.max_turns == 5
    assert "read_file" in agent.config.tools
    assert "write_file" in agent.config.tools
    assert agent.config.permissions_profile == "test_safe"
    assert "Provide concise" in agent.config.instructions

    # Check prefetch configuration
    assert len(agent.config.prefetch) == 1
    prefetch = agent.config.prefetch[0]
    assert prefetch["tool"] == "search_memory"
    assert prefetch["assign"] == "memories"

    # Check content
    assert "You are a test agent." in agent.content
    assert "{{ user_prompt }}" in agent.content


def test_parse_minimal_agent_file(temp_dir):
    """Test parsing an agent file with minimal configuration."""
    content = """---
name: minimal_agent
extends: none
---

# Simple Agent
Just a basic agent.
"""
    agent_file = temp_dir / "minimal.md"
    agent_file.write_text(content)

    agent = parse_agent_file(agent_file)

    assert agent.name == "minimal_agent"
    assert agent.config.model is None  # Model is optional, will use config default
    assert agent.config.max_turns == 5  # Default
    assert agent.config.tools == []  # Default
    assert agent.config.prefetch == []  # Default


def test_parse_nonexistent_file(temp_dir):
    """Test parsing a file that doesn't exist."""
    nonexistent = temp_dir / "nonexistent.md"

    with pytest.raises(FileNotFoundError, match="Agent file not found"):
        parse_agent_file(nonexistent)


def test_parse_file_without_frontmatter(temp_dir):
    """Test parsing a file without YAML frontmatter."""
    content = "# Just markdown content\nNo frontmatter here."
    agent_file = temp_dir / "no_frontmatter.md"
    agent_file.write_text(content)

    with pytest.raises(ValueError, match="must start with YAML frontmatter"):
        parse_agent_file(agent_file)


def test_parse_file_invalid_frontmatter(temp_dir):
    """Test parsing a file with invalid YAML frontmatter."""
    content = """---
name: test
invalid: yaml: content: here
---

# Content
"""
    agent_file = temp_dir / "invalid_yaml.md"
    agent_file.write_text(content)

    with pytest.raises(ValueError, match="Invalid YAML frontmatter"):
        parse_agent_file(agent_file)


def test_parse_file_incomplete_frontmatter(temp_dir):
    """Test parsing a file with incomplete frontmatter markers."""
    content = """---
name: test
model: ollama:test

# Content without closing frontmatter
"""
    agent_file = temp_dir / "incomplete.md"
    agent_file.write_text(content)

    with pytest.raises(ValueError, match="Invalid YAML frontmatter format"):
        parse_agent_file(agent_file)


def test_agent_config_defaults():
    """Test AgentConfig default values."""
    config = AgentConfig(name="test")

    assert config.name == "test"
    assert config.model is None  # Model is optional, will use config default
    assert config.max_turns == 5
    assert config.tools == []
    assert config.prefetch == []
    assert config.permissions_profile == "default"
    assert config.context_budget == {"tokens": 8000, "priority": ["system", "task"]}
    assert config.instructions == ""


def test_extract_directives_basic():
    """Test extracting basic directives from content."""
    content = """
# Test Content

<!-- tsu:tool name=read_file args={"path": "test.txt"} assign=content -->

Some text.

<!-- tsu:await output=result -->

More text.
"""

    directives = extract_directives(content)

    assert len(directives) == 2

    # First directive
    tool_directive = directives[0]
    assert tool_directive["type"] == "tool"
    assert tool_directive["name"] == "read_file"
    assert tool_directive["assign"] == "content"

    # Second directive
    await_directive = directives[1]
    assert await_directive["type"] == "await"
    assert "output=result" in await_directive["raw_args"]


def test_extract_directives_empty():
    """Test extracting directives from content without any."""
    content = """
# No Directives

Just plain markdown content here.
No special comments.
"""

    directives = extract_directives(content)
    assert len(directives) == 0


def test_extract_directives_malformed():
    """Test extracting directives with malformed syntax."""
    content = """
<!-- tsu:tool incomplete args -->
<!-- not-tsu:tool name=test -->
<!-- tsu:tool name=valid_tool args={"test": "value"} -->
"""

    directives = extract_directives(content)

    # Should only extract the valid one
    assert len(directives) == 2  # One incomplete, one valid
    valid_directive = [d for d in directives if d.get("name") == "valid_tool"][0]
    assert valid_directive["type"] == "tool"


def test_validate_agent_valid(sample_agent_file):
    """Test validating a valid agent."""
    agent = parse_agent_file(sample_agent_file)
    errors = validate_agent(agent)

    assert len(errors) == 0


def test_validate_agent_missing_name(temp_dir):
    """Test validating an agent without a name."""
    content = """---
model: ollama:test
---

# Agent without name
"""
    agent_file = temp_dir / "no_name.md"
    agent_file.write_text(content)

    # This should fail during parsing since name is required
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        parse_agent_file(agent_file)


def test_validate_agent_missing_model(temp_dir):
    """Test validating an agent without a model (should use config default)."""
    content = """---
name: test_agent
extends: none
---

# Agent
"""
    agent_file = temp_dir / "no_model.md"
    agent_file.write_text(content)

    agent = parse_agent_file(agent_file)
    errors = validate_agent(agent)

    # Should be valid (will use config default at runtime)
    assert len(errors) == 0
    assert agent.config.model is None  # Model is optional


def test_validate_agent_invalid_tools(temp_dir):
    """Test validating an agent with invalid tool specifications."""
    content = """---
name: test_agent
tools: [123, "valid_tool", null]
---

# Agent
"""
    agent_file = temp_dir / "invalid_tools.md"
    agent_file.write_text(content)

    # Pydantic now validates tools at parse time - should raise ValidationError
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        parse_agent_file(agent_file)


def test_validate_agent_directive_errors(temp_dir):
    """Test validating an agent with directive errors."""
    content = """---
name: test_agent
---

# Agent

<!-- tsu:tool args={"test": "value"} -->
<!-- tsu:tool name=valid_tool args={"valid": true} -->
"""
    agent_file = temp_dir / "directive_errors.md"
    agent_file.write_text(content)

    agent = parse_agent_file(agent_file)
    errors = validate_agent(agent)

    # Should have error for tool directive without name
    assert len(errors) > 0
    assert any("Tool directive missing name" in error for error in errors)


def test_agent_file_path_preserved(sample_agent_file):
    """Test that the original file path is preserved in the agent."""
    agent = parse_agent_file(sample_agent_file)

    assert agent.file_path == sample_agent_file
    assert agent.file_path.exists()


def test_complex_frontmatter(temp_dir):
    """Test parsing agent with complex frontmatter structure."""
    content = """---
name: complex_agent
extends: none
model: ollama:qwen2.5-coder:14b
max_turns: 10
tools:
  - read_file
  - write_file
  - fetch_json
prefetch:
  - tool: search_memory
    args:
      query: "test query"
      top_k: 5
    assign: memories
  - tool: git_status
    assign: git_info
permissions_profile: automation_safe
context_budget:
  tokens: 16000
  priority:
    - system
    - task
    - memories
  strategy: simple_truncate
---

# Complex Agent

This agent has a complex configuration.
"""
    agent_file = temp_dir / "complex.md"
    agent_file.write_text(content)

    agent = parse_agent_file(agent_file)

    assert agent.name == "complex_agent"
    assert agent.config.model == "ollama:qwen2.5-coder:14b"
    assert agent.config.max_turns == 10
    assert len(agent.config.tools) == 3
    assert len(agent.config.prefetch) == 2

    # Check nested structure
    first_prefetch = agent.config.prefetch[0]
    assert first_prefetch["args"]["query"] == "test query"
    assert first_prefetch["args"]["top_k"] == 5

    assert agent.config.context_budget["strategy"] == "simple_truncate"
    assert len(agent.config.context_budget["priority"]) == 3


def test_parse_docs_example_agent():
    """Parse the docs example agent to mirror the developer demo."""
    example_path = Path("docs/examples/agents/system_monitor.md")
    if not example_path.exists():
        pytest.skip("docs example agent not present")

    agent = parse_agent_file(example_path)

    assert agent.name == agent.config.name
    assert agent.config.tools  # Should include at least one tool

    directives = extract_directives(agent.content)
    assert len(directives) >= 1


def test_initial_tasks_simple_string_format(temp_dir):
    """Test parsing agent with initial_tasks as simple strings."""
    content = """---
name: task_agent
extends: none
initial_tasks:
  - "Analyze requirements"
  - "Write code"
  - "Test implementation"
---

# Task Agent
"""
    agent_file = temp_dir / "task_agent.md"
    agent_file.write_text(content)

    agent = parse_agent_file(agent_file)

    assert len(agent.config.initial_tasks) == 3
    # Check normalization to dicts
    assert agent.config.initial_tasks[0]["title"] == "Analyze requirements"
    assert agent.config.initial_tasks[0]["status"] == "pending"
    assert agent.config.initial_tasks[0]["optional"] is False
    assert agent.config.initial_tasks[1]["title"] == "Write code"
    assert agent.config.initial_tasks[2]["title"] == "Test implementation"


def test_initial_tasks_detailed_dict_format(temp_dir):
    """Test parsing agent with initial_tasks as detailed dicts."""
    content = """---
name: task_agent
extends: none
initial_tasks:
  - title: "Required task"
    status: pending
    optional: false
  - title: "Optional enhancement"
    status: pending
    optional: true
  - title: "Already in progress"
    status: in_progress
---

# Task Agent
"""
    agent_file = temp_dir / "task_agent.md"
    agent_file.write_text(content)

    agent = parse_agent_file(agent_file)

    assert len(agent.config.initial_tasks) == 3

    # Check first task
    assert agent.config.initial_tasks[0]["title"] == "Required task"
    assert agent.config.initial_tasks[0]["status"] == "pending"
    assert agent.config.initial_tasks[0]["optional"] is False

    # Check optional task
    assert agent.config.initial_tasks[1]["title"] == "Optional enhancement"
    assert agent.config.initial_tasks[1]["optional"] is True

    # Check in-progress task
    assert agent.config.initial_tasks[2]["title"] == "Already in progress"
    assert agent.config.initial_tasks[2]["status"] == "in_progress"


def test_initial_tasks_mixed_format(temp_dir):
    """Test parsing agent with mixed string and dict initial_tasks."""
    content = """---
name: task_agent
extends: none
initial_tasks:
  - "Simple required task"
  - title: "Optional detailed task"
    optional: true
  - "Another simple task"
---

# Task Agent
"""
    agent_file = temp_dir / "task_agent.md"
    agent_file.write_text(content)

    agent = parse_agent_file(agent_file)

    assert len(agent.config.initial_tasks) == 3

    # Check string format task
    assert agent.config.initial_tasks[0]["title"] == "Simple required task"
    assert agent.config.initial_tasks[0]["optional"] is False

    # Check dict format task
    assert agent.config.initial_tasks[1]["title"] == "Optional detailed task"
    assert agent.config.initial_tasks[1]["optional"] is True

    # Check another string format
    assert agent.config.initial_tasks[2]["title"] == "Another simple task"
    assert agent.config.initial_tasks[2]["optional"] is False


def test_initial_tasks_empty_list(temp_dir):
    """Test parsing agent with empty initial_tasks list."""
    content = """---
name: task_agent
extends: none
initial_tasks: []
---

# Task Agent
"""
    agent_file = temp_dir / "task_agent.md"
    agent_file.write_text(content)

    agent = parse_agent_file(agent_file)

    assert agent.config.initial_tasks == []


def test_initial_tasks_defaults_to_empty(temp_dir):
    """Test that initial_tasks defaults to empty list when not specified."""
    content = """---
name: task_agent
extends: none
---

# Task Agent
"""
    agent_file = temp_dir / "task_agent.md"
    agent_file.write_text(content)

    agent = parse_agent_file(agent_file)

    assert agent.config.initial_tasks == []


def test_initial_tasks_invalid_type(temp_dir):
    """Test that invalid initial_tasks entries raise an error."""
    content = """---
name: task_agent
extends: none
initial_tasks:
  - "Valid string"
  - 123
---

# Task Agent
"""
    agent_file = temp_dir / "task_agent.md"
    agent_file.write_text(content)

    # Pydantic now validates initial_tasks at parse time - should raise ValidationError
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        parse_agent_file(agent_file)
