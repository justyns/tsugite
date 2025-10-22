"""Tests for agent inheritance system."""

from pathlib import Path

import pytest

from tsugite.agent_inheritance import (
    detect_circular_inheritance,
    find_agent_file,
    get_global_agents_paths,
    merge_agent_configs,
    resolve_agent_inheritance,
)
from tsugite.md_agents import AgentConfig, parse_agent


def test_get_global_agents_paths():
    """Test global agents path resolution."""
    paths = get_global_agents_paths()

    # Should have 2-3 paths depending on XDG_CONFIG_HOME
    assert len(paths) >= 2
    assert all(isinstance(p, Path) for p in paths)
    assert any(".tsugite" in str(p) for p in paths)
    assert any(".config" in str(p) for p in paths)


def test_merge_agent_configs_scalars():
    """Test merging scalar fields (child overwrites parent)."""
    parent = AgentConfig(
        name="parent",
        description="Parent agent",
        model="ollama:parent-model",
        max_steps=3,
    )

    child = AgentConfig(
        name="child",
        model="openai:gpt-4",
        max_steps=10,
    )

    merged = merge_agent_configs(parent, child)

    assert merged.name == "child"
    assert merged.model == "openai:gpt-4"
    assert merged.max_steps == 10


def test_merge_agent_configs_lists():
    """Test merging list fields (tools, prefetch)."""
    parent = AgentConfig(
        name="parent",
        tools=["read_file", "write_file"],
        prefetch=[{"tool": "get_env", "assign": "env"}],
    )

    child = AgentConfig(
        name="child",
        tools=["web_search", "read_file"],  # read_file is duplicate
        prefetch=[{"tool": "get_config", "assign": "config"}],
    )

    merged = merge_agent_configs(parent, child)

    # Tools should be merged and deduplicated, order preserved
    assert merged.tools == ["read_file", "write_file", "web_search"]

    # Prefetch should be concatenated (parent first)
    assert len(merged.prefetch) == 2
    assert merged.prefetch[0]["tool"] == "get_env"
    assert merged.prefetch[1]["tool"] == "get_config"


def test_merge_agent_configs_dicts():
    """Test merging dict fields (mcp_servers)."""
    parent = AgentConfig(
        name="parent",
        mcp_servers={"server1": ["tool1"], "server2": None},
    )

    child = AgentConfig(
        name="child",
        mcp_servers={"server2": ["tool2"], "server3": ["tool3"]},
    )

    merged = merge_agent_configs(parent, child)

    # Child keys should override parent
    assert merged.mcp_servers["server1"] == ["tool1"]
    assert merged.mcp_servers["server2"] == ["tool2"]
    assert merged.mcp_servers["server3"] == ["tool3"]


def test_merge_agent_configs_instructions():
    """Test merging instruction strings (concatenate)."""
    parent = AgentConfig(
        name="parent",
        instructions="Parent instructions.",
    )

    child = AgentConfig(
        name="child",
        instructions="Child instructions.",
    )

    merged = merge_agent_configs(parent, child)

    assert "Parent instructions." in merged.instructions
    assert "Child instructions." in merged.instructions
    assert merged.instructions == "Parent instructions.\n\nChild instructions."


def test_merge_agent_configs_empty_instructions():
    """Test merging when one side has empty instructions."""
    parent = AgentConfig(
        name="parent",
        instructions="Parent only.",
    )

    child = AgentConfig(
        name="child",
        instructions="",
    )

    merged = merge_agent_configs(parent, child)
    assert merged.instructions == "Parent only."


def test_merge_agent_configs_attachments():
    """Test merging attachments list (merge and deduplicate)."""
    parent = AgentConfig(
        name="parent",
        attachments=["standards", "api-docs"],
    )

    child = AgentConfig(
        name="child",
        attachments=["examples", "standards"],  # standards is duplicate
    )

    merged = merge_agent_configs(parent, child)

    # Attachments should be merged and deduplicated, order preserved
    assert merged.attachments == ["standards", "api-docs", "examples"]


def test_merge_agent_configs_reasoning_effort():
    """Test merging reasoning_effort scalar (child overwrites parent)."""
    parent = AgentConfig(
        name="parent",
        reasoning_effort="low",
    )

    child = AgentConfig(
        name="child",
        reasoning_effort="high",
    )

    merged = merge_agent_configs(parent, child)
    assert merged.reasoning_effort == "high"

    # Test child with None inherits from parent
    child_none = AgentConfig(
        name="child2",
        reasoning_effort=None,
    )

    merged2 = merge_agent_configs(parent, child_none)
    assert merged2.reasoning_effort == "low"


def test_merge_agent_configs_custom_tools():
    """Test merging custom_tools (merge and deduplicate by name)."""
    parent = AgentConfig(
        name="parent",
        custom_tools=[
            {"name": "tool1", "command": "echo parent1"},
            {"name": "tool2", "command": "echo parent2"},
        ],
    )

    child = AgentConfig(
        name="child",
        custom_tools=[
            {"name": "tool2", "command": "echo child2"},  # Overrides parent's tool2
            {"name": "tool3", "command": "echo child3"},
        ],
    )

    merged = merge_agent_configs(parent, child)

    # Should have 3 tools total (tool1 from parent, tool2 from child, tool3 from child)
    assert len(merged.custom_tools) == 3

    # Build dict by name for easier testing
    tools_by_name = {tool["name"]: tool for tool in merged.custom_tools}
    assert tools_by_name["tool1"]["command"] == "echo parent1"
    assert tools_by_name["tool2"]["command"] == "echo child2"  # Child overrides parent
    assert tools_by_name["tool3"]["command"] == "echo child3"


def test_merge_agent_configs_text_mode():
    """Test merging text_mode boolean (child overwrites if True).

    Since text_mode defaults to False, we can't distinguish between "not set"
    and "explicitly set to False". The merge logic treats False as the default,
    so parent's value is inherited unless child explicitly sets True.
    """
    parent = AgentConfig(
        name="parent",
        text_mode=True,
    )

    child = AgentConfig(
        name="child",
        text_mode=False,  # Default value, so parent's True is inherited
    )

    merged = merge_agent_configs(parent, child)
    # Child has default value (False), so parent's True is inherited
    assert merged.text_mode

    # When child explicitly sets True, it overrides parent
    child_true = AgentConfig(
        name="child2",
        text_mode=True,
    )

    merged2 = merge_agent_configs(parent, child_true)
    assert merged2.text_mode

    # When both are False, result is False
    parent_false = AgentConfig(
        name="parent2",
        text_mode=False,
    )

    merged3 = merge_agent_configs(parent_false, child)
    assert not merged3.text_mode


def test_detect_circular_inheritance():
    """Test circular inheritance detection."""
    path1 = Path("/fake/agent1.md")
    path2 = Path("/fake/agent2.md")

    chain = {path1, path2}

    # Should detect circular reference
    assert detect_circular_inheritance(path1, chain) is True
    assert detect_circular_inheritance(path2, chain) is True

    # New path should not be circular
    path3 = Path("/fake/agent3.md")
    assert detect_circular_inheritance(path3, chain) is False


def test_find_agent_file_with_path(tmp_path):
    """Test finding agent file when reference is a path."""
    # Create a test agent file
    agent_dir = tmp_path / "agents"
    agent_dir.mkdir()
    agent_file = agent_dir / "test.md"
    agent_file.write_text("---\nname: test\n---\n")

    # Test relative path
    current_dir = tmp_path
    found = find_agent_file("agents/test.md", current_dir)

    assert found is not None
    assert found.name == "test.md"


def test_find_agent_file_by_name(tmp_path):
    """Test finding agent file by name in standard locations."""
    # Create .tsugite directory with agent
    tsugite_dir = tmp_path / ".tsugite"
    tsugite_dir.mkdir()
    agent_file = tsugite_dir / "base.md"
    agent_file.write_text("---\nname: base\n---\n")

    # Test name-based lookup
    found = find_agent_file("base", tmp_path)

    assert found is not None
    assert found.name == "base.md"


def test_find_agent_file_not_found(tmp_path):
    """Test when agent file is not found."""
    found = find_agent_file("nonexistent", tmp_path)
    assert found is None


def test_resolve_agent_inheritance_no_extends(tmp_path):
    """Test agent without extends field."""
    agent_file = tmp_path / "simple.md"
    agent_file.write_text("---\nname: simple\nmodel: ollama:test\n---\nContent")

    agent = parse_agent(agent_file.read_text(), agent_file)
    resolved = resolve_agent_inheritance(agent, agent_file)

    # Should return essentially the same agent
    assert resolved.config.name == "simple"
    assert resolved.config.model == "ollama:test"


def test_resolve_agent_inheritance_opt_out(tmp_path):
    """Test agent that opts out of default inheritance."""
    agent_file = tmp_path / "standalone.md"
    agent_file.write_text("---\nname: standalone\nextends: none\n---\nContent")

    agent = parse_agent(agent_file.read_text(), agent_file)
    resolved = resolve_agent_inheritance(agent, agent_file)

    assert resolved.config.extends == "none"


def test_resolve_agent_inheritance_with_parent(tmp_path):
    """Test agent that extends a parent."""
    # Create parent agent
    tsugite_dir = tmp_path / ".tsugite"
    tsugite_dir.mkdir()

    parent_file = tsugite_dir / "base.md"
    parent_file.write_text(
        """---
name: base
extends: none
model: ollama:base-model
tools: [read_file, write_file]
instructions: Base instructions.
---
Base content
"""
    )

    # Create child agent
    child_file = tmp_path / "child.md"
    child_file.write_text(
        """---
name: child
extends: base
tools: [web_search]
instructions: Child instructions.
---
Child content
"""
    )

    # Parse and resolve child
    child_agent = parse_agent(child_file.read_text(), child_file)
    resolved = resolve_agent_inheritance(child_agent, child_file)

    # Check merged config
    assert resolved.config.name == "child"
    assert resolved.config.model == "ollama:base-model"
    assert set(resolved.config.tools) == {"read_file", "write_file", "web_search"}
    assert "Base instructions." in resolved.config.instructions
    assert "Child instructions." in resolved.config.instructions
    # Content should be merged (parent first, then child)
    assert "Base content" in resolved.content
    assert "Child content" in resolved.content
    assert resolved.content.index("Base content") < resolved.content.index("Child content")


def test_resolve_agent_inheritance_chain(tmp_path):
    """Test multi-level inheritance chain."""
    tsugite_dir = tmp_path / ".tsugite"
    tsugite_dir.mkdir()

    # Create a test default that opts out (to prevent global default from being used)
    default_file = tsugite_dir / "default.md"
    default_file.write_text(
        """---
name: default
extends: none
tools: []
---
"""
    )

    # Grandparent
    grandparent_file = tsugite_dir / "grandparent.md"
    grandparent_file.write_text(
        """---
name: grandparent
extends: none
tools: [tool1]
---
"""
    )

    # Parent extends grandparent (also opts out of default to keep test focused)
    parent_file = tsugite_dir / "parent.md"
    parent_file.write_text(
        """---
name: parent
extends: grandparent
tools: [tool2]
---
"""
    )

    # Child extends parent (uses grandparent's opt-out implicitly)
    child_file = tmp_path / "child.md"
    child_file.write_text(
        """---
name: child
extends: parent
tools: [tool3]
---
"""
    )

    child_agent = parse_agent(child_file.read_text(), child_file)
    resolved = resolve_agent_inheritance(child_agent, child_file)

    # Should have all three tools
    assert set(resolved.config.tools) == {"tool1", "tool2", "tool3"}


def test_circular_inheritance_error(tmp_path):
    """Test that circular inheritance raises error."""
    tsugite_dir = tmp_path / ".tsugite"
    tsugite_dir.mkdir()

    # Agent A extends B
    agent_a = tsugite_dir / "a.md"
    agent_a.write_text(
        """---
name: a
extends: b
---
"""
    )

    # Agent B extends A (circular)
    agent_b = tsugite_dir / "b.md"
    agent_b.write_text(
        """---
name: b
extends: a
---
"""
    )

    # Try to resolve A
    a_agent = parse_agent(agent_a.read_text(), agent_a)

    with pytest.raises(ValueError, match="Circular inheritance"):
        resolve_agent_inheritance(a_agent, agent_a)


def test_missing_parent_error(tmp_path):
    """Test that missing parent agent raises error."""
    child_file = tmp_path / "child.md"
    child_file.write_text(
        """---
name: child
extends: nonexistent
---
"""
    )

    child_agent = parse_agent(child_file.read_text(), child_file)

    with pytest.raises(ValueError, match="Extended agent not found"):
        resolve_agent_inheritance(child_agent, child_file)


def test_merge_preserves_child_name():
    """Test that merge always uses child name."""
    parent = AgentConfig(name="parent", description="Parent")
    child = AgentConfig(name="child", description="")

    merged = merge_agent_configs(parent, child)

    assert merged.name == "child"
    assert merged.description == "Parent"


def test_merge_with_empty_parent():
    """Test merging with minimal parent config."""
    parent = AgentConfig(name="parent")
    child = AgentConfig(
        name="child",
        model="openai:gpt-4",
        tools=["tool1"],
    )

    merged = merge_agent_configs(parent, child)

    assert merged.name == "child"
    assert merged.model == "openai:gpt-4"
    assert merged.tools == ["tool1"]


def test_default_max_steps_inheritance():
    """Test that default max_steps value doesn't override parent."""
    parent = AgentConfig(name="parent", max_steps=10)
    child = AgentConfig(name="child")  # max_steps defaults to 5

    merged = merge_agent_configs(parent, child)

    # Child's default 5 should not override parent's explicit 10
    assert merged.max_steps == 10
