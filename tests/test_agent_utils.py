"""Tests for agent utility functions."""

from tsugite.agent_utils import build_inheritance_chain, list_local_agents


def test_build_inheritance_chain_no_extends(tmp_path):
    """Test inheritance chain for agent with no extends."""
    agent_file = tmp_path / "simple.md"
    agent_file.write_text("""---
name: simple
extends: none
---
""")

    chain = build_inheritance_chain(agent_file)

    assert len(chain) == 1
    assert chain[0][0] == "simple"


def test_build_inheritance_chain_with_parent(tmp_path):
    """Test inheritance chain with explicit parent."""
    tsugite_dir = tmp_path / ".tsugite"
    tsugite_dir.mkdir()

    # Parent
    parent_file = tsugite_dir / "parent.md"
    parent_file.write_text("""---
name: parent
extends: none
---
""")

    # Child
    child_file = tmp_path / "child.md"
    child_file.write_text("""---
name: child
extends: parent
---
""")

    chain = build_inheritance_chain(child_file)

    assert len(chain) == 2
    assert chain[0][0] == "parent"
    assert chain[1][0] == "child"


def test_build_inheritance_chain_three_levels(tmp_path):
    """Test three-level inheritance chain."""
    tsugite_dir = tmp_path / ".tsugite"
    tsugite_dir.mkdir()

    # Create empty default to prevent global default
    default_file = tsugite_dir / "default.md"
    default_file.write_text("""---
name: default
extends: none
---
""")

    # Grandparent
    grandparent_file = tsugite_dir / "grandparent.md"
    grandparent_file.write_text("""---
name: grandparent
extends: none
---
""")

    # Parent
    parent_file = tsugite_dir / "parent.md"
    parent_file.write_text("""---
name: parent
extends: grandparent
---
""")

    # Child
    child_file = tmp_path / "child.md"
    child_file.write_text("""---
name: child
extends: parent
---
""")

    chain = build_inheritance_chain(child_file)

    assert len(chain) == 3
    assert chain[0][0] == "grandparent"
    assert chain[1][0] == "parent"
    assert chain[2][0] == "child"


def test_build_inheritance_chain_auto_default(tmp_path):
    """Test auto-inheritance from default."""
    tsugite_dir = tmp_path / ".tsugite"
    tsugite_dir.mkdir()

    # Default agent
    default_file = tsugite_dir / "default.md"
    default_file.write_text("""---
name: default
extends: none
---
""")

    # Agent with no extends (should auto-inherit default)
    agent_file = tmp_path / "auto.md"
    agent_file.write_text("""---
name: auto
---
""")

    chain = build_inheritance_chain(agent_file)

    # Should have default + auto
    assert len(chain) == 2
    assert chain[0][0] == "default"
    assert chain[1][0] == "auto"


def test_list_local_agents_empty(tmp_path):
    """Test listing agents in empty directory."""
    result = list_local_agents(tmp_path)

    # Should have built-in agents even in empty directory
    assert len(result) == 1
    assert "Built-in" in result
    assert len(result["Built-in"]) == 4  # default, file_searcher, code_searcher, onboard


def test_list_local_agents_current_dir(tmp_path):
    """Test listing agents in current directory."""
    # Valid agent
    agent1 = tmp_path / "agent1.md"
    agent1.write_text("""---
name: agent1
---
""")

    # Another valid agent
    agent2 = tmp_path / "agent2.md"
    agent2.write_text("""---
name: agent2
---
""")

    # Not an agent (no frontmatter)
    readme = tmp_path / "README.md"
    readme.write_text("# README\nNot an agent")

    result = list_local_agents(tmp_path)

    assert "Current directory" in result
    assert len(result["Current directory"]) == 2
    agent_names = [f.stem for f in result["Current directory"]]
    assert "agent1" in agent_names
    assert "agent2" in agent_names
    assert "README" not in agent_names


def test_list_local_agents_tsugite_dir(tmp_path):
    """Test listing agents in .tsugite directory."""
    tsugite_dir = tmp_path / ".tsugite"
    tsugite_dir.mkdir()

    agent = tsugite_dir / "base.md"
    agent.write_text("""---
name: base
---
""")

    result = list_local_agents(tmp_path)

    assert ".tsugite/" in result
    assert len(result[".tsugite/"]) == 1
    assert result[".tsugite/"][0].stem == "base"


def test_list_local_agents_agents_dir(tmp_path):
    """Test listing agents in agents/ directory."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    agent = agents_dir / "custom.md"
    agent.write_text("""---
name: custom
---
""")

    result = list_local_agents(tmp_path)

    assert "agents/" in result
    assert len(result["agents/"]) == 1
    assert result["agents/"][0].stem == "custom"


def test_list_local_agents_multiple_locations(tmp_path):
    """Test listing agents across multiple locations."""
    # Current dir
    (tmp_path / "current.md").write_text("---\nname: current\n---\n")

    # .tsugite/
    tsugite_dir = tmp_path / ".tsugite"
    tsugite_dir.mkdir()
    (tsugite_dir / "base.md").write_text("---\nname: base\n---\n")

    # agents/
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "custom.md").write_text("---\nname: custom\n---\n")

    result = list_local_agents(tmp_path)

    # Should include built-in + 3 locations
    assert len(result) == 4
    assert "Built-in" in result
    assert "Current directory" in result
    assert ".tsugite/" in result
    assert "agents/" in result
