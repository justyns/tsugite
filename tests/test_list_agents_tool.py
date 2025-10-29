"""Tests for the list_agents tool."""

from unittest.mock import patch

from tsugite.tools.agents import list_agents


class TestListAgentsTool:
    """Test the list_agents tool function."""

    def test_list_agents_empty_directories(self, tmp_path, monkeypatch):
        """Test list_agents returns empty string when no agents found."""
        # Change to temp directory so no agents are found
        monkeypatch.chdir(tmp_path)

        # Mock both global paths and builtin path to return empty/non-existent
        fake_builtin_dir = tmp_path / "fake_builtins"  # Non-existent directory
        with (
            patch("tsugite.agent_inheritance.get_global_agents_paths", return_value=[]),
            patch("tsugite.agent_inheritance.get_builtin_agents_path", return_value=fake_builtin_dir),
        ):
            result = list_agents()
            assert result == ""

    def test_list_agents_finds_local_agents(self, tmp_path, monkeypatch):
        """Test list_agents discovers agents in local directories."""
        monkeypatch.chdir(tmp_path)

        # Create agents directory
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        # Create test agent
        agent_file = agents_dir / "test_agent.md"
        agent_file.write_text("""---
name: test_agent
description: A test agent for unit testing
---
Test content
""")

        result = list_agents()

        assert result != ""
        assert "test_agent" in result
        assert "A test agent for unit testing" in result
        assert "agents/test_agent.md" in result

    def test_list_agents_finds_tsugite_agents(self, tmp_path, monkeypatch):
        """Test list_agents discovers agents in .tsugite/agents/."""
        monkeypatch.chdir(tmp_path)

        # Create .tsugite/agents directory
        tsugite_dir = tmp_path / ".tsugite" / "agents"
        tsugite_dir.mkdir(parents=True)

        # Create test agent
        agent_file = tsugite_dir / "internal_agent.md"
        agent_file.write_text("""---
name: internal_agent
description: Internal team agent
---
Content
""")

        result = list_agents()

        assert "internal_agent" in result
        assert "Internal team agent" in result
        assert ".tsugite/agents/internal_agent.md" in result

    def test_list_agents_includes_all_agents(self, tmp_path, monkeypatch):
        """Test that all agents are included in the list, including package-provided ones."""
        monkeypatch.chdir(tmp_path)

        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        # Create a user agent with builtin-like name (should be included)
        builtin_like_file = agents_dir / "builtin-test.md"
        builtin_like_file.write_text("""---
name: builtin-test
description: User agent with builtin-like name
---
Content
""")

        # Create a normal agent
        normal_file = agents_dir / "normal.md"
        normal_file.write_text("""---
name: normal
description: Should be included
---
Content
""")

        result = list_agents()

        # Both user agents should be included
        assert "builtin-test" in result
        assert "normal" in result
        # Package-provided agents should also be included
        assert "default" in result
        assert "chat-assistant" in result

    def test_list_agents_priority_order(self, tmp_path, monkeypatch):
        """Test that higher priority paths win for duplicate agent names."""
        monkeypatch.chdir(tmp_path)

        # Create both .tsugite and agents directories
        tsugite_dir = tmp_path / ".tsugite" / "agents"
        tsugite_dir.mkdir(parents=True)
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        # Create same-named agent in both locations
        tsugite_agent = tsugite_dir / "duplicate.md"
        tsugite_agent.write_text("""---
name: duplicate
description: From .tsugite (higher priority)
---
""")

        agents_agent = agents_dir / "duplicate.md"
        agents_agent.write_text("""---
name: duplicate
description: From agents (lower priority)
---
""")

        # Mock global paths to avoid interference
        with patch("tsugite.agent_inheritance.get_global_agents_paths", return_value=[]):
            result = list_agents()

            # Should show higher priority duplicate + package-provided agents
            assert result.count("\n") == 2  # 3 agents total: duplicate, default, chat-assistant
            assert "From .tsugite (higher priority)" in result
            assert "From agents (lower priority)" not in result
            # Package-provided agents should also be listed
            assert "default" in result
            assert "chat-assistant" in result

    def test_list_agents_format(self, tmp_path, monkeypatch):
        """Test that list_agents returns proper markdown format."""
        monkeypatch.chdir(tmp_path)

        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        agent_file = agents_dir / "formatter.md"
        agent_file.write_text("""---
name: formatter
description: Formats code beautifully
---
""")

        result = list_agents()

        # Should be markdown list format
        assert result.startswith("- **")
        assert "**formatter**" in result
        assert "(`agents/formatter.md`)" in result
        assert ": Formats code beautifully" in result

    def test_list_agents_handles_invalid_files(self, tmp_path, monkeypatch):
        """Test that list_agents gracefully skips unparseable files."""
        monkeypatch.chdir(tmp_path)

        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        # Create invalid agent file (no frontmatter)
        invalid_file = agents_dir / "invalid.md"
        invalid_file.write_text("Just some text without frontmatter")

        # Create valid agent file
        valid_file = agents_dir / "valid.md"
        valid_file.write_text("""---
name: valid
description: Valid agent
---
""")

        # Should not raise an error, just skip invalid file
        result = list_agents()

        assert "valid" in result
        assert "invalid" not in result

    def test_list_agents_no_description(self, tmp_path, monkeypatch):
        """Test list_agents handles agents without descriptions."""
        monkeypatch.chdir(tmp_path)

        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        agent_file = agents_dir / "nodesc.md"
        agent_file.write_text("""---
name: nodesc
---
Content
""")

        result = list_agents()

        assert "nodesc" in result
        assert "No description" in result

    def test_list_agents_multiple_agents(self, tmp_path, monkeypatch):
        """Test list_agents with multiple agents."""
        monkeypatch.chdir(tmp_path)

        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        # Create three agents
        for i in range(3):
            agent_file = agents_dir / f"agent{i}.md"
            agent_file.write_text(f"""---
name: agent{i}
description: Agent number {i}
---
""")

        result = list_agents()

        # All three should be listed
        assert "agent0" in result
        assert "agent1" in result
        assert "agent2" in result
        # Should be newline separated
        assert result.count("\n") >= 2
