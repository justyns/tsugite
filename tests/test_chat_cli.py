"""Tests for chat CLI functionality."""

import pytest

from tsugite.config import update_config
from tsugite.ui.chat import ChatManager


class TestChatCLI:
    """Test chat CLI loading and initialization."""

    @pytest.fixture
    def test_agent(self, tmp_path):
        """Create a test agent for chat."""
        agent_content = """---
name: test_chat
model: ollama:qwen2.5-coder:7b
tools: []
max_turns: 3
---

You are a helpful test assistant.

{% if chat_history %}
## Previous Conversation
{% for turn in chat_history %}
User: {{ turn.user_message }}
Assistant: {{ turn.agent_response }}
{% endfor %}
{% endif %}

## Current Request
{{ user_prompt }}
"""
        agent_path = tmp_path / "test_chat.md"
        agent_path.write_text(agent_content)
        return agent_path

    def test_chat_manager_initialization(self, test_agent):
        """Test that ChatManager initializes with agent path."""
        manager = ChatManager(
            agent_path=test_agent,
            model_override=None,
            max_history=50,
        )

        assert manager.agent_path == test_agent
        assert manager.max_history == 50
        assert len(manager.conversation_history) == 0

    def test_chat_manager_with_model_override(self, test_agent):
        """Test ChatManager with model override."""
        manager = ChatManager(
            agent_path=test_agent,
            model_override="ollama:llama3.2",
            max_history=10,
        )

        assert manager.model_override == "ollama:llama3.2"
        assert manager.max_history == 10

    def test_parse_agent_for_chat(self, test_agent):
        """Test that agent file can be parsed for chat UI."""
        from tsugite.md_agents import parse_agent_file

        agent = parse_agent_file(test_agent)

        assert agent.config.name == "test_chat"
        assert agent.config.model == "ollama:qwen2.5-coder:7b"
        assert agent.config.max_turns == 3

    def test_chat_cli_loads_agent_info(self, test_agent):
        """Test that chat CLI can load agent info without crashing."""
        from tsugite.md_agents import parse_agent_file
        from tsugite.models import resolve_effective_model

        agent = parse_agent_file(test_agent)
        agent_name = agent.config.name or test_agent.stem
        model = resolve_effective_model(agent_model=agent.config.model) or "unknown"

        assert agent_name == "test_chat"
        assert model == "ollama:qwen2.5-coder:7b"

    def test_chat_cli_with_model_override(self, test_agent):
        """Test agent info loading with model override."""
        from tsugite.md_agents import parse_agent_file
        from tsugite.models import resolve_effective_model

        agent = parse_agent_file(test_agent)
        agent_name = agent.config.name or test_agent.stem
        model = resolve_effective_model("ollama:custom-model", agent.config.model) or "unknown"

        assert agent_name == "test_chat"
        assert model == "ollama:custom-model"

    def test_save_conversation_uses_agent_config(self, test_agent, tmp_path):
        """Test that save_conversation accesses agent.config correctly."""
        manager = ChatManager(
            agent_path=test_agent,
            model_override=None,
            max_history=50,
        )

        # Add a test turn
        manager.add_turn("Hello", "Hi there!")

        # Save should work without errors
        save_path = tmp_path / "conversation.json"
        manager.save_conversation(save_path)

        assert save_path.exists()

        # Verify contents
        import json

        data = json.loads(save_path.read_text())
        assert data["agent"] == "test_chat"
        assert data["model"] == "ollama:qwen2.5-coder:7b"
        assert len(data["turns"]) == 1

    def test_chat_history_context_format(self, test_agent):
        """Test that chat_history is passed as list to agent."""

        from tsugite.ui.chat import ChatTurn

        manager = ChatManager(agent_path=test_agent)

        # Add some turns
        manager.add_turn("Hello", "Hi!")
        manager.add_turn("How are you?", "I'm good!")

        # Verify history is list of ChatTurn objects
        assert len(manager.conversation_history) == 2
        assert isinstance(manager.conversation_history[0], ChatTurn)
        assert manager.conversation_history[0].user_message == "Hello"
        assert manager.conversation_history[0].agent_response == "Hi!"

    def test_token_counting(self, test_agent):
        """Test that token counts can be provided when adding turns."""
        manager = ChatManager(agent_path=test_agent)

        # Add a turn with explicit token count
        manager.add_turn("Hello, how are you?", "I'm doing well, thank you!", token_count=25)

        # Verify token count was stored
        turn = manager.conversation_history[0]
        assert turn.token_count == 25

        # Get stats and verify total tokens
        stats = manager.get_stats()
        assert stats["total_tokens"] == 25

    def test_token_counting_multiple_turns(self, test_agent):
        """Test that token counts accumulate across multiple turns."""
        manager = ChatManager(agent_path=test_agent)

        # Add several turns with explicit token counts
        manager.add_turn("First question", "First answer", token_count=10)
        manager.add_turn("Second question", "Second answer", token_count=15)
        manager.add_turn("Third question", "Third answer", token_count=20)

        # Get stats
        stats = manager.get_stats()

        # Verify total tokens is sum of all turns
        expected_total = sum(turn.token_count for turn in manager.conversation_history if turn.token_count)
        assert stats["total_tokens"] == expected_total
        assert stats["total_tokens"] == 45

    def test_get_stats_model_from_agent_config(self, test_agent):
        """get_stats returns agent config model, not 'default'."""
        manager = ChatManager(agent_path=test_agent, model_override=None)
        stats = manager.get_stats()
        assert stats["model"] == "ollama:qwen2.5-coder:7b"
        assert stats["model"] != "default"

    def test_get_stats_model_no_model_uses_config_default(self, tmp_path, monkeypatch):
        """get_stats falls back to config default, not 'default' string."""
        agent_path = tmp_path / "no_model.md"
        agent_path.write_text("---\nname: no_model\ntools: []\n---\nHello\n{{ user_prompt }}")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        config_path = tmp_path / "tsugite" / "config.json"
        update_config(config_path, lambda cfg: setattr(cfg, "default_model", "openai:gpt-4"))
        manager = ChatManager(agent_path=agent_path, model_override=None)
        stats = manager.get_stats()
        assert stats["model"] == "openai:gpt-4"
