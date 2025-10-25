"""Tests for conversation history system."""

from datetime import datetime, timezone

import pytest

from tsugite.history import (
    delete_conversation,
    generate_conversation_id,
    get_conversation_metadata,
    get_history_dir,
    load_conversation,
    prune_conversations,
    query_index,
    rebuild_index,
    remove_from_index,
    save_turn_to_history,
    update_index,
)
from tsugite.ui.chat_history import (
    format_conversation_for_display,
    get_machine_name,
    save_chat_turn,
    start_conversation,
)


class TestConversationID:
    """Tests for conversation ID generation."""

    def test_generate_conversation_id(self):
        """Test generating a unique conversation ID."""
        agent_name = "test_agent"
        conv_id = generate_conversation_id(agent_name)

        assert conv_id is not None
        assert isinstance(conv_id, str)
        assert "test_agent" in conv_id
        assert len(conv_id) > 0

    def test_generate_conversation_id_with_timestamp(self):
        """Test generating ID with specific timestamp."""
        agent_name = "chat"
        timestamp = datetime(2025, 10, 24, 10, 30, 0, tzinfo=timezone.utc)
        conv_id = generate_conversation_id(agent_name, timestamp)

        assert conv_id.startswith("20251024_103000")
        assert "chat" in conv_id

    def test_generate_conversation_id_unique(self):
        """Test that generated IDs are unique."""
        agent_name = "agent"
        id1 = generate_conversation_id(agent_name)
        id2 = generate_conversation_id(agent_name)

        assert id1 != id2


class TestStorage:
    """Tests for JSONL storage operations."""

    def test_get_history_dir(self, tmp_path, monkeypatch):
        """Test getting history directory path."""

        # Mock XDG data path to use tmp_path
        def mock_get_xdg_data_path(subdir):
            return tmp_path / "tsugite" / subdir

        monkeypatch.setattr("tsugite.history.storage.get_xdg_data_path", mock_get_xdg_data_path)

        history_dir = get_history_dir()
        assert history_dir == tmp_path / "tsugite" / "history"

    def test_save_and_load_conversation(self, tmp_path, monkeypatch):
        """Test saving and loading conversation turns."""
        # Mock history dir
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)

        conv_id = "20251024_103000_test_abc123"

        # Save metadata
        metadata = {
            "type": "metadata",
            "id": conv_id,
            "agent": "test_agent",
            "model": "openai:gpt-4o",
            "machine": "test_machine",
            "created_at": "2025-10-24T10:30:00+00:00",
        }
        save_turn_to_history(conv_id, metadata)

        # Save turn
        turn = {
            "type": "turn",
            "timestamp": "2025-10-24T10:30:05+00:00",
            "user": "Hello",
            "assistant": "Hi there!",
            "tools": [],
            "tokens": 50,
            "cost": 0.001,
        }
        save_turn_to_history(conv_id, turn)

        # Load conversation
        turns = load_conversation(conv_id)

        assert len(turns) == 2
        assert turns[0]["type"] == "metadata"
        assert turns[0]["id"] == conv_id
        assert turns[1]["type"] == "turn"
        assert turns[1]["user"] == "Hello"

    def test_load_nonexistent_conversation(self, tmp_path, monkeypatch):
        """Test loading a conversation that doesn't exist."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)

        with pytest.raises(FileNotFoundError):
            load_conversation("nonexistent_conversation")

    def test_save_with_malformed_line(self, tmp_path, monkeypatch):
        """Test loading conversation with malformed JSONL line."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)

        conv_id = "test_malformed"
        conv_file = tmp_path / f"{conv_id}.jsonl"

        # Write a valid line and a malformed line
        conv_file.write_text('{"type": "metadata", "id": "test"}\nnot valid json\n{"type": "turn"}')

        # Should skip malformed line and return valid lines
        turns = load_conversation(conv_id)
        assert len(turns) == 2  # Should have 2 valid turns

    def test_delete_conversation(self, tmp_path, monkeypatch):
        """Test deleting a conversation."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)

        conv_id = "test_delete"
        save_turn_to_history(conv_id, {"type": "metadata"})

        # Delete conversation
        result = delete_conversation(conv_id)
        assert result is True

        # Verify it's deleted
        with pytest.raises(FileNotFoundError):
            load_conversation(conv_id)

        # Try deleting again
        result = delete_conversation(conv_id)
        assert result is False

    def test_prune_conversations_by_count(self, tmp_path, monkeypatch):
        """Test pruning conversations by count."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)

        # Create 5 conversations
        for i in range(5):
            conv_id = f"conversation_{i}"
            save_turn_to_history(conv_id, {"type": "metadata"})

        # Keep only 3 most recent
        deleted = prune_conversations(keep_count=3)
        assert deleted == 2

        # Verify only 3 remain
        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 3


class TestIndex:
    """Tests for JSON index operations."""

    def test_update_and_query_index(self, tmp_path, monkeypatch):
        """Test updating and querying the index."""
        monkeypatch.setattr("tsugite.history.index.get_history_dir", lambda: tmp_path)

        conv_id = "test_conversation"
        metadata = {
            "agent": "chat_assistant",
            "model": "openai:gpt-4o",
            "machine": "laptop",
            "created_at": "2025-10-24T10:00:00+00:00",
            "updated_at": "2025-10-24T10:30:00+00:00",
            "turn_count": 5,
            "total_tokens": 1000,
            "total_cost": 0.05,
        }

        update_index(conv_id, metadata)

        # Query index
        results = query_index()
        assert len(results) == 1
        assert results[0]["conversation_id"] == conv_id
        assert results[0]["agent"] == "chat_assistant"

    def test_query_index_by_machine(self, tmp_path, monkeypatch):
        """Test querying index by machine name."""
        monkeypatch.setattr("tsugite.history.index.get_history_dir", lambda: tmp_path)

        # Add conversations on different machines
        update_index("conv1", {"machine": "laptop", "agent": "agent1"})
        update_index("conv2", {"machine": "desktop", "agent": "agent2"})

        results = query_index(machine="laptop")
        assert len(results) == 1
        assert results[0]["conversation_id"] == "conv1"

    def test_query_index_by_agent(self, tmp_path, monkeypatch):
        """Test querying index by agent name."""
        monkeypatch.setattr("tsugite.history.index.get_history_dir", lambda: tmp_path)

        update_index("conv1", {"agent": "chat", "machine": "laptop"})
        update_index("conv2", {"agent": "research", "machine": "laptop"})

        results = query_index(agent="chat")
        assert len(results) == 1
        assert results[0]["conversation_id"] == "conv1"

    def test_query_index_with_limit(self, tmp_path, monkeypatch):
        """Test querying index with result limit."""
        monkeypatch.setattr("tsugite.history.index.get_history_dir", lambda: tmp_path)

        # Add 5 conversations
        for i in range(5):
            update_index(f"conv{i}", {"agent": "test", "machine": "laptop"})

        results = query_index(limit=3)
        assert len(results) == 3

    def test_rebuild_index(self, tmp_path, monkeypatch):
        """Test rebuilding index from JSONL files."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.index.get_history_dir", lambda: tmp_path)

        # Create conversation files without index
        conv_id = "test_rebuild"
        save_turn_to_history(
            conv_id,
            {
                "type": "metadata",
                "agent": "test_agent",
                "model": "test_model",
                "machine": "test_machine",
                "timestamp": "2025-10-24T10:00:00+00:00",
            },
        )
        save_turn_to_history(
            conv_id, {"type": "turn", "timestamp": "2025-10-24T10:01:00+00:00", "tokens": 100, "cost": 0.01}
        )

        # Rebuild index
        count = rebuild_index()
        assert count == 1

        # Verify index was created
        metadata = get_conversation_metadata(conv_id)
        assert metadata is not None
        assert metadata["agent"] == "test_agent"

    def test_remove_from_index(self, tmp_path, monkeypatch):
        """Test removing conversation from index."""
        monkeypatch.setattr("tsugite.history.index.get_history_dir", lambda: tmp_path)

        conv_id = "test_remove"
        update_index(conv_id, {"agent": "test", "machine": "laptop"})

        # Remove from index
        result = remove_from_index(conv_id)
        assert result is True

        # Verify it's removed
        metadata = get_conversation_metadata(conv_id)
        assert metadata is None

        # Try removing again
        result = remove_from_index(conv_id)
        assert result is False


class TestChatIntegration:
    """Tests for chat history integration."""

    def test_start_conversation(self, tmp_path, monkeypatch):
        """Test starting a new conversation."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.index.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.ui.chat_history.get_machine_name", lambda: "test_machine")

        agent_name = "chat_assistant"
        model = "openai:gpt-4o"

        conv_id = start_conversation(agent_name, model)

        assert conv_id is not None
        assert agent_name in conv_id

        # Verify conversation file was created
        turns = load_conversation(conv_id)
        assert len(turns) == 1
        assert turns[0]["type"] == "metadata"
        assert turns[0]["agent"] == agent_name

        # Verify index was created
        metadata = get_conversation_metadata(conv_id)
        assert metadata is not None
        assert metadata["turn_count"] == 0

    def test_save_chat_turn(self, tmp_path, monkeypatch):
        """Test saving a chat turn."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.index.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.ui.chat_history.get_machine_name", lambda: "test_machine")

        # Start conversation
        conv_id = start_conversation("test_agent", "test_model")

        # Save a turn
        save_chat_turn(
            conversation_id=conv_id,
            user_message="Hello",
            agent_response="Hi there!",
            tool_calls=["web_search"],
            token_count=100,
            cost=0.002,
        )

        # Verify turn was saved
        turns = load_conversation(conv_id)
        assert len(turns) == 2  # metadata + turn
        assert turns[1]["type"] == "turn"
        assert turns[1]["user"] == "Hello"
        assert turns[1]["assistant"] == "Hi there!"

        # Verify index was updated
        metadata = get_conversation_metadata(conv_id)
        assert metadata["turn_count"] == 1
        assert metadata["total_tokens"] == 100
        assert metadata["total_cost"] == 0.002

    def test_save_multiple_turns(self, tmp_path, monkeypatch):
        """Test saving multiple turns and cumulative stats."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.index.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.ui.chat_history.get_machine_name", lambda: "test_machine")

        conv_id = start_conversation("test_agent", "test_model")

        # Save multiple turns
        for i in range(3):
            save_chat_turn(
                conversation_id=conv_id,
                user_message=f"Message {i}",
                agent_response=f"Response {i}",
                tool_calls=[],
                token_count=50,
                cost=0.001,
            )

        # Verify cumulative stats
        metadata = get_conversation_metadata(conv_id)
        assert metadata["turn_count"] == 3
        assert metadata["total_tokens"] == 150
        assert metadata["total_cost"] == 0.003

    def test_format_conversation_for_display(self, tmp_path, monkeypatch):
        """Test formatting conversation for display."""
        turns = [
            {
                "type": "metadata",
                "id": "test_conv",
                "agent": "chat",
                "model": "gpt-4o",
                "machine": "laptop",
                "created_at": "2025-10-24T10:00:00+00:00",
            },
            {
                "type": "turn",
                "timestamp": "2025-10-24T10:00:05+00:00",
                "user": "Hello",
                "assistant": "Hi!",
                "tools": ["search"],
                "tokens": 50,
                "cost": 0.001,
            },
        ]

        output = format_conversation_for_display(turns)

        assert "test_conv" in output
        assert "chat" in output
        assert "Hello" in output
        assert "Hi!" in output
        assert "search" in output

    def test_get_machine_name(self, monkeypatch):
        """Test getting machine name from config or hostname."""
        # Test auto-detection
        machine_name = get_machine_name()
        assert machine_name is not None
        assert len(machine_name) > 0

        # Test config override
        from tsugite.config import Config

        def mock_load_config():
            config = Config()
            config.machine_name = "custom_machine"
            return config

        monkeypatch.setattr("tsugite.ui.chat_history.load_config", mock_load_config)

        machine_name = get_machine_name()
        assert machine_name == "custom_machine"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_conversation(self, tmp_path, monkeypatch):
        """Test handling empty conversation file."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)

        conv_id = "empty_conv"
        conv_file = tmp_path / f"{conv_id}.jsonl"
        conv_file.write_text("")

        turns = load_conversation(conv_id)
        assert len(turns) == 0

    def test_prune_with_no_criteria(self, tmp_path, monkeypatch):
        """Test pruning without specifying criteria raises error."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)

        with pytest.raises(ValueError, match="Must specify"):
            prune_conversations()

    def test_index_corruption_handling(self, tmp_path, monkeypatch):
        """Test handling of corrupted index file."""
        monkeypatch.setattr("tsugite.history.index.get_history_dir", lambda: tmp_path)

        # Create corrupted index
        index_file = tmp_path / "index.json"
        index_file.write_text("not valid json {")

        # Should return empty dict instead of crashing
        from tsugite.history.index import load_index

        index = load_index()
        assert index == {}
