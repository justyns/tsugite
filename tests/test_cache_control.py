"""Tests for conversation history caching functionality."""

from datetime import datetime, timezone

import pytest

from tsugite.agent_runner.history_integration import apply_cache_control_to_messages
from tsugite.history import Turn, save_turn_to_history
from tsugite.ui.chat_history import start_conversation


@pytest.fixture
def temp_history_dir(tmp_path, monkeypatch):
    """Create isolated history directory for tests."""
    history_dir = tmp_path / "history"
    history_dir.mkdir()

    # Mock all history directory getters
    monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: history_dir)
    monkeypatch.setattr("tsugite.history.index.get_history_dir", lambda: history_dir)

    return history_dir


class TestApplyCacheControlToMessages:
    """Tests for apply_cache_control_to_messages function."""

    def test_empty_messages(self):
        """Test with empty message list."""
        messages = []
        result = apply_cache_control_to_messages(messages)
        assert result == []

    def test_single_turn_caching(self):
        """Test caching a single turn - all messages cached."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = apply_cache_control_to_messages(messages)

        # Both messages should have cache_control
        assert len(result) == 2
        assert result[0]["cache_control"] == {"type": "ephemeral"}
        assert result[1]["cache_control"] == {"type": "ephemeral"}

    def test_multiple_turns_all_cached(self):
        """Test that all turns are cached (industry best practice)."""
        messages = [
            {"role": "user", "content": "Turn 1 user"},
            {"role": "assistant", "content": "Turn 1 assistant"},
            {"role": "user", "content": "Turn 2 user"},
            {"role": "assistant", "content": "Turn 2 assistant"},
            {"role": "user", "content": "Turn 3 user"},
            {"role": "assistant", "content": "Turn 3 assistant"},
        ]
        result = apply_cache_control_to_messages(messages)

        # All messages should have cache_control
        assert len(result) == 6
        for i, msg in enumerate(result):
            assert msg["cache_control"] == {"type": "ephemeral"}, f"Message {i} not cached"

    def test_many_turns_all_cached(self):
        """Test that even many turns are all cached."""
        messages = []
        for i in range(10):
            messages.append({"role": "user", "content": f"Turn {i}"})
            messages.append({"role": "assistant", "content": f"Response {i}"})

        result = apply_cache_control_to_messages(messages)

        # All 20 messages should be cached
        assert len(result) == 20
        for msg in result:
            assert msg["cache_control"] == {"type": "ephemeral"}

    def test_preserves_existing_fields(self):
        """Test that existing message fields are preserved."""
        messages = [
            {"role": "user", "content": "Hello", "name": "Alice"},
            {"role": "assistant", "content": "Hi", "function_call": {"name": "test"}},
        ]
        result = apply_cache_control_to_messages(messages)

        # Check original fields are preserved
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"
        assert result[0]["name"] == "Alice"
        assert result[0]["cache_control"] == {"type": "ephemeral"}

        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Hi"
        assert result[1]["function_call"] == {"name": "test"}
        assert result[1]["cache_control"] == {"type": "ephemeral"}


class TestCacheControlIntegration:
    """Integration tests for cache control with conversation history."""

    def test_load_and_cache_conversation(self, temp_history_dir):
        """Test loading conversation and applying cache control to all messages."""
        from tsugite.agent_runner.history_integration import load_conversation_messages

        # Create conversation with 3 turns
        conv_id = start_conversation("test_agent", "test:model")

        for i in range(3):
            turn = Turn(
                timestamp=datetime.now(timezone.utc),
                user=f"Question {i}",
                assistant=f"Answer {i}",
                tools=[],
                tokens=50,
                cost=0.001,
            )
            save_turn_to_history(conv_id, turn)

        # Load messages
        messages = load_conversation_messages(conv_id)
        assert len(messages) == 6  # 3 turns × 2 messages

        # Apply cache control to all messages (industry best practice)
        cached = apply_cache_control_to_messages(messages)

        # All messages should be cached
        assert len(cached) == 6
        for msg in cached:
            assert msg["cache_control"] == {"type": "ephemeral"}

    def test_long_conversation_all_cached(self, temp_history_dir):
        """Test that even long conversations have all messages cached."""
        from tsugite.agent_runner.history_integration import load_conversation_messages

        # Create conversation with 10 turns
        conv_id = start_conversation("test_agent", "test:model")

        for i in range(10):
            turn = Turn(
                timestamp=datetime.now(timezone.utc),
                user=f"Question {i}",
                assistant=f"Answer {i}",
                tools=[],
                tokens=50,
                cost=0.001,
            )
            save_turn_to_history(conv_id, turn)

        messages = load_conversation_messages(conv_id)
        assert len(messages) == 20  # 10 turns × 2 messages

        # Apply cache control - should cache all messages
        cached = apply_cache_control_to_messages(messages)

        # All 20 messages should be cached
        cached_count = sum(1 for msg in cached if "cache_control" in msg)
        assert cached_count == 20
