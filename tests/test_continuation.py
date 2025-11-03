"""Tests for conversation continuation functionality."""

from datetime import datetime, timezone

import pytest

from tsugite.history import Turn, save_turn_to_history
from tsugite.ui.chat import ChatManager
from tsugite.ui.chat_history import (
    get_latest_conversation,
    load_conversation_history,
    start_conversation,
)


@pytest.fixture
def temp_history_dir(tmp_path, monkeypatch):
    """Create isolated history directory for tests."""
    history_dir = tmp_path / "history"
    history_dir.mkdir()

    # Mock all history directory getters in the history module only
    monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: history_dir)
    monkeypatch.setattr("tsugite.history.index.get_history_dir", lambda: history_dir)

    return history_dir


@pytest.fixture
def temp_agent_file(tmp_path):
    """Create a simple test agent file."""
    agent_file = tmp_path / "test_agent.md"
    agent_content = """---
name: test_agent
extends: none
model: openai:gpt-4o-mini
max_turns: 5
tools: []
---

# Test Agent

Task: {{ user_prompt }}
"""
    agent_file.write_text(agent_content)
    return agent_file


class TestGetLatestConversation:
    """Tests for getting the latest conversation."""

    def test_get_latest_conversation_no_conversations(self, temp_history_dir):
        """Test getting latest conversation when none exist."""
        result = get_latest_conversation()
        assert result is None

    def test_get_latest_conversation_single(self, temp_history_dir):
        """Test getting latest conversation with one conversation."""
        # Create a conversation
        conv_id = start_conversation("test_agent", "test:model")

        # Get latest
        latest = get_latest_conversation()
        assert latest == conv_id

    def test_get_latest_conversation_multiple(self, temp_history_dir):
        """Test getting latest conversation with multiple conversations."""
        # Create multiple conversations with different timestamps
        _conv_id_1 = start_conversation(
            "agent1",
            "model1",
            timestamp=datetime(2025, 10, 24, 10, 0, 0, tzinfo=timezone.utc),
        )
        _conv_id_2 = start_conversation(
            "agent2",
            "model2",
            timestamp=datetime(2025, 10, 24, 11, 0, 0, tzinfo=timezone.utc),
        )
        conv_id_3 = start_conversation(
            "agent3",
            "model3",
            timestamp=datetime(2025, 10, 24, 12, 0, 0, tzinfo=timezone.utc),
        )

        # Latest should be the most recently created
        latest = get_latest_conversation()
        assert latest == conv_id_3


class TestLoadConversationHistory:
    """Tests for loading conversation history."""

    def test_load_conversation_history_empty(self, temp_history_dir):
        """Test loading conversation with no turns."""
        # Create conversation with metadata only
        conv_id = start_conversation("test_agent", "test:model")

        # Load history
        turns = load_conversation_history(conv_id)
        assert turns == []

    def test_load_conversation_history_single_turn(self, temp_history_dir):
        """Test loading conversation with one turn."""
        # Create conversation
        conv_id = start_conversation("test_agent", "test:model")

        # Add a turn
        timestamp = datetime.now(timezone.utc)
        turn = Turn(
            timestamp=timestamp,
            user="Hello",
            assistant="Hi there!",
            tools=["final_answer"],
            tokens=100,
            cost=0.001,
        )
        save_turn_to_history(conv_id, turn)

        # Load history
        turns = load_conversation_history(conv_id)
        assert len(turns) == 1
        assert turns[0].user == "Hello"
        assert turns[0].assistant == "Hi there!"
        assert turns[0].tools == ["final_answer"]
        assert turns[0].tokens == 100
        assert turns[0].cost == 0.001

    def test_load_conversation_history_multiple_turns(self, temp_history_dir):
        """Test loading conversation with multiple turns."""
        # Create conversation
        conv_id = start_conversation("test_agent", "test:model")

        # Add multiple turns
        for i in range(3):
            turn = Turn(
                timestamp=datetime.now(timezone.utc),
                user=f"Question {i}",
                assistant=f"Answer {i}",
                tools=[],
                tokens=50,
                cost=0.0005,
            )
            save_turn_to_history(conv_id, turn)

        # Load history
        turns = load_conversation_history(conv_id)
        assert len(turns) == 3
        assert turns[0].user == "Question 0"
        assert turns[1].user == "Question 1"
        assert turns[2].user == "Question 2"

    def test_load_conversation_history_not_found(self, temp_history_dir):
        """Test loading non-existent conversation."""
        with pytest.raises(FileNotFoundError):
            load_conversation_history("nonexistent_conv_id")


class TestChatManagerLoadFromHistory:
    """Tests for ChatManager.load_from_history()."""

    def test_load_from_history_empty_conversation(self, temp_history_dir, temp_agent_file):
        """Test loading empty conversation into ChatManager."""
        # Create conversation
        conv_id = start_conversation("test_agent", "test:model")

        # Create ChatManager and load history
        manager = ChatManager(temp_agent_file, disable_history=True)
        turns = load_conversation_history(conv_id)
        manager.load_from_history(conv_id, turns)

        # Verify state
        assert manager.conversation_id == conv_id
        assert len(manager.conversation_history) == 0

    def test_load_from_history_with_turns(self, temp_history_dir, temp_agent_file):
        """Test loading conversation with turns into ChatManager."""
        # Create conversation
        conv_id = start_conversation("test_agent", "test:model")

        # Add turns
        timestamps = [
            datetime(2025, 10, 24, 10, 0, 0, tzinfo=timezone.utc),
            datetime(2025, 10, 24, 10, 5, 0, tzinfo=timezone.utc),
        ]

        for i, ts in enumerate(timestamps):
            turn = Turn(
                timestamp=ts,
                user=f"Question {i}",
                assistant=f"Answer {i}",
                tools=["tool1", "tool2"],
                tokens=100,
                cost=0.002,
            )
            save_turn_to_history(conv_id, turn)

        # Create ChatManager and load history
        manager = ChatManager(temp_agent_file, disable_history=True)
        turns = load_conversation_history(conv_id)
        manager.load_from_history(conv_id, turns)

        # Verify state
        assert manager.conversation_id == conv_id
        assert len(manager.conversation_history) == 2

        # Check first turn
        assert manager.conversation_history[0].user_message == "Question 0"
        assert manager.conversation_history[0].agent_response == "Answer 0"
        assert manager.conversation_history[0].tool_calls == ["tool1", "tool2"]
        assert manager.conversation_history[0].token_count == 100
        assert manager.conversation_history[0].cost == 0.002

        # Check session start was updated to first turn's timestamp (timezone stripped to naive)
        assert manager.session_start == timestamps[0].replace(tzinfo=None)

    def test_load_from_history_respects_max_history(self, temp_history_dir, temp_agent_file):
        """Test that load_from_history respects max_history limit."""
        # Create conversation
        conv_id = start_conversation("test_agent", "test:model")

        # Add 10 turns
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

        # Create ChatManager with max_history=5 and load history
        manager = ChatManager(temp_agent_file, max_history=5, disable_history=True)
        turns = load_conversation_history(conv_id)
        manager.load_from_history(conv_id, turns)

        # Should only keep last 5 turns
        assert len(manager.conversation_history) == 5
        assert manager.conversation_history[0].user_message == "Question 5"
        assert manager.conversation_history[4].user_message == "Question 9"

    def test_load_from_history_can_add_new_turns(self, temp_history_dir, temp_agent_file):
        """Test that new turns can be added after loading from history."""
        # Create conversation
        conv_id = start_conversation("test_agent", "test:model")

        # Add initial turn
        turn = Turn(
            timestamp=datetime.now(timezone.utc),
            user="Initial question",
            assistant="Initial answer",
            tools=[],
            tokens=50,
            cost=0.001,
        )
        save_turn_to_history(conv_id, turn)

        # Create ChatManager and load history
        manager = ChatManager(temp_agent_file, resume_conversation_id=conv_id)
        turns = load_conversation_history(conv_id)
        manager.load_from_history(conv_id, turns)

        # Add new turn
        manager.add_turn("New question", "New answer", token_count=60, cost=0.0012)

        # Verify both turns are present
        assert len(manager.conversation_history) == 2
        assert manager.conversation_history[0].user_message == "Initial question"
        assert manager.conversation_history[1].user_message == "New question"


class TestRunModeContinuation:
    """Tests for run mode continuation functionality."""

    def test_save_run_to_history_with_continuation(self, temp_history_dir, temp_agent_file):
        """Test saving run to history with continuation ID."""
        from tsugite.agent_runner.history_integration import save_run_to_history

        # Create initial conversation
        conv_id = start_conversation("test_agent", "test:model")

        # Add initial turn
        turn = Turn(
            timestamp=datetime.now(timezone.utc),
            user="First question",
            assistant="First answer",
            tools=[],
            tokens=50,
            cost=0.001,
        )
        save_turn_to_history(conv_id, turn)

        # Save continuation turn
        result_conv_id = save_run_to_history(
            agent_path=temp_agent_file,
            agent_name="test_agent",
            prompt="Second question",
            result="Second answer",
            model="test:model",
            token_count=60,
            cost=0.0012,
            execution_steps=[],
            continue_conversation_id=conv_id,
        )

        # Should return same conversation ID
        assert result_conv_id == conv_id

        # Load conversation and verify both turns
        turns = load_conversation_history(conv_id)
        assert len(turns) == 2
        assert turns[0].user == "First question"
        assert turns[1].user == "Second question"

    def test_save_run_to_history_new_conversation(self, temp_history_dir, temp_agent_file):
        """Test saving run to history creates new conversation when no continuation ID."""
        from tsugite.agent_runner.history_integration import save_run_to_history

        # Save without continuation ID
        result_conv_id = save_run_to_history(
            agent_path=temp_agent_file,
            agent_name="test_agent",
            prompt="Question",
            result="Answer",
            model="test:model",
            token_count=50,
            cost=0.001,
            execution_steps=[],
            continue_conversation_id=None,
        )

        # Should create new conversation
        assert result_conv_id is not None

        # Load conversation and verify single turn
        turns = load_conversation_history(result_conv_id)
        assert len(turns) == 1
        assert turns[0].user == "Question"


class TestToolCallHistory:
    """Tests for tool call preservation in conversation history."""

    def test_load_messages_with_steps(self, temp_history_dir):
        """Test loading messages with execution steps (tool calls)."""
        from tsugite.agent_runner.history_integration import load_conversation_messages

        conv_id = start_conversation("test_agent", "test:model")

        turn = Turn(
            timestamp=datetime.now(timezone.utc),
            user="Read config.json",
            assistant="The config contains: {...}",
            tools=["read_file"],
            tokens=100,
            cost=0.001,
            steps=[
                {
                    "thought": "I need to read the config file",
                    "code": "read_file('config.json')",
                    "output": '{"key": "value"}',
                },
            ],
        )
        save_turn_to_history(conv_id, turn)

        messages = load_conversation_messages(conv_id)

        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Read config.json"
        assert messages[1]["role"] == "assistant"
        assert "Thought: I need to read the config file" in messages[1]["content"]
        assert "read_file('config.json')" in messages[1]["content"]
        assert messages[2]["role"] == "user"
        assert "Observation:" in messages[2]["content"]
        assert '{"key": "value"}' in messages[2]["content"]
        assert messages[3]["role"] == "assistant"
        assert messages[3]["content"] == "The config contains: {...}"

    def test_load_messages_with_multiple_steps(self, temp_history_dir):
        """Test loading messages with multiple tool calls."""
        from tsugite.agent_runner.history_integration import load_conversation_messages

        conv_id = start_conversation("test_agent", "test:model")

        turn = Turn(
            timestamp=datetime.now(timezone.utc),
            user="Search and write results",
            assistant="Results written to file",
            tools=["web_search", "write_file"],
            tokens=200,
            cost=0.002,
            steps=[
                {
                    "thought": "First search the web",
                    "code": "web_search('python')",
                    "output": "Found 100 results",
                },
                {
                    "thought": "Now write to file",
                    "code": "write_file('results.txt', data)",
                    "output": "File written successfully",
                },
            ],
        )
        save_turn_to_history(conv_id, turn)

        messages = load_conversation_messages(conv_id)

        assert len(messages) == 6
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert "web_search" in messages[1]["content"]
        assert messages[2]["role"] == "user"
        assert "Found 100 results" in messages[2]["content"]
        assert messages[3]["role"] == "assistant"
        assert "write_file" in messages[3]["content"]
        assert messages[4]["role"] == "user"
        assert "File written successfully" in messages[4]["content"]
        assert messages[5]["role"] == "assistant"

    def test_load_messages_with_stored_messages(self, temp_history_dir):
        """Test loading from pre-saved messages field."""
        from tsugite.agent_runner.history_integration import load_conversation_messages

        conv_id = start_conversation("test_agent", "test:model")

        turn = Turn(
            timestamp=datetime.now(timezone.utc),
            user="Test task",
            assistant="Done",
            tools=["read_file"],
            tokens=50,
            cost=0.001,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Test task"},
                {"role": "assistant", "content": "Thought: Reading file\n\n```python\nread_file('test.txt')\n```"},
                {"role": "user", "content": "Observation: file contents"},
                {"role": "assistant", "content": "Done"},
            ],
        )
        save_turn_to_history(conv_id, turn)

        messages = load_conversation_messages(conv_id)

        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert "read_file" in messages[1]["content"]
        assert messages[2]["role"] == "user"
        assert "Observation:" in messages[2]["content"]
        assert messages[3]["role"] == "assistant"

    def test_load_messages_backward_compat(self, temp_history_dir):
        """Test loading old format without steps or messages."""
        from tsugite.agent_runner.history_integration import load_conversation_messages

        conv_id = start_conversation("test_agent", "test:model")

        turn = Turn(
            timestamp=datetime.now(timezone.utc),
            user="Simple question",
            assistant="Simple answer",
            tools=[],
            tokens=25,
            cost=0.0005,
        )
        save_turn_to_history(conv_id, turn)

        messages = load_conversation_messages(conv_id)

        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Simple question"}
        assert messages[1] == {"role": "assistant", "content": "Simple answer"}

    def test_cache_control_with_tool_calls(self, temp_history_dir):
        """Test that cache control is applied to all messages including tool calls."""
        from tsugite.agent_runner.history_integration import (
            apply_cache_control_to_messages,
            load_conversation_messages,
        )

        conv_id = start_conversation("test_agent", "test:model")

        turn = Turn(
            timestamp=datetime.now(timezone.utc),
            user="Task with tools",
            assistant="Completed",
            tools=["read_file"],
            tokens=100,
            cost=0.001,
            steps=[
                {
                    "thought": "Reading file",
                    "code": "read_file('test.txt')",
                    "output": "contents",
                },
            ],
        )
        save_turn_to_history(conv_id, turn)

        messages = load_conversation_messages(conv_id)
        cached = apply_cache_control_to_messages(messages)

        assert len(cached) == 4
        for msg in cached:
            assert msg["cache_control"] == {"type": "ephemeral"}
