"""Tests for conversation continuation functionality."""

from datetime import datetime, timezone

import pytest

from tsugite.history import SessionStorage, Turn
from tsugite.ui.chat import ChatManager


@pytest.fixture
def temp_history_dir(tmp_path, monkeypatch):
    """Create isolated history directory for tests."""
    history_dir = tmp_path / "history"
    history_dir.mkdir()

    monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: history_dir)
    monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test_machine")

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
        from tsugite.history import list_session_files

        files = list_session_files()
        assert len(files) == 0

    def test_get_latest_conversation_single(self, temp_history_dir):
        """Test getting latest conversation with one conversation."""
        storage = SessionStorage.create(agent_name="test_agent", model="test:model")

        from tsugite.history import list_session_files

        files = list_session_files()
        assert len(files) == 1
        assert storage.session_id in str(files[0])

    def test_get_latest_conversation_multiple(self, temp_history_dir):
        """Test getting latest conversation with multiple conversations."""
        storage1 = SessionStorage.create(
            agent_name="agent1",
            model="model1",
            timestamp=datetime(2025, 10, 24, 10, 0, 0, tzinfo=timezone.utc),
        )
        storage2 = SessionStorage.create(
            agent_name="agent2",
            model="model2",
            timestamp=datetime(2025, 10, 24, 11, 0, 0, tzinfo=timezone.utc),
        )
        storage3 = SessionStorage.create(
            agent_name="agent3",
            model="model3",
            timestamp=datetime(2025, 10, 24, 12, 0, 0, tzinfo=timezone.utc),
        )

        from tsugite.history import list_session_files

        files = list_session_files()
        assert len(files) == 3
        # Files are sorted newest first
        assert storage3.session_id in str(files[0])


class TestLoadConversationHistory:
    """Tests for loading conversation history."""

    def test_load_conversation_history_empty(self, temp_history_dir):
        """Test loading conversation with no turns."""
        storage = SessionStorage.create(agent_name="test_agent", model="test:model")

        records = storage.load_records()
        turns = [r for r in records if isinstance(r, Turn)]
        assert turns == []

    def test_load_conversation_history_single_turn(self, temp_history_dir):
        """Test loading conversation with one turn."""
        storage = SessionStorage.create(agent_name="test_agent", model="test:model")

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        storage.record_turn(
            messages=messages,
            final_answer="Hi there!",
            tokens=100,
            cost=0.001,
            functions_called=["final_answer"],
        )

        records = storage.load_records()
        turns = [r for r in records if isinstance(r, Turn)]
        assert len(turns) == 1
        assert turns[0].final_answer == "Hi there!"
        assert turns[0].tokens == 100
        assert turns[0].cost == 0.001

    def test_load_conversation_history_multiple_turns(self, temp_history_dir):
        """Test loading conversation with multiple turns."""
        storage = SessionStorage.create(agent_name="test_agent", model="test:model")

        for i in range(3):
            messages = [
                {"role": "user", "content": f"Question {i}"},
                {"role": "assistant", "content": f"Answer {i}"},
            ]
            storage.record_turn(
                messages=messages,
                final_answer=f"Answer {i}",
                tokens=50,
                cost=0.0005,
            )

        records = storage.load_records()
        turns = [r for r in records if isinstance(r, Turn)]
        assert len(turns) == 3
        assert turns[0].final_answer == "Answer 0"
        assert turns[1].final_answer == "Answer 1"
        assert turns[2].final_answer == "Answer 2"

    def test_load_conversation_history_not_found(self, temp_history_dir):
        """Test loading non-existent conversation."""
        with pytest.raises(FileNotFoundError):
            SessionStorage.load(temp_history_dir / "nonexistent.jsonl")


class TestChatManagerLoadFromHistory:
    """Tests for ChatManager.load_from_history()."""

    def test_load_from_history_empty_conversation(self, temp_history_dir, temp_agent_file):
        """Test loading empty conversation into ChatManager."""
        storage = SessionStorage.create(agent_name="test_agent", model="test:model")

        manager = ChatManager(temp_agent_file, disable_history=True)
        records = storage.load_records()
        turns = [r for r in records if isinstance(r, Turn)]
        manager.load_from_history(storage.session_id, turns)

        assert manager.conversation_id == storage.session_id
        assert len(manager.conversation_history) == 0

    def test_load_from_history_with_turns(self, temp_history_dir, temp_agent_file):
        """Test loading conversation with turns into ChatManager."""
        storage = SessionStorage.create(agent_name="test_agent", model="test:model")

        timestamps = [
            datetime(2025, 10, 24, 10, 0, 0, tzinfo=timezone.utc),
            datetime(2025, 10, 24, 10, 5, 0, tzinfo=timezone.utc),
        ]

        for i, ts in enumerate(timestamps):
            messages = [
                {"role": "user", "content": f"Question {i}"},
                {"role": "assistant", "content": f"Answer {i}"},
            ]
            storage.record_turn(
                messages=messages,
                final_answer=f"Answer {i}",
                tokens=100,
                cost=0.002,
                functions_called=["tool1", "tool2"],
            )

        manager = ChatManager(temp_agent_file, disable_history=True)
        records = storage.load_records()
        turns = [r for r in records if isinstance(r, Turn)]
        manager.load_from_history(storage.session_id, turns)

        assert manager.conversation_id == storage.session_id
        assert len(manager.conversation_history) == 2

        # Check first turn - note that V2 stores user_summary not user_message directly
        assert manager.conversation_history[0].agent_response == "Answer 0"
        assert manager.conversation_history[0].tool_calls == ["tool1", "tool2"]
        assert manager.conversation_history[0].token_count == 100
        assert manager.conversation_history[0].cost == 0.002

    def test_load_from_history_respects_max_history(self, temp_history_dir, temp_agent_file):
        """Test that load_from_history respects max_history limit."""
        storage = SessionStorage.create(agent_name="test_agent", model="test:model")

        for i in range(10):
            messages = [
                {"role": "user", "content": f"Question {i}"},
                {"role": "assistant", "content": f"Answer {i}"},
            ]
            storage.record_turn(
                messages=messages,
                final_answer=f"Answer {i}",
                tokens=50,
                cost=0.001,
            )

        manager = ChatManager(temp_agent_file, max_history=5, disable_history=True)
        records = storage.load_records()
        turns = [r for r in records if isinstance(r, Turn)]
        manager.load_from_history(storage.session_id, turns)

        # Should only keep last 5 turns
        assert len(manager.conversation_history) == 5
        assert manager.conversation_history[0].agent_response == "Answer 5"
        assert manager.conversation_history[4].agent_response == "Answer 9"

    def test_load_from_history_can_add_new_turns(self, temp_history_dir, temp_agent_file, monkeypatch):
        """Test that new turns can be added after loading from history."""
        # Also mock for history import in chat module
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: temp_history_dir)

        storage = SessionStorage.create(agent_name="test_agent", model="test:model")

        messages = [
            {"role": "user", "content": "Initial question"},
            {"role": "assistant", "content": "Initial answer"},
        ]
        storage.record_turn(
            messages=messages,
            final_answer="Initial answer",
            tokens=50,
            cost=0.001,
        )

        manager = ChatManager(temp_agent_file, resume_conversation_id=storage.session_id, disable_history=True)
        records = storage.load_records()
        turns = [r for r in records if isinstance(r, Turn)]
        manager.load_from_history(storage.session_id, turns)

        # Add new turn
        manager.add_turn("New question", "New answer", token_count=60, cost=0.0012)

        # Verify both turns are present
        assert len(manager.conversation_history) == 2
        assert manager.conversation_history[1].user_message == "New question"


class TestRunModeContinuation:
    """Tests for run mode continuation functionality."""

    def test_save_run_to_history_with_continuation(self, temp_history_dir, temp_agent_file, monkeypatch):
        """Test saving run to history with continuation ID."""
        from tsugite.agent_runner.history_integration import save_run_to_history

        # Must also mock in the history_integration module
        monkeypatch.setattr("tsugite.agent_runner.history_integration.get_history_dir", lambda: temp_history_dir)
        monkeypatch.setattr("tsugite.config.load_config", lambda: type("Config", (), {"history_enabled": True})())
        monkeypatch.setattr(
            "tsugite.md_agents.parse_agent_file",
            lambda p: type("Agent", (), {"config": type("Config", (), {"disable_history": False})()})(),
        )

        # Create initial conversation
        storage = SessionStorage.create(agent_name="test_agent", model="test:model")
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
        ]
        storage.record_turn(
            messages=messages,
            final_answer="First answer",
            tokens=50,
            cost=0.001,
        )

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
            continue_conversation_id=storage.session_id,
        )

        # Should return same conversation ID
        assert result_conv_id == storage.session_id

        # Load conversation and verify both turns
        reloaded = SessionStorage.load(storage.session_path)
        records = reloaded.load_records()
        turns = [r for r in records if isinstance(r, Turn)]
        assert len(turns) == 2

    def test_save_run_to_history_new_conversation(self, temp_history_dir, temp_agent_file, monkeypatch):
        """Test saving run to history creates new conversation when no continuation ID."""
        from tsugite.agent_runner.history_integration import save_run_to_history

        # Must also mock in the history_integration module
        monkeypatch.setattr("tsugite.agent_runner.history_integration.get_history_dir", lambda: temp_history_dir)
        monkeypatch.setattr("tsugite.config.load_config", lambda: type("Config", (), {"history_enabled": True})())
        monkeypatch.setattr(
            "tsugite.md_agents.parse_agent_file",
            lambda p: type("Agent", (), {"config": type("Config", (), {"disable_history": False})()})(),
        )

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

        assert result_conv_id is not None

        # Load conversation and verify single turn
        session_path = temp_history_dir / f"{result_conv_id}.jsonl"
        storage = SessionStorage.load(session_path)
        records = storage.load_records()
        turns = [r for r in records if isinstance(r, Turn)]
        assert len(turns) == 1


class TestToolCallHistory:
    """Tests for tool call preservation in conversation history."""

    def test_load_messages_with_steps(self, temp_history_dir):
        """Test loading messages with execution steps (tool calls)."""
        from tsugite.history import reconstruct_messages

        storage = SessionStorage.create(agent_name="test_agent", model="test:model")

        # V2 stores full messages directly
        messages = [
            {"role": "user", "content": "Read config.json"},
            {"role": "assistant", "content": "```python\nread_file('config.json')\n```"},
            {
                "role": "user",
                "content": '<tsugite_execution_result status="success">\n<output>{"key": "value"}</output>\n</tsugite_execution_result>',
            },
            {"role": "assistant", "content": "The config contains: {...}"},
        ]
        storage.record_turn(
            messages=messages,
            final_answer="The config contains: {...}",
            tokens=100,
            cost=0.001,
            functions_called=["read_file"],
        )

        reconstructed = reconstruct_messages(storage.session_path)

        assert len(reconstructed) == 4
        assert reconstructed[0]["role"] == "user"
        assert reconstructed[0]["content"] == "Read config.json"
        assert reconstructed[1]["role"] == "assistant"
        assert "read_file('config.json')" in reconstructed[1]["content"]
        assert reconstructed[2]["role"] == "user"
        assert "<tsugite_execution_result" in reconstructed[2]["content"]
        assert reconstructed[3]["role"] == "assistant"
        assert reconstructed[3]["content"] == "The config contains: {...}"

    def test_load_messages_with_multiple_steps(self, temp_history_dir):
        """Test loading messages with multiple tool calls."""
        from tsugite.history import reconstruct_messages

        storage = SessionStorage.create(agent_name="test_agent", model="test:model")

        messages = [
            {"role": "user", "content": "Search and write results"},
            {"role": "assistant", "content": "```python\nweb_search('python')\n```"},
            {
                "role": "user",
                "content": '<tsugite_execution_result status="success">\n<output>Found 100 results</output>\n</tsugite_execution_result>',
            },
            {"role": "assistant", "content": "```python\nwrite_file('results.txt', data)\n```"},
            {
                "role": "user",
                "content": '<tsugite_execution_result status="success">\n<output>File written successfully</output>\n</tsugite_execution_result>',
            },
            {"role": "assistant", "content": "Results written to file"},
        ]
        storage.record_turn(
            messages=messages,
            final_answer="Results written to file",
            tokens=200,
            cost=0.002,
            functions_called=["web_search", "write_file"],
        )

        reconstructed = reconstruct_messages(storage.session_path)

        assert len(reconstructed) == 6
        assert reconstructed[0]["role"] == "user"
        assert reconstructed[1]["role"] == "assistant"
        assert "web_search" in reconstructed[1]["content"]
        assert reconstructed[2]["role"] == "user"
        assert "Found 100 results" in reconstructed[2]["content"]
        assert reconstructed[3]["role"] == "assistant"
        assert "write_file" in reconstructed[3]["content"]
        assert reconstructed[4]["role"] == "user"
        assert "File written successfully" in reconstructed[4]["content"]
        assert reconstructed[5]["role"] == "assistant"

    def test_cache_control_with_tool_calls(self, temp_history_dir):
        """Test that cache control is applied to all messages including tool calls."""
        from tsugite.history.reconstruction import apply_cache_control_to_messages

        storage = SessionStorage.create(agent_name="test_agent", model="test:model")

        messages = [
            {"role": "user", "content": "Task with tools"},
            {"role": "assistant", "content": "```python\nread_file('test.txt')\n```"},
            {"role": "user", "content": "<tsugite_execution_result>contents</tsugite_execution_result>"},
            {"role": "assistant", "content": "Completed"},
        ]
        storage.record_turn(
            messages=messages,
            final_answer="Completed",
            tokens=100,
            cost=0.001,
            functions_called=["read_file"],
        )

        from tsugite.history import reconstruct_messages

        reconstructed = reconstruct_messages(storage.session_path)
        cached = apply_cache_control_to_messages(reconstructed)

        assert len(cached) == 4
        for msg in cached:
            assert msg["cache_control"] == {"type": "ephemeral"}
