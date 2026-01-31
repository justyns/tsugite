"""Tests for history tools (V2 format)."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tsugite.history import SessionStorage, Turn
from tsugite.tools import call_tool


@pytest.fixture
def history_tools(reset_tool_registry):
    """Register history tools for testing."""
    from tsugite.tools import tool
    from tsugite.tools.history import list_conversations, read_conversation

    tool(read_conversation)
    tool(list_conversations)


@pytest.fixture
def mock_storage():
    """Create a mock SessionStorage with V2 Turn records."""
    storage = MagicMock(spec=SessionStorage)
    storage.session_id = "20251110_120000_chat_abc123"
    storage.agent = "chat-assistant"
    storage.model = "anthropic:claude-sonnet-4.5"
    storage.machine = "test-machine"
    storage.created_at = datetime(2025, 11, 10, 12, 0, 0, tzinfo=timezone.utc)
    storage.turn_count = 2
    storage.total_tokens = 200
    storage.total_cost = 0.004

    turn1 = Turn(
        timestamp=datetime(2025, 11, 10, 12, 0, 5, tzinfo=timezone.utc),
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ],
        final_answer="The capital of France is Paris.",
        user_summary="What is the capital of France?",
        functions_called=[],
        tokens=50,
        cost=0.001,
    )

    turn2 = Turn(
        timestamp=datetime(2025, 11, 10, 12, 0, 15, tzinfo=timezone.utc),
        messages=[
            {"role": "user", "content": "Read the file config.json"},
            {"role": "assistant", "content": "```python\nread_file('config.json')\n```"},
            {"role": "user", "content": "<tsugite_execution_result>File content: {...}</tsugite_execution_result>"},
            {"role": "assistant", "content": "Here is the config.json content: {...}"},
        ],
        final_answer="Here is the config.json content: {...}",
        user_summary="Read the file config.json",
        functions_called=["read_file"],
        tokens=150,
        cost=0.003,
    )

    storage.load_records.return_value = [turn1, turn2]
    return storage


@pytest.fixture
def mock_storage_list():
    """Create a list of mock SessionStorages for list_conversations."""
    storages = []

    storage1 = MagicMock(spec=SessionStorage)
    storage1.session_id = "20251110_120000_chat_abc123"
    storage1.agent = "chat-assistant"
    storage1.model = "anthropic:claude-sonnet-4.5"
    storage1.machine = "test-machine"
    storage1.created_at = datetime(2025, 11, 10, 12, 0, 0, tzinfo=timezone.utc)
    storage1.turn_count = 2
    storage1.total_tokens = 200
    storage1.total_cost = 0.004
    storages.append(storage1)

    storage2 = MagicMock(spec=SessionStorage)
    storage2.session_id = "20251110_110000_default_def456"
    storage2.agent = "default"
    storage2.model = "openai:gpt-4o-mini"
    storage2.machine = "test-machine"
    storage2.created_at = datetime(2025, 11, 10, 11, 0, 0, tzinfo=timezone.utc)
    storage2.turn_count = 5
    storage2.total_tokens = 1000
    storage2.total_cost = 0.02
    storages.append(storage2)

    return storages


def test_read_conversation_success(history_tools, mock_storage):
    """Test reading a conversation successfully."""
    with patch("tsugite.tools.history.SessionStorage.load", return_value=mock_storage):
        result = call_tool("read_conversation", conversation_id="20251110_120000_chat_abc123")

        assert "metadata" in result
        assert "turns" in result
        assert "summary" in result

        metadata = result["metadata"]
        assert metadata["conversation_id"] == "20251110_120000_chat_abc123"
        assert metadata["agent"] == "chat-assistant"
        assert metadata["model"] == "anthropic:claude-sonnet-4.5"
        assert metadata["machine"] == "test-machine"

        turns = result["turns"]
        assert len(turns) == 2
        assert turns[0]["user_summary"] == "What is the capital of France?"
        assert turns[0]["final_answer"] == "The capital of France is Paris."
        assert turns[0]["functions_called"] == []
        assert turns[0]["tokens"] == 50

        assert turns[1]["user_summary"] == "Read the file config.json"
        assert turns[1]["functions_called"] == ["read_file"]
        assert turns[1]["tokens"] == 150

        summary = result["summary"]
        assert summary["turn_count"] == 2
        assert summary["total_tokens"] == 200
        assert summary["total_cost"] == 0.004
        assert summary["functions_used"] == ["read_file"]


def test_read_conversation_not_found(history_tools):
    """Test reading a nonexistent conversation."""
    with patch(
        "tsugite.tools.history.SessionStorage.load",
        side_effect=FileNotFoundError("Session not found"),
    ):
        with pytest.raises(RuntimeError, match="Tool 'read_conversation' failed"):
            call_tool("read_conversation", conversation_id="nonexistent_id")


def test_read_conversation_empty(history_tools):
    """Test reading an empty conversation."""
    empty_storage = MagicMock(spec=SessionStorage)
    empty_storage.load_records.return_value = []

    with patch("tsugite.tools.history.SessionStorage.load", return_value=empty_storage):
        with pytest.raises(RuntimeError, match="Tool 'read_conversation' failed"):
            call_tool("read_conversation", conversation_id="empty_id")


def test_list_conversations_basic(history_tools, mock_storage_list):
    """Test listing conversations without filters."""
    session_files = [Path(f"/fake/history/{s.session_id}.jsonl") for s in mock_storage_list]

    with patch("tsugite.tools.history.list_session_files", return_value=session_files):
        with patch("tsugite.tools.history.SessionStorage.load", side_effect=mock_storage_list):
            result = call_tool("list_conversations")

            assert isinstance(result, list)
            assert len(result) == 2

            conv1 = result[0]
            assert conv1["conversation_id"] == "20251110_120000_chat_abc123"
            assert conv1["agent"] == "chat-assistant"
            assert conv1["model"] == "anthropic:claude-sonnet-4.5"
            assert conv1["turn_count"] == 2
            assert conv1["total_tokens"] == 200
            assert conv1["total_cost"] == 0.004

            conv2 = result[1]
            assert conv2["conversation_id"] == "20251110_110000_default_def456"
            assert conv2["agent"] == "default"
            assert conv2["turn_count"] == 5


def test_list_conversations_with_agent_filter(history_tools, mock_storage_list):
    """Test listing conversations with agent filter."""
    session_files = [Path(f"/fake/history/{s.session_id}.jsonl") for s in mock_storage_list]

    with patch("tsugite.tools.history.list_session_files", return_value=session_files):
        with patch("tsugite.tools.history.SessionStorage.load", side_effect=mock_storage_list):
            result = call_tool("list_conversations", agent="chat-assistant")

            assert len(result) == 1
            assert result[0]["agent"] == "chat-assistant"


def test_list_conversations_with_machine_filter(history_tools, mock_storage_list):
    """Test listing conversations with machine filter."""
    session_files = [Path(f"/fake/history/{s.session_id}.jsonl") for s in mock_storage_list]

    with patch("tsugite.tools.history.list_session_files", return_value=session_files):
        with patch("tsugite.tools.history.SessionStorage.load", side_effect=mock_storage_list):
            result = call_tool("list_conversations", machine="test-machine")

            assert len(result) == 2


def test_list_conversations_limit_validation(history_tools):
    """Test that list_conversations validates limit parameter."""
    with pytest.raises(RuntimeError, match="Tool 'list_conversations' failed"):
        call_tool("list_conversations", limit=0)

    with pytest.raises(RuntimeError, match="Tool 'list_conversations' failed"):
        call_tool("list_conversations", limit=101)


def test_list_conversations_empty_results(history_tools):
    """Test listing conversations when no results found."""
    with patch("tsugite.tools.history.list_session_files", return_value=[]):
        result = call_tool("list_conversations")

        assert isinstance(result, list)
        assert len(result) == 0


def test_read_conversation_with_missing_optional_fields(history_tools):
    """Test reading conversation with turns that have no tokens or cost."""
    storage = MagicMock(spec=SessionStorage)
    storage.session_id = "20251110_120000_test_xyz789"
    storage.agent = "test-agent"
    storage.model = "openai:gpt-4o-mini"
    storage.machine = "test-machine"
    storage.created_at = datetime(2025, 11, 10, 12, 0, 0, tzinfo=timezone.utc)

    turn = Turn(
        timestamp=datetime(2025, 11, 10, 12, 0, 5, tzinfo=timezone.utc),
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        final_answer="Hi there!",
        user_summary="Hello",
        functions_called=[],
        tokens=None,
        cost=None,
    )

    storage.load_records.return_value = [turn]

    with patch("tsugite.tools.history.SessionStorage.load", return_value=storage):
        result = call_tool("read_conversation", conversation_id="20251110_120000_test_xyz789")

        summary = result["summary"]
        assert summary["turn_count"] == 1
        assert summary["total_tokens"] is None
        assert summary["total_cost"] is None
        assert summary["functions_used"] == []

        assert len(result["turns"]) == 1
        assert result["turns"][0]["tokens"] is None
        assert result["turns"][0]["cost"] is None
        assert result["turns"][0]["functions_called"] == []


def test_list_conversations_default_limit(history_tools, mock_storage_list):
    """Test that list_conversations respects limit."""
    # Create more storages than the limit
    many_storages = mock_storage_list * 10  # 20 storages
    session_files = [Path(f"/fake/history/session_{i}.jsonl") for i in range(20)]

    with patch("tsugite.tools.history.list_session_files", return_value=session_files):
        with patch("tsugite.tools.history.SessionStorage.load", side_effect=many_storages):
            result = call_tool("list_conversations", limit=5)

            # Should only return 5 results
            assert len(result) == 5
