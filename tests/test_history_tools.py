"""Tests for history tools."""

from datetime import datetime
from unittest.mock import patch

import pytest

from tsugite.history.models import ConversationMetadata, Turn
from tsugite.tools import call_tool


@pytest.fixture
def history_tools(reset_tool_registry):
    """Register history tools for testing."""
    from tsugite.tools import tool
    from tsugite.tools.history import list_conversations, read_conversation

    # Re-register the tools after registry reset
    tool(read_conversation)
    tool(list_conversations)


@pytest.fixture
def mock_conversation():
    """Create mock conversation data."""
    metadata = ConversationMetadata(
        id="20251110_120000_chat_abc123",
        agent="chat-assistant",
        model="anthropic:claude-sonnet-4.5",
        machine="test-machine",
        created_at=datetime(2025, 11, 10, 12, 0, 0),
    )

    turn1 = Turn(
        timestamp=datetime(2025, 11, 10, 12, 0, 5),
        user="What is the capital of France?",
        assistant="The capital of France is Paris.",
        tools=[],
        tokens=50,
        cost=0.001,
        steps=None,
        messages=None,
    )

    turn2 = Turn(
        timestamp=datetime(2025, 11, 10, 12, 0, 15),
        user="Read the file config.json",
        assistant="Here is the config.json content: {...}",
        tools=["read_file"],
        tokens=150,
        cost=0.003,
        steps=[
            {"type": "thought", "content": "I'll read the config file"},
            {"type": "code", "content": "read_file('config.json')"},
            {"type": "observation", "content": "File content: {...}"},
        ],
        messages=None,
    )

    return [metadata, turn1, turn2]


@pytest.fixture
def mock_index_results():
    """Create mock index query results."""
    return [
        {
            "conversation_id": "20251110_120000_chat_abc123",
            "agent": "chat-assistant",
            "model": "anthropic:claude-sonnet-4.5",
            "machine": "test-machine",
            "created_at": "2025-11-10T12:00:00",
            "updated_at": "2025-11-10T12:00:15",
            "turn_count": 2,
            "total_tokens": 200,
            "total_cost": 0.004,
        },
        {
            "conversation_id": "20251110_110000_default_def456",
            "agent": "default",
            "model": "openai:gpt-4o-mini",
            "machine": "test-machine",
            "created_at": "2025-11-10T11:00:00",
            "updated_at": "2025-11-10T11:05:00",
            "turn_count": 5,
            "total_tokens": 1000,
            "total_cost": 0.02,
        },
    ]


def test_read_conversation_success(history_tools, mock_conversation):
    """Test reading a conversation successfully."""
    with patch("tsugite.tools.history.load_conversation", return_value=mock_conversation):
        result = call_tool("read_conversation", conversation_id="20251110_120000_chat_abc123")

        # Verify structure
        assert "metadata" in result
        assert "turns" in result
        assert "summary" in result

        # Verify metadata
        metadata = result["metadata"]
        assert metadata["conversation_id"] == "20251110_120000_chat_abc123"
        assert metadata["agent"] == "chat-assistant"
        assert metadata["model"] == "anthropic:claude-sonnet-4.5"
        assert metadata["machine"] == "test-machine"

        # Verify turns
        turns = result["turns"]
        assert len(turns) == 2
        assert turns[0]["user"] == "What is the capital of France?"
        assert turns[0]["assistant"] == "The capital of France is Paris."
        assert turns[0]["tools"] == []
        assert turns[0]["tokens"] == 50

        assert turns[1]["user"] == "Read the file config.json"
        assert turns[1]["tools"] == ["read_file"]
        assert turns[1]["tokens"] == 150
        assert "steps" in turns[1]
        assert len(turns[1]["steps"]) == 3

        # Verify summary
        summary = result["summary"]
        assert summary["turn_count"] == 2
        assert summary["total_tokens"] == 200
        assert summary["total_cost"] == 0.004
        assert summary["tools_used"] == ["read_file"]


def test_read_conversation_not_found(history_tools):
    """Test reading a nonexistent conversation."""
    with patch(
        "tsugite.tools.history.load_conversation",
        side_effect=FileNotFoundError("Conversation not found"),
    ):
        with pytest.raises(RuntimeError, match="Tool 'read_conversation' failed"):
            call_tool("read_conversation", conversation_id="nonexistent_id")


def test_read_conversation_empty(history_tools):
    """Test reading an empty conversation."""
    with patch("tsugite.tools.history.load_conversation", return_value=[]):
        with pytest.raises(RuntimeError, match="Tool 'read_conversation' failed"):
            call_tool("read_conversation", conversation_id="empty_id")


def test_list_conversations_basic(history_tools, mock_index_results):
    """Test listing conversations without filters."""
    with patch("tsugite.tools.history.query_index", return_value=mock_index_results):
        result = call_tool("list_conversations")

        assert isinstance(result, list)
        assert len(result) == 2

        # Verify first conversation
        conv1 = result[0]
        assert conv1["conversation_id"] == "20251110_120000_chat_abc123"
        assert conv1["agent"] == "chat-assistant"
        assert conv1["model"] == "anthropic:claude-sonnet-4.5"
        assert conv1["turn_count"] == 2
        assert conv1["total_tokens"] == 200
        assert conv1["total_cost"] == 0.004

        # Verify second conversation
        conv2 = result[1]
        assert conv2["conversation_id"] == "20251110_110000_default_def456"
        assert conv2["agent"] == "default"
        assert conv2["turn_count"] == 5


def test_list_conversations_with_filters(history_tools, mock_index_results):
    """Test listing conversations with filters."""
    filtered_results = [mock_index_results[0]]  # Only chat-assistant

    with patch("tsugite.tools.history.query_index", return_value=filtered_results) as mock_query:
        result = call_tool("list_conversations", agent="chat-assistant", limit=5)

        # Verify query was called with correct parameters
        mock_query.assert_called_once_with(machine=None, agent="chat-assistant", limit=5)

        assert len(result) == 1
        assert result[0]["agent"] == "chat-assistant"


def test_list_conversations_limit_validation(history_tools):
    """Test that list_conversations validates limit parameter."""
    with pytest.raises(RuntimeError, match="Tool 'list_conversations' failed"):
        call_tool("list_conversations", limit=0)

    with pytest.raises(RuntimeError, match="Tool 'list_conversations' failed"):
        call_tool("list_conversations", limit=101)


def test_list_conversations_empty_results(history_tools):
    """Test listing conversations when no results found."""
    with patch("tsugite.tools.history.query_index", return_value=[]):
        result = call_tool("list_conversations")

        assert isinstance(result, list)
        assert len(result) == 0


def test_read_conversation_with_missing_optional_fields(history_tools):
    """Test reading conversation with turns that have no tokens or cost."""
    metadata = ConversationMetadata(
        id="20251110_120000_test_xyz789",
        agent="test-agent",
        model="openai:gpt-4o-mini",
        machine="test-machine",
        created_at=datetime(2025, 11, 10, 12, 0, 0),
    )

    turn = Turn(
        timestamp=datetime(2025, 11, 10, 12, 0, 5),
        user="Hello",
        assistant="Hi there!",
        tools=[],  # Empty list, not None
        tokens=None,
        cost=None,
        steps=None,
        messages=None,
    )

    with patch("tsugite.tools.history.load_conversation", return_value=[metadata, turn]):
        result = call_tool("read_conversation", conversation_id="20251110_120000_test_xyz789")

        # Verify summary handles None values
        summary = result["summary"]
        assert summary["turn_count"] == 1
        assert summary["total_tokens"] is None
        assert summary["total_cost"] is None
        assert summary["tools_used"] == []

        # Verify turn structure
        assert len(result["turns"]) == 1
        assert result["turns"][0]["tokens"] is None
        assert result["turns"][0]["cost"] is None
        assert result["turns"][0]["tools"] == []


def test_list_conversations_with_machine_filter(history_tools, mock_index_results):
    """Test listing conversations with machine filter."""
    with patch("tsugite.tools.history.query_index", return_value=mock_index_results) as mock_query:
        call_tool("list_conversations", machine="test-machine", limit=10)

        # Verify query was called with machine parameter
        mock_query.assert_called_once_with(machine="test-machine", agent=None, limit=10)


def test_list_conversations_default_limit(history_tools, mock_index_results):
    """Test that list_conversations uses default limit of 10."""
    with patch("tsugite.tools.history.query_index", return_value=mock_index_results) as mock_query:
        call_tool("list_conversations")

        # Verify default limit is 10
        mock_query.assert_called_once_with(machine=None, agent=None, limit=10)
