"""Tests for Pydantic history models."""

import json
from datetime import datetime, timezone

from tsugite.history.models import ConversationMetadata, IndexEntry, Turn


class TestConversationMetadata:
    """Tests for ConversationMetadata model."""

    def test_valid_metadata(self):
        """Test creating valid metadata."""
        now = datetime.now(timezone.utc)
        metadata = ConversationMetadata(
            id="20251024_103000_chat_abc123",
            agent="chat_assistant",
            model="openai:gpt-4o",
            machine="laptop",
            created_at=now,
        )

        assert metadata.id == "20251024_103000_chat_abc123"
        assert metadata.agent == "chat_assistant"
        assert metadata.model == "openai:gpt-4o"
        assert metadata.machine == "laptop"
        assert metadata.created_at == now

    def test_metadata_from_dict(self):
        """Test creating metadata from dict (model_validate)."""
        now = datetime.now(timezone.utc)
        data = {
            "id": "test_id",
            "agent": "test_agent",
            "model": "test:model",
            "machine": "test_machine",
            "created_at": now.isoformat(),
        }

        metadata = ConversationMetadata.model_validate(data)
        assert metadata.id == "test_id"
        assert metadata.agent == "test_agent"
        assert isinstance(metadata.created_at, datetime)

    def test_metadata_to_dict(self):
        """Test serializing metadata to dict."""
        now = datetime.now(timezone.utc)
        metadata = ConversationMetadata(
            id="test_id",
            agent="test_agent",
            model="test:model",
            machine="test_machine",
            created_at=now,
        )

        data = metadata.model_dump(mode="json")
        assert data["id"] == "test_id"
        assert data["agent"] == "test_agent"
        assert isinstance(data["created_at"], str)  # ISO format string

    def test_metadata_json_round_trip(self):
        """Test JSON serialization round-trip."""
        now = datetime.now(timezone.utc)
        original = ConversationMetadata(
            id="test_id",
            agent="test_agent",
            model="test:model",
            machine="test_machine",
            created_at=now,
        )

        # Serialize to JSON
        json_str = original.model_dump_json()

        # Deserialize back
        data = json.loads(json_str)
        restored = ConversationMetadata.model_validate(data)

        assert restored.id == original.id
        assert restored.agent == original.agent
        assert restored.model == original.model

    def test_metadata_whitespace_stripping(self):
        """Test that whitespace is stripped from strings."""
        data = {
            "id": "  test_id  ",
            "agent": "  test_agent  ",
            "model": "  test:model  ",
            "machine": "  test_machine  ",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        metadata = ConversationMetadata.model_validate(data)
        assert metadata.id == "test_id"  # No leading/trailing spaces
        assert metadata.agent == "test_agent"


class TestTurn:
    """Tests for Turn model."""

    def test_valid_turn(self):
        """Test creating valid turn."""
        now = datetime.now(timezone.utc)
        turn = Turn(
            timestamp=now,
            user="Hello",
            assistant="Hi there!",
            tools=["web_search"],
            tokens=100,
            cost=0.01,
        )

        assert turn.timestamp == now
        assert turn.user == "Hello"
        assert turn.assistant == "Hi there!"
        assert turn.tools == ["web_search"]
        assert turn.tokens == 100
        assert turn.cost == 0.01

    def test_turn_defaults(self):
        """Test turn with default values."""
        now = datetime.now(timezone.utc)
        turn = Turn(
            timestamp=now,
            user="Hello",
            assistant="Hi",
        )

        assert turn.tools == []  # Default empty list
        assert turn.tokens is None  # Default None
        assert turn.cost is None  # Default None

    def test_turn_from_dict(self):
        """Test creating turn from dict."""
        now = datetime.now(timezone.utc)
        data = {
            "timestamp": now.isoformat(),
            "user": "Hello",
            "assistant": "Hi",
            "tools": ["read_file", "write_file"],
            "tokens": 150,
            "cost": 0.02,
        }

        turn = Turn.model_validate(data)
        assert isinstance(turn.timestamp, datetime)
        assert turn.tools == ["read_file", "write_file"]

    def test_turn_to_dict(self):
        """Test serializing turn to dict."""
        now = datetime.now(timezone.utc)
        turn = Turn(
            timestamp=now,
            user="Hello",
            assistant="Hi",
            tools=["web_search"],
            tokens=100,
        )

        data = turn.model_dump(mode="json", exclude_none=True)
        assert data["user"] == "Hello"
        assert isinstance(data["timestamp"], str)
        assert "cost" not in data  # Excluded because it's None

    def test_turn_json_serialization(self):
        """Test JSON serialization."""
        now = datetime.now(timezone.utc)
        turn = Turn(
            timestamp=now,
            user="Hello",
            assistant="Hi",
        )

        json_str = turn.model_dump_json(exclude_none=True)
        assert "Hello" in json_str
        assert "timestamp" in json_str

    def test_turn_with_channel_metadata(self):
        """Test turn with channel routing metadata."""
        now = datetime.now(timezone.utc)
        metadata = {
            "source": "discord",
            "channel_id": "123456789",
            "user_id": "user123",
            "reply_to": "discord:123456789",
            "author_name": "TestUser",
        }

        turn = Turn(
            timestamp=now,
            user="Hello from Discord",
            assistant="Hi there!",
            metadata=metadata,
        )

        assert turn.metadata is not None
        assert turn.metadata["source"] == "discord"
        assert turn.metadata["channel_id"] == "123456789"
        assert turn.metadata["user_id"] == "user123"
        assert turn.metadata["reply_to"] == "discord:123456789"

    def test_turn_metadata_serialization(self):
        """Test metadata field serialization."""
        now = datetime.now(timezone.utc)
        metadata = {
            "source": "cli",
            "user_id": "cli-user",
            "reply_to": "cli",
        }

        turn = Turn(
            timestamp=now,
            user="Hello",
            assistant="Hi",
            metadata=metadata,
        )

        # Serialize
        data = turn.model_dump(mode="json")
        assert "metadata" in data
        assert data["metadata"]["source"] == "cli"

        # Deserialize
        turn2 = Turn.model_validate(data)
        assert turn2.metadata == metadata


class TestIndexEntry:
    """Tests for IndexEntry model."""

    def test_valid_index_entry(self):
        """Test creating valid index entry."""
        now = datetime.now(timezone.utc)
        entry = IndexEntry(
            agent="chat_assistant",
            model="openai:gpt-4o",
            machine="laptop",
            created_at=now,
            updated_at=now,
            turn_count=5,
            total_tokens=1000,
            total_cost=0.05,
        )

        assert entry.agent == "chat_assistant"
        assert entry.turn_count == 5
        assert entry.total_tokens == 1000
        assert entry.total_cost == 0.05

    def test_index_entry_defaults(self):
        """Test index entry with defaults."""
        now = datetime.now(timezone.utc)
        entry = IndexEntry(
            agent="test_agent",
            model="test:model",
            machine="test_machine",
            created_at=now,
            updated_at=now,
        )

        assert entry.turn_count == 0  # Default
        assert entry.total_tokens is None  # Default
        assert entry.total_cost is None  # Default

    def test_index_entry_from_dict(self):
        """Test creating index entry from dict."""
        now = datetime.now(timezone.utc)
        data = {
            "agent": "chat_assistant",
            "model": "openai:gpt-4o",
            "machine": "laptop",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "turn_count": 10,
            "total_tokens": 2000,
            "total_cost": 0.1,
        }

        entry = IndexEntry.model_validate(data)
        assert entry.turn_count == 10
        assert isinstance(entry.created_at, datetime)
        assert isinstance(entry.updated_at, datetime)

    def test_index_entry_to_dict(self):
        """Test serializing index entry to dict."""
        now = datetime.now(timezone.utc)
        entry = IndexEntry(
            agent="test_agent",
            model="test:model",
            machine="test_machine",
            created_at=now,
            updated_at=now,
            turn_count=5,
        )

        data = entry.model_dump(mode="json", exclude_none=True)
        assert data["agent"] == "test_agent"
        assert data["turn_count"] == 5
        assert isinstance(data["created_at"], str)
        assert "total_tokens" not in data  # Excluded because None

class TestDatetimeHandling:
    """Tests for datetime parsing across all models."""

    def test_iso_string_with_z(self):
        """Test parsing ISO string with Z suffix."""
        data = {
            "id": "test",
            "agent": "test",
            "model": "test",
            "machine": "test",
            "created_at": "2025-10-24T10:30:00Z",
        }

        metadata = ConversationMetadata.model_validate(data)
        assert isinstance(metadata.created_at, datetime)
        assert metadata.created_at.tzinfo is not None  # Has timezone info

    def test_iso_string_with_offset(self):
        """Test parsing ISO string with timezone offset."""
        data = {
            "timestamp": "2025-10-24T10:30:00+00:00",
            "user": "test",
            "assistant": "test",
        }

        turn = Turn.model_validate(data)
        assert isinstance(turn.timestamp, datetime)

    def test_datetime_object(self):
        """Test that datetime objects pass through unchanged."""
        now = datetime.now(timezone.utc)
        turn = Turn(
            timestamp=now,
            user="test",
            assistant="test",
        )

        assert turn.timestamp == now
        assert isinstance(turn.timestamp, datetime)
