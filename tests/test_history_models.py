"""Tests for Pydantic history models (V2 format)."""

import json
from datetime import datetime, timezone

from tsugite.history.models import (
    AttachmentRef,
    CompactionSummary,
    ContextSnapshot,
    ContextUpdate,
    SessionMeta,
    Turn,
)


class TestSessionMeta:
    """Tests for SessionMeta model."""

    def test_valid_metadata(self):
        """Test creating valid session metadata."""
        now = datetime.now(timezone.utc)
        meta = SessionMeta(
            agent="chat_assistant",
            model="openai:gpt-4o",
            machine="laptop",
            created_at=now,
        )

        assert meta.type == "session_meta"
        assert meta.agent == "chat_assistant"
        assert meta.model == "openai:gpt-4o"
        assert meta.machine == "laptop"
        assert meta.created_at == now
        assert meta.workspace is None
        assert meta.compacted_from is None

    def test_metadata_from_dict(self):
        """Test creating metadata from dict (model_validate)."""
        now = datetime.now(timezone.utc)
        data = {
            "type": "session_meta",
            "agent": "test_agent",
            "model": "test:model",
            "machine": "test_machine",
            "created_at": now.isoformat(),
        }

        meta = SessionMeta.model_validate(data)
        assert meta.agent == "test_agent"
        assert isinstance(meta.created_at, datetime)

    def test_metadata_to_dict(self):
        """Test serializing metadata to dict."""
        now = datetime.now(timezone.utc)
        meta = SessionMeta(
            agent="test_agent",
            model="test:model",
            machine="test_machine",
            created_at=now,
        )

        data = meta.model_dump(mode="json")
        assert data["type"] == "session_meta"
        assert data["agent"] == "test_agent"
        assert isinstance(data["created_at"], str)

    def test_metadata_json_round_trip(self):
        """Test JSON serialization round-trip."""
        now = datetime.now(timezone.utc)
        original = SessionMeta(
            agent="test_agent",
            model="test:model",
            machine="test_machine",
            created_at=now,
            workspace="my_workspace",
        )

        json_str = original.model_dump_json()
        data = json.loads(json_str)
        restored = SessionMeta.model_validate(data)

        assert restored.agent == original.agent
        assert restored.model == original.model
        assert restored.workspace == original.workspace

    def test_metadata_whitespace_stripping(self):
        """Test that whitespace is stripped from strings."""
        data = {
            "type": "session_meta",
            "agent": "  test_agent  ",
            "model": "  test:model  ",
            "machine": "  test_machine  ",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        meta = SessionMeta.model_validate(data)
        assert meta.agent == "test_agent"

    def test_metadata_with_compaction(self):
        """Test metadata with compaction info."""
        now = datetime.now(timezone.utc)
        meta = SessionMeta(
            agent="chat",
            model="openai:gpt-4o",
            machine="laptop",
            created_at=now,
            compacted_from="20251024_100000_chat_abc123",
        )

        assert meta.compacted_from == "20251024_100000_chat_abc123"


class TestTurn:
    """Tests for Turn model (V2 format with messages array)."""

    def test_valid_turn(self):
        """Test creating valid turn."""
        now = datetime.now(timezone.utc)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        turn = Turn(
            timestamp=now,
            messages=messages,
            final_answer="Hi there!",
            tokens=100,
            cost=0.01,
            functions_called=["web_search"],
        )

        assert turn.type == "turn"
        assert turn.timestamp == now
        assert turn.messages == messages
        assert turn.final_answer == "Hi there!"
        assert turn.tokens == 100
        assert turn.cost == 0.01
        assert turn.functions_called == ["web_search"]

    def test_turn_defaults(self):
        """Test turn with default values."""
        now = datetime.now(timezone.utc)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        turn = Turn(
            timestamp=now,
            messages=messages,
        )

        assert turn.functions_called == []
        assert turn.tokens is None
        assert turn.cost is None
        assert turn.final_answer is None
        assert turn.user_summary is None

    def test_turn_from_dict(self):
        """Test creating turn from dict."""
        now = datetime.now(timezone.utc)
        data = {
            "type": "turn",
            "timestamp": now.isoformat(),
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            "final_answer": "Hi",
            "functions_called": ["read_file", "write_file"],
            "tokens": 150,
            "cost": 0.02,
        }

        turn = Turn.model_validate(data)
        assert isinstance(turn.timestamp, datetime)
        assert turn.functions_called == ["read_file", "write_file"]
        assert len(turn.messages) == 2

    def test_turn_to_dict(self):
        """Test serializing turn to dict."""
        now = datetime.now(timezone.utc)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        turn = Turn(
            timestamp=now,
            messages=messages,
            final_answer="Hi",
            functions_called=["web_search"],
            tokens=100,
        )

        data = turn.model_dump(mode="json", exclude_none=True)
        assert data["type"] == "turn"
        assert data["messages"] == messages
        assert isinstance(data["timestamp"], str)
        assert "cost" not in data

    def test_turn_json_serialization(self):
        """Test JSON serialization."""
        now = datetime.now(timezone.utc)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        turn = Turn(
            timestamp=now,
            messages=messages,
        )

        json_str = turn.model_dump_json(exclude_none=True)
        assert "Hello" in json_str
        assert "timestamp" in json_str

    def test_turn_with_channel_metadata(self):
        """Test turn with channel routing metadata."""
        now = datetime.now(timezone.utc)
        messages = [
            {"role": "user", "content": "Hello from Discord"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        metadata = {
            "source": "discord",
            "channel_id": "123456789",
            "user_id": "user123",
            "reply_to": "discord:123456789",
            "author_name": "TestUser",
            "is_daemon_managed": True,
        }

        turn = Turn(
            timestamp=now,
            messages=messages,
            final_answer="Hi there!",
            metadata=metadata,
        )

        assert turn.metadata is not None
        assert turn.metadata["source"] == "discord"
        assert turn.metadata["channel_id"] == "123456789"
        assert turn.metadata["is_daemon_managed"] is True

    def test_turn_metadata_serialization(self):
        """Test metadata field serialization."""
        now = datetime.now(timezone.utc)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        metadata = {
            "source": "cli",
            "user_id": "cli-user",
            "reply_to": "cli",
        }

        turn = Turn(
            timestamp=now,
            messages=messages,
            metadata=metadata,
        )

        data = turn.model_dump(mode="json")
        assert "metadata" in data
        assert data["metadata"]["source"] == "cli"

        turn2 = Turn.model_validate(data)
        assert turn2.metadata == metadata

    def test_turn_with_user_summary(self):
        """Test turn with user summary."""
        now = datetime.now(timezone.utc)
        messages = [
            {"role": "user", "content": "This is a very long message..."},
            {"role": "assistant", "content": "Response"},
        ]
        turn = Turn(
            timestamp=now,
            messages=messages,
            user_summary="This is a very long message...",
        )

        assert turn.user_summary == "This is a very long message..."


class TestAttachmentRef:
    """Tests for AttachmentRef model."""

    def test_file_attachment_ref(self):
        """Test file-based attachment reference."""
        ref = AttachmentRef(
            hash="sha256_abc123",
            type="text",
            source="file",
        )

        assert ref.hash == "sha256_abc123"
        assert ref.type == "text"
        assert ref.source == "file"
        assert ref.url is None

    def test_url_attachment_ref(self):
        """Test URL-based attachment reference."""
        ref = AttachmentRef(
            url="https://example.com/image.png",
            type="image",
            source="url",
            mime_type="image/png",
        )

        assert ref.url == "https://example.com/image.png"
        assert ref.type == "image"
        assert ref.source == "url"
        assert ref.mime_type == "image/png"
        assert ref.hash is None

    def test_attachment_ref_with_original_path(self):
        """Test attachment with fallback path."""
        ref = AttachmentRef(
            hash="sha256_xyz",
            type="document",
            source="file",
            original_path="/home/user/docs/report.pdf",
            mime_type="application/pdf",
        )

        assert ref.original_path == "/home/user/docs/report.pdf"


class TestContextSnapshot:
    """Tests for ContextSnapshot model."""

    def test_valid_context_snapshot(self):
        """Test creating valid context snapshot."""
        attachments = {
            "PERSONA.md": AttachmentRef(hash="abc123", type="text", source="file"),
            "logo.png": AttachmentRef(url="https://example.com/logo.png", type="image", source="url"),
        }
        snapshot = ContextSnapshot(
            attachments=attachments,
            skills=["web_search", "code_review"],
            hash="ctx_hash_123",
        )

        assert snapshot.type == "context"
        assert len(snapshot.attachments) == 2
        assert snapshot.skills == ["web_search", "code_review"]
        assert snapshot.hash == "ctx_hash_123"

    def test_context_snapshot_empty(self):
        """Test context snapshot with no attachments."""
        snapshot = ContextSnapshot(
            attachments={},
            skills=[],
            hash="empty_ctx_hash",
        )

        assert snapshot.attachments == {}
        assert snapshot.skills == []


class TestContextUpdate:
    """Tests for ContextUpdate model."""

    def test_valid_context_update(self):
        """Test creating valid context update."""
        now = datetime.now(timezone.utc)
        changed = {
            "PERSONA.md": AttachmentRef(hash="new_hash", type="text", source="file"),
        }
        update = ContextUpdate(
            changed=changed,
            removed=["old_file.md"],
            added_skills=["new_skill"],
            removed_skills=["old_skill"],
            timestamp=now,
            hash="new_ctx_hash",
        )

        assert update.type == "context_update"
        assert len(update.changed) == 1
        assert update.removed == ["old_file.md"]
        assert update.added_skills == ["new_skill"]
        assert update.removed_skills == ["old_skill"]

    def test_context_update_defaults(self):
        """Test context update with defaults."""
        now = datetime.now(timezone.utc)
        update = ContextUpdate(
            timestamp=now,
            hash="hash123",
        )

        assert update.changed == {}
        assert update.removed == []
        assert update.added_skills == []
        assert update.removed_skills == []


class TestCompactionSummary:
    """Tests for CompactionSummary model."""

    def test_valid_compaction_summary(self):
        """Test creating valid compaction summary."""
        summary = CompactionSummary(
            summary="The user discussed project setup and configuration.",
            previous_turns=50,
        )

        assert summary.type == "compaction_summary"
        assert "project setup" in summary.summary
        assert summary.previous_turns == 50

    def test_compaction_summary_serialization(self):
        """Test compaction summary serialization."""
        summary = CompactionSummary(
            summary="Previous conversation summary.",
            previous_turns=25,
        )

        data = summary.model_dump(mode="json")
        assert data["type"] == "compaction_summary"
        assert data["previous_turns"] == 25

        restored = CompactionSummary.model_validate(data)
        assert restored.summary == summary.summary


class TestDatetimeHandling:
    """Tests for datetime parsing across all models."""

    def test_iso_string_with_z(self):
        """Test parsing ISO string with Z suffix."""
        data = {
            "type": "session_meta",
            "agent": "test",
            "model": "test",
            "machine": "test",
            "created_at": "2025-10-24T10:30:00Z",
        }

        meta = SessionMeta.model_validate(data)
        assert isinstance(meta.created_at, datetime)
        assert meta.created_at.tzinfo is not None

    def test_iso_string_with_offset(self):
        """Test parsing ISO string with timezone offset."""
        data = {
            "type": "turn",
            "timestamp": "2025-10-24T10:30:00+00:00",
            "messages": [{"role": "user", "content": "test"}],
        }

        turn = Turn.model_validate(data)
        assert isinstance(turn.timestamp, datetime)

    def test_datetime_object(self):
        """Test that datetime objects pass through unchanged."""
        now = datetime.now(timezone.utc)
        messages = [{"role": "user", "content": "test"}]
        turn = Turn(
            timestamp=now,
            messages=messages,
        )

        assert turn.timestamp == now
        assert isinstance(turn.timestamp, datetime)
