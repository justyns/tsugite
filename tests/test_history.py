"""Tests for conversation history system V2."""

from datetime import datetime, timezone

import pytest

from tsugite.history import (
    SessionStorage,
    Turn,
    generate_session_id,
    get_history_dir,
    list_session_files,
    reconstruct_messages,
)
from tsugite.history.storage import get_machine_name


class TestSessionID:
    """Tests for session ID generation."""

    def test_generate_session_id(self):
        """Test generating a unique session ID."""
        agent_name = "test_agent"
        session_id = generate_session_id(agent_name)

        assert session_id is not None
        assert isinstance(session_id, str)
        assert "test_agent" in session_id
        assert len(session_id) > 0

    def test_generate_session_id_with_timestamp(self):
        """Test generating ID with specific timestamp."""
        agent_name = "chat"
        timestamp = datetime(2025, 10, 24, 10, 30, 0, tzinfo=timezone.utc)
        session_id = generate_session_id(agent_name, timestamp)

        assert session_id.startswith("20251024_103000")
        assert "chat" in session_id

    def test_generate_session_id_unique(self):
        """Test that generated IDs are unique."""
        agent_name = "agent"
        id1 = generate_session_id(agent_name)
        id2 = generate_session_id(agent_name)

        assert id1 != id2


class TestSessionStorage:
    """Tests for SessionStorage class."""

    def test_get_history_dir(self, tmp_path, monkeypatch):
        """Test getting history directory path."""

        def mock_get_xdg_data_path(subdir):
            return tmp_path / "tsugite" / subdir

        monkeypatch.setattr("tsugite.history.storage.get_xdg_data_path", mock_get_xdg_data_path)

        history_dir = get_history_dir()
        assert history_dir == tmp_path / "tsugite" / "history"

    def test_create_session(self, tmp_path, monkeypatch):
        """Test creating a new session."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test_machine")

        storage = SessionStorage.create(
            agent_name="test_agent",
            model="openai:gpt-4o",
        )

        assert storage.session_id is not None
        assert storage.agent == "test_agent"
        assert storage.model == "openai:gpt-4o"
        assert storage.machine == "test_machine"
        assert storage.turn_count == 0

    def test_record_turn(self, tmp_path, monkeypatch):
        """Test recording a turn."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test_machine")

        storage = SessionStorage.create(
            agent_name="test_agent",
            model="openai:gpt-4o",
        )

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        storage.record_turn(
            messages=messages,
            final_answer="Hi there!",
            tokens=50,
            cost=0.001,
            functions_called=["test_tool"],
        )

        assert storage.turn_count == 1
        assert storage.total_tokens == 50
        assert storage.total_cost == 0.001

    def test_load_session(self, tmp_path, monkeypatch):
        """Test loading an existing session."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test_machine")

        # Create a session
        storage1 = SessionStorage.create(
            agent_name="test_agent",
            model="openai:gpt-4o",
        )

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        storage1.record_turn(messages=messages, final_answer="Hi!", tokens=100, cost=0.01)

        # Load it
        storage2 = SessionStorage.load(storage1.session_path)

        assert storage2.session_id == storage1.session_id
        assert storage2.agent == "test_agent"
        assert storage2.turn_count == 1
        assert storage2.total_tokens == 100

    def test_load_records(self, tmp_path, monkeypatch):
        """Test loading all records from session."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test_machine")

        storage = SessionStorage.create(
            agent_name="test_agent",
            model="openai:gpt-4o",
        )

        storage.record_turn(
            messages=[{"role": "user", "content": "msg1"}, {"role": "assistant", "content": "resp1"}],
            final_answer="resp1",
        )
        storage.record_turn(
            messages=[{"role": "user", "content": "msg2"}, {"role": "assistant", "content": "resp2"}],
            final_answer="resp2",
        )

        records = storage.load_records()
        # session_meta + 2 turns
        assert len(records) == 3

        turns = [r for r in records if isinstance(r, Turn)]
        assert len(turns) == 2
        assert turns[0].final_answer == "resp1"
        assert turns[1].final_answer == "resp2"

    def test_context_recording(self, tmp_path, monkeypatch):
        """Test recording initial context."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test_machine")
        monkeypatch.setattr("tsugite.cache.get_xdg_cache_path", lambda x: tmp_path / "cache" / x)

        storage = SessionStorage.create(
            agent_name="test_agent",
            model="openai:gpt-4o",
        )

        from tsugite.attachments.base import Attachment, AttachmentContentType

        attachments = [
            Attachment(
                name="test.txt",
                content="Hello world",
                content_type=AttachmentContentType.TEXT,
                mime_type="text/plain",
            )
        ]

        storage.record_initial_context(attachments=attachments, skills=["test_skill"])

        assert storage._current_context_hash is not None
        assert "test.txt" in storage._current_attachments
        assert "test_skill" in storage._current_skills


class TestListSessions:
    """Tests for listing session files."""

    def test_list_session_files(self, tmp_path, monkeypatch):
        """Test listing session files."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test_machine")

        # Create a few sessions
        for i in range(3):
            SessionStorage.create(agent_name=f"agent{i}", model="test")

        files = list_session_files()
        assert len(files) == 3


class TestMachineName:
    """Tests for machine name detection."""

    def test_get_machine_name(self, monkeypatch):
        """Test getting machine name."""
        machine_name = get_machine_name()
        assert machine_name is not None
        assert len(machine_name) > 0

    def test_get_machine_name_config_override(self, monkeypatch):
        """Test machine name from config."""
        from tsugite.config import Config

        def mock_load_config():
            config = Config()
            config.machine_name = "custom_machine"
            return config

        monkeypatch.setattr("tsugite.history.storage.load_config", mock_load_config)

        machine_name = get_machine_name()
        assert machine_name == "custom_machine"


class TestReconstruction:
    """Tests for message reconstruction."""

    def test_reconstruct_messages(self, tmp_path, monkeypatch):
        """Test reconstructing messages from session."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test_machine")

        storage = SessionStorage.create(
            agent_name="test_agent",
            model="openai:gpt-4o",
        )

        storage.record_turn(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            final_answer="Hi there!",
        )

        messages = reconstruct_messages(storage.session_path)

        # Should have the turn messages
        assert len(messages) >= 2
        user_msgs = [m for m in messages if m.get("role") == "user"]
        asst_msgs = [m for m in messages if m.get("role") == "assistant"]
        assert len(user_msgs) >= 1
        assert len(asst_msgs) >= 1


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_session(self, tmp_path, monkeypatch):
        """Test handling empty session file."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)

        session_file = tmp_path / "empty.jsonl"
        session_file.write_text("")

        storage = SessionStorage(session_file)
        records = storage.load_records()
        assert len(records) == 0

    def test_malformed_jsonl(self, tmp_path, monkeypatch):
        """Test handling malformed JSONL line."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)

        session_file = tmp_path / "malformed.jsonl"
        session_file.write_text(
            '{"type": "session_meta", "agent": "test", "model": "test", "machine": "test", "created_at": "2025-10-24T10:00:00+00:00"}\n'
            + "not valid json\n"
            + '{"type": "turn", "timestamp": "2025-10-24T10:00:00+00:00", "messages": []}'
        )

        storage = SessionStorage(session_file)
        records = storage.load_records()
        # Should skip malformed line
        assert len(records) == 2

    def test_nonexistent_session(self, tmp_path):
        """Test loading a session that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            SessionStorage.load(tmp_path / "nonexistent.jsonl")


class TestCompactionSummary:
    """Tests for compaction summary."""

    def test_record_compaction_summary(self, tmp_path, monkeypatch):
        """Test recording compaction summary."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test_machine")

        storage = SessionStorage.create(
            agent_name="test_agent",
            model="openai:gpt-4o",
            compacted_from="old_session_123",
        )

        storage.record_compaction_summary(
            summary="Previous conversation was about testing.",
            previous_turns=10,
        )

        records = storage.load_records()

        from tsugite.history.models import CompactionSummary

        summaries = [r for r in records if isinstance(r, CompactionSummary)]
        assert len(summaries) == 1
        assert summaries[0].summary == "Previous conversation was about testing."
        assert summaries[0].previous_turns == 10
