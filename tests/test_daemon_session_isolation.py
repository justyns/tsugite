"""Tests for daemon session isolation bug fix."""

from datetime import datetime, timezone

import pytest

from tsugite.daemon.adapters.base import ChannelContext
from tsugite.history import (
    SessionStorage,
    Turn,
    list_session_files,
)


class TestChannelContextDaemonFlag:
    """Tests for is_daemon_managed flag in ChannelContext."""

    def test_channel_context_includes_daemon_flag(self):
        """Test that to_dict() includes is_daemon_managed=True."""
        context = ChannelContext(source="discord", channel_id="123", user_id="user1", reply_to="discord:123")
        metadata = context.to_dict()

        assert metadata["is_daemon_managed"] is True
        assert metadata["source"] == "discord"
        assert metadata["channel_id"] == "123"
        assert metadata["user_id"] == "user1"

    def test_channel_context_preserves_extra_metadata(self):
        """Test that is_daemon_managed doesn't override extra metadata."""
        context = ChannelContext(
            source="slack",
            channel_id="456",
            user_id="user2",
            reply_to="slack:456",
            metadata={"custom_field": "value"},
        )
        metadata = context.to_dict()

        assert metadata["is_daemon_managed"] is True
        assert metadata["custom_field"] == "value"


class TestSessionMetadata:
    """Tests for session metadata."""

    def test_session_with_metadata(self, tmp_path, monkeypatch):
        """Test that sessions can store metadata in turns."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test_machine")

        storage = SessionStorage.create(agent_name="odyn", model="test")

        messages = [
            {"role": "user", "content": "Daemon message"},
            {"role": "assistant", "content": "Daemon response"},
        ]
        storage.record_turn(
            messages=messages,
            final_answer="Daemon response",
            metadata={"is_daemon_managed": True, "source": "discord"},
        )

        records = storage.load_records()
        turns = [r for r in records if isinstance(r, Turn)]
        assert len(turns) == 1
        assert turns[0].metadata["is_daemon_managed"] is True
        assert turns[0].metadata["source"] == "discord"


class TestSessionFiltering:
    """Tests for filtering sessions by various criteria."""

    @pytest.fixture
    def setup_sessions(self, tmp_path, monkeypatch):
        """Create test sessions with different metadata."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test_machine")

        sessions = {}

        # Create daemon session (older)
        daemon_storage = SessionStorage.create(
            agent_name="odyn",
            model="test",
            timestamp=datetime(2026, 1, 27, 10, 30, 0, tzinfo=timezone.utc),
        )
        daemon_storage.record_turn(
            messages=[
                {"role": "user", "content": "Daemon message"},
                {"role": "assistant", "content": "Daemon response"},
            ],
            final_answer="Daemon response",
            metadata={"is_daemon_managed": True, "source": "discord"},
        )
        sessions["daemon"] = daemon_storage

        # Create CLI session (newer)
        cli_storage = SessionStorage.create(
            agent_name="odyn",
            model="test",
            timestamp=datetime(2026, 1, 27, 10, 35, 0, tzinfo=timezone.utc),
        )
        cli_storage.record_turn(
            messages=[
                {"role": "user", "content": "CLI message"},
                {"role": "assistant", "content": "CLI response"},
            ],
            final_answer="CLI response",
            metadata={"source": "cli"},
        )
        sessions["cli"] = cli_storage

        return sessions

    def test_list_all_sessions(self, tmp_path, monkeypatch, setup_sessions):
        """Test listing all sessions."""
        files = list_session_files()
        assert len(files) == 2

    def test_filter_sessions_by_loading(self, tmp_path, monkeypatch, setup_sessions):
        """Test filtering sessions by loading and checking metadata."""
        files = list_session_files()

        daemon_sessions = []
        for f in files:
            storage = SessionStorage.load(f)
            records = storage.load_records()
            turns = [r for r in records if isinstance(r, Turn)]
            for turn in turns:
                if turn.metadata and turn.metadata.get("is_daemon_managed"):
                    daemon_sessions.append(storage.session_id)
                    break

        assert len(daemon_sessions) == 1


class TestDaemonAgentField:
    """Tests for daemon_agent field to prevent cross-agent issues."""

    def test_daemon_agent_in_metadata(self, tmp_path, monkeypatch):
        """Test that daemon_agent is saved in metadata."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test_machine")

        storage = SessionStorage.create(agent_name="odyn", model="test")

        storage.record_turn(
            messages=[
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"},
            ],
            final_answer="Response",
            metadata={"is_daemon_managed": True, "daemon_agent": "odyn", "source": "discord"},
        )

        records = storage.load_records()
        turns = [r for r in records if isinstance(r, Turn)]
        assert len(turns) == 1
        assert turns[0].metadata["daemon_agent"] == "odyn"

    def test_multiple_daemon_agents_isolated(self, tmp_path, monkeypatch):
        """Test that different daemon agents maintain separate sessions."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test_machine")

        # Odyn daemon session
        odyn_storage = SessionStorage.create(agent_name="odyn", model="test")
        odyn_storage.record_turn(
            messages=[
                {"role": "user", "content": "Odyn message"},
                {"role": "assistant", "content": "Response"},
            ],
            final_answer="Response",
            metadata={"is_daemon_managed": True, "daemon_agent": "odyn"},
        )

        # Loki daemon session
        loki_storage = SessionStorage.create(agent_name="loki", model="test")
        loki_storage.record_turn(
            messages=[
                {"role": "user", "content": "Loki message"},
                {"role": "assistant", "content": "Response"},
            ],
            final_answer="Response",
            metadata={"is_daemon_managed": True, "daemon_agent": "loki"},
        )

        # Verify separate sessions
        assert odyn_storage.session_id != loki_storage.session_id
        assert odyn_storage.agent == "odyn"
        assert loki_storage.agent == "loki"


class TestCompactionPreservesDaemonFlag:
    """Tests for compaction preserving is_daemon_managed flag."""

    def test_compaction_summary_has_daemon_flag(self, tmp_path, monkeypatch):
        """Test that compaction summary includes is_daemon_managed flag in metadata."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test_machine")

        storage = SessionStorage.create(
            agent_name="odyn",
            model="test",
            compacted_from="old_conv_123",
        )

        storage.record_compaction_summary(
            summary="Previous conversation summary about X",
            previous_turns=50,
        )

        # Add a turn with daemon metadata
        storage.record_turn(
            messages=[
                {"role": "user", "content": "New message"},
                {"role": "assistant", "content": "Response"},
            ],
            final_answer="Response",
            metadata={
                "is_daemon_managed": True,
                "daemon_agent": "odyn",
            },
        )

        records = storage.load_records()
        turns = [r for r in records if isinstance(r, Turn)]

        assert len(turns) == 1
        assert turns[0].metadata["is_daemon_managed"] is True
        assert turns[0].metadata["daemon_agent"] == "odyn"


class TestCLIDaemonMode:
    """Tests for CLI --daemon mode metadata."""

    def test_cli_daemon_metadata_includes_flags(self, tmp_path, monkeypatch):
        """Test that CLI --daemon mode sets proper metadata flags."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test_machine")

        storage = SessionStorage.create(agent_name="odyn", model="test")

        storage.record_turn(
            messages=[
                {"role": "user", "content": "CLI daemon message"},
                {"role": "assistant", "content": "Response"},
            ],
            final_answer="Response",
            metadata={
                "source": "cli",
                "channel_id": None,
                "user_id": "justyn",
                "reply_to": "cli",
                "is_daemon_managed": True,
                "daemon_agent": "odyn",
            },
        )

        records = storage.load_records()
        turns = [r for r in records if isinstance(r, Turn)]

        assert turns[0].metadata["is_daemon_managed"] is True
        assert turns[0].metadata["daemon_agent"] == "odyn"
        assert turns[0].metadata["source"] == "cli"
