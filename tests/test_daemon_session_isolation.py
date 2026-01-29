"""Tests for daemon session isolation bug fix."""

from datetime import datetime, timezone

import pytest

from tsugite.daemon.adapters.base import ChannelContext
from tsugite.history import (
    IndexEntry,
    Turn,
    generate_conversation_id,
    load_conversation,
    save_turn_to_history,
    update_index,
)
from tsugite.history.index import find_latest_session


class TestChannelContextDaemonFlag:
    """Tests for is_daemon_managed flag in ChannelContext."""

    def test_channel_context_includes_daemon_flag(self):
        """Test that to_dict() includes is_daemon_managed=True."""
        context = ChannelContext(
            source="discord", channel_id="123", user_id="user1", reply_to="discord:123"
        )
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


class TestFindLatestSessionDaemonOnly:
    """Tests for find_latest_session with daemon_only parameter."""

    def test_find_latest_session_daemon_only_true(self, tmp_path, monkeypatch):
        """Test daemon_only=True filters to only daemon sessions."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.index.get_history_dir", lambda: tmp_path)

        # Create daemon session (older)
        daemon_conv = generate_conversation_id("odyn")
        daemon_turn = Turn(
            timestamp=datetime(2026, 1, 27, 10, 30, 0, tzinfo=timezone.utc),
            user="Daemon message",
            assistant="Daemon response",
            metadata={"is_daemon_managed": True, "source": "discord"},
        )
        save_turn_to_history(daemon_conv, daemon_turn)
        update_index(
            daemon_conv,
            IndexEntry(
                agent="odyn",
                model="test",
                machine="test",
                created_at=datetime(2026, 1, 27, 10, 30, 0, tzinfo=timezone.utc),
                updated_at=datetime(2026, 1, 27, 10, 30, 0, tzinfo=timezone.utc),
                is_daemon_managed=True,
            ),
        )

        # Create ad-hoc CLI session (newer)
        cli_conv = generate_conversation_id("odyn")
        cli_turn = Turn(
            timestamp=datetime(2026, 1, 27, 10, 35, 0, tzinfo=timezone.utc),
            user="Quick CLI question",
            assistant="CLI answer",
            metadata={"source": "cli"},
        )
        save_turn_to_history(cli_conv, cli_turn)
        update_index(
            cli_conv,
            IndexEntry(
                agent="odyn",
                model="test",
                machine="test",
                created_at=datetime(2026, 1, 27, 10, 35, 0, tzinfo=timezone.utc),
                updated_at=datetime(2026, 1, 27, 10, 35, 0, tzinfo=timezone.utc),
            ),
        )

        # Without daemon_only: should find CLI (newest)
        result = find_latest_session("odyn", daemon_only=False)
        assert result == cli_conv

        # With daemon_only: should find daemon session
        result = find_latest_session("odyn", daemon_only=True)
        assert result == daemon_conv

    def test_find_latest_session_daemon_only_no_daemon_sessions(self, tmp_path, monkeypatch):
        """Test daemon_only=True returns None when no daemon sessions exist."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.index.get_history_dir", lambda: tmp_path)

        # Create only ad-hoc CLI session
        cli_conv = generate_conversation_id("odyn")
        cli_turn = Turn(
            timestamp=datetime.now(timezone.utc),
            user="CLI question",
            assistant="CLI answer",
            metadata={"source": "cli"},
        )
        save_turn_to_history(cli_conv, cli_turn)
        update_index(
            cli_conv,
            IndexEntry(
                agent="odyn",
                model="test",
                machine="test",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            ),
        )

        # daemon_only should return None
        result = find_latest_session("odyn", daemon_only=True)
        assert result is None

    def test_find_latest_session_daemon_only_with_user_id(self, tmp_path, monkeypatch):
        """Test daemon_only combined with user_id filtering."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.index.get_history_dir", lambda: tmp_path)

        # Daemon session for user1
        daemon1_conv = generate_conversation_id("odyn")
        daemon1_turn = Turn(
            timestamp=datetime(2026, 1, 27, 10, 30, 0, tzinfo=timezone.utc),
            user="User1 message",
            assistant="Response",
            metadata={"is_daemon_managed": True, "source": "discord", "user_id": "user1"},
        )
        save_turn_to_history(daemon1_conv, daemon1_turn)
        update_index(
            daemon1_conv,
            IndexEntry(
                agent="odyn",
                model="test",
                machine="test",
                created_at=datetime(2026, 1, 27, 10, 30, 0, tzinfo=timezone.utc),
                updated_at=datetime(2026, 1, 27, 10, 30, 0, tzinfo=timezone.utc),
                is_daemon_managed=True,
            ),
        )

        # Daemon session for user2
        daemon2_conv = generate_conversation_id("odyn")
        daemon2_turn = Turn(
            timestamp=datetime(2026, 1, 27, 10, 40, 0, tzinfo=timezone.utc),
            user="User2 message",
            assistant="Response",
            metadata={"is_daemon_managed": True, "source": "discord", "user_id": "user2"},
        )
        save_turn_to_history(daemon2_conv, daemon2_turn)
        update_index(
            daemon2_conv,
            IndexEntry(
                agent="odyn",
                model="test",
                machine="test",
                created_at=datetime(2026, 1, 27, 10, 40, 0, tzinfo=timezone.utc),
                updated_at=datetime(2026, 1, 27, 10, 40, 0, tzinfo=timezone.utc),
                is_daemon_managed=True,
            ),
        )

        # Find daemon session for user1
        result = find_latest_session("odyn", user_id="user1", daemon_only=True)
        assert result == daemon1_conv

        # Find daemon session for user2
        result = find_latest_session("odyn", user_id="user2", daemon_only=True)
        assert result == daemon2_conv

    def test_find_latest_session_daemon_only_no_index(self, tmp_path, monkeypatch):
        """Test daemon_only=True when index file doesn't exist."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.index.get_history_dir", lambda: tmp_path)

        # No index file exists, so find_latest_session should return None
        result = find_latest_session("odyn", daemon_only=True)
        assert result is None

        # Same for regular search
        result = find_latest_session("odyn", daemon_only=False)
        assert result is None


class TestDaemonAgentField:
    """Tests for daemon_agent field to prevent cross-agent issues."""

    def test_daemon_agent_in_metadata(self, tmp_path, monkeypatch):
        """Test that daemon_agent is saved in metadata."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)

        conv_id = generate_conversation_id("odyn")

        # Simulate daemon metadata with daemon_agent field
        turn = Turn(
            timestamp=datetime.now(timezone.utc),
            user="Test",
            assistant="Response",
            metadata={"is_daemon_managed": True, "daemon_agent": "odyn", "source": "discord"},
        )
        save_turn_to_history(conv_id, turn)

        # Verify saved
        turns = load_conversation(conv_id)
        assert len(turns) == 1
        assert turns[0].metadata["daemon_agent"] == "odyn"

    def test_multiple_daemon_agents_isolated(self, tmp_path, monkeypatch):
        """Test that different daemon agents maintain separate sessions."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.index.get_history_dir", lambda: tmp_path)

        # Odyn daemon session
        odyn_conv = generate_conversation_id("odyn")
        odyn_turn = Turn(
            timestamp=datetime.now(timezone.utc),
            user="Odyn message",
            assistant="Response",
            metadata={"is_daemon_managed": True, "daemon_agent": "odyn"},
        )
        save_turn_to_history(odyn_conv, odyn_turn)
        update_index(
            odyn_conv,
            IndexEntry(
                agent="odyn",
                model="test",
                machine="test",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                is_daemon_managed=True,
            ),
        )

        # Loki daemon session
        loki_conv = generate_conversation_id("loki")
        loki_turn = Turn(
            timestamp=datetime.now(timezone.utc),
            user="Loki message",
            assistant="Response",
            metadata={"is_daemon_managed": True, "daemon_agent": "loki"},
        )
        save_turn_to_history(loki_conv, loki_turn)
        update_index(
            loki_conv,
            IndexEntry(
                agent="loki",
                model="test",
                machine="test",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                is_daemon_managed=True,
            ),
        )

        # Find latest for each agent
        odyn_result = find_latest_session("odyn", daemon_only=True)
        loki_result = find_latest_session("loki", daemon_only=True)

        assert odyn_result == odyn_conv
        assert loki_result == loki_conv


class TestCompactionPreservesDaemonFlag:
    """Tests for compaction preserving is_daemon_managed flag."""

    def test_compaction_summary_has_daemon_flag(self, tmp_path, monkeypatch):
        """Test that compaction summary includes is_daemon_managed flag."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)

        # Simulate compaction summary turn
        new_conv = generate_conversation_id("odyn")
        summary_turn = Turn(
            timestamp=datetime.now(timezone.utc),
            user="[Session Summary]",
            assistant="Previous session summary:\n\nConversation about X",
            metadata={
                "type": "compaction_summary",
                "source_session": "old_conv_123",
                "compacted_at": datetime.now(timezone.utc).isoformat(),
                "original_message_count": 50,
                "is_daemon_managed": True,
                "daemon_agent": "odyn",
            },
        )
        save_turn_to_history(new_conv, summary_turn)

        # Verify flag is preserved
        turns = load_conversation(new_conv)
        assert len(turns) == 1
        assert turns[0].metadata["is_daemon_managed"] is True
        assert turns[0].metadata["daemon_agent"] == "odyn"
        assert turns[0].metadata["type"] == "compaction_summary"

    def test_compacted_session_found_by_daemon_only(self, tmp_path, monkeypatch):
        """Test that compacted session is still found with daemon_only=True."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.index.get_history_dir", lambda: tmp_path)

        # Create new compacted session
        new_conv = generate_conversation_id("odyn")
        summary_turn = Turn(
            timestamp=datetime.now(timezone.utc),
            user="[Session Summary]",
            assistant="Summary",
            metadata={"is_daemon_managed": True, "daemon_agent": "odyn"},
        )
        save_turn_to_history(new_conv, summary_turn)
        update_index(
            new_conv,
            IndexEntry(
                agent="odyn",
                model="test",
                machine="test",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                is_daemon_managed=True,
            ),
        )

        # Should be found with daemon_only=True
        result = find_latest_session("odyn", daemon_only=True)
        assert result == new_conv


class TestCLIDaemonMode:
    """Tests for CLI --daemon mode metadata."""

    def test_cli_daemon_metadata_includes_flags(self, tmp_path, monkeypatch):
        """Test that CLI --daemon mode sets proper metadata flags."""
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)

        conv_id = generate_conversation_id("odyn")

        # Simulate CLI --daemon metadata
        cli_daemon_turn = Turn(
            timestamp=datetime.now(timezone.utc),
            user="CLI daemon message",
            assistant="Response",
            metadata={
                "source": "cli",
                "channel_id": None,
                "user_id": "justyn",
                "reply_to": "cli",
                "is_daemon_managed": True,
                "daemon_agent": "odyn",
            },
        )
        save_turn_to_history(conv_id, cli_daemon_turn)

        # Verify metadata
        turns = load_conversation(conv_id)
        assert turns[0].metadata["is_daemon_managed"] is True
        assert turns[0].metadata["daemon_agent"] == "odyn"
        assert turns[0].metadata["source"] == "cli"
