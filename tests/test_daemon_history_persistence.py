"""Tests for daemon conversation history persistence.

These tests verify that:
1. Conversation history is saved after each message (in-session memory)
2. History persists across daemon restarts
3. Multi-adapter sessions work (Discord + CLI share same conversation)
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tsugite.daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite.daemon.config import AgentConfig
from tsugite.daemon.session import SessionManager
from tsugite.history import CompactionSummary, SessionStorage, Turn
from tsugite.history.reconstruction import reconstruct_messages


class _StubAdapter(BaseAdapter):
    """Minimal concrete adapter for testing."""

    async def start(self):
        pass

    async def stop(self):
        pass


def create_mock_result(response: str = "Test response", token_count: int = 100) -> MagicMock:
    """Create a mock AgentExecutionResult with correct attributes."""
    mock_result = MagicMock()
    mock_result.__str__ = lambda self: response
    # Use correct AgentExecutionResult attributes
    mock_result.token_count = token_count
    mock_result.cost = 0.001
    mock_result.execution_steps = []
    mock_result.system_message = "System prompt"
    mock_result.attachments = []
    mock_result.context_window = None
    # Prevent MagicMock auto-creating token_usage.total_tokens
    mock_result.token_usage = None
    return mock_result


class TestSessionManagerPersistence:
    """Tests for SessionManager file-based persistence."""

    def test_session_file_created_on_first_message(self, tmp_path):
        """Session file should be created when a new session starts."""
        sm = SessionManager("test_agent", tmp_path)
        conv_id = sm.get_or_create_session("user123")

        session_file = tmp_path / "daemon_sessions" / "user123.json"
        assert session_file.exists()

        data = json.loads(session_file.read_text())
        assert data["conversation_id"] == conv_id
        assert data["compaction_count"] == 0

    def test_session_persists_after_restart(self, tmp_path):
        """Session should be restored from file after SessionManager recreated."""
        # First session manager instance
        sm1 = SessionManager("test_agent", tmp_path)
        original_conv_id = sm1.get_or_create_session("user456")

        # Simulate restart: create new SessionManager instance
        sm2 = SessionManager("test_agent", tmp_path)
        restored_conv_id = sm2.get_or_create_session("user456")

        assert restored_conv_id == original_conv_id

    def test_compaction_updates_session_file(self, tmp_path):
        """Compaction should update session file with new conversation ID."""
        sm = SessionManager("test_agent", tmp_path)
        original_conv_id = sm.get_or_create_session("user789")

        # Compact session
        new_conv_id = sm.compact_session("user789")

        assert new_conv_id != original_conv_id

        # Verify file updated
        session_file = tmp_path / "daemon_sessions" / "user789.json"
        data = json.loads(session_file.read_text())
        assert data["conversation_id"] == new_conv_id
        assert data["compaction_count"] == 1

    def test_session_restored_after_compaction_and_restart(self, tmp_path):
        """After compaction and restart, should use compacted conversation ID."""
        sm1 = SessionManager("test_agent", tmp_path)
        sm1.get_or_create_session("userABC")
        compacted_conv_id = sm1.compact_session("userABC")

        # Restart
        sm2 = SessionManager("test_agent", tmp_path)
        restored_conv_id = sm2.get_or_create_session("userABC")

        assert restored_conv_id == compacted_conv_id


class TestDaemonHistorySaving:
    """Tests for conversation history being saved by daemon."""

    @pytest.fixture
    def mock_workspace(self, tmp_path):
        """Create a minimal workspace directory."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        return workspace

    @pytest.fixture
    def mock_agent_config(self, mock_workspace):
        """Create a minimal agent config."""
        return AgentConfig(
            workspace_dir=mock_workspace,
            agent_file="test.md",
        )

    @pytest.fixture
    def mock_session_manager(self, mock_workspace):
        """Create a session manager."""
        return SessionManager("test_agent", mock_workspace)

    def test_history_saved_after_message(
        self, tmp_path, mock_workspace, mock_agent_config, mock_session_manager, monkeypatch
    ):
        """Conversation history should be saved after handle_message."""
        # Patch history directory
        history_dir = tmp_path / "history"
        history_dir.mkdir()
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: history_dir)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test_machine")
        monkeypatch.setattr("tsugite.agent_runner.history_integration.get_history_dir", lambda: history_dir)

        # Get conversation ID that will be used
        conv_id = mock_session_manager.get_or_create_session("test_user")

        adapter = _StubAdapter("test_agent", mock_agent_config, mock_session_manager)

        mock_result = create_mock_result("Test response", 100)

        with patch("tsugite.daemon.adapters.base.run_agent", return_value=mock_result):
            with patch("tsugite.daemon.adapters.base.resolve_agent_path", return_value=Path("/fake/agent.md")):
                channel_context = ChannelContext(
                    source="test", channel_id="ch1", user_id="test_user", reply_to="test:ch1"
                )

                asyncio.run(adapter.handle_message("test_user", "Hello bot", channel_context))

        # Verify history file was created
        history_file = history_dir / f"{conv_id}.jsonl"
        assert history_file.exists(), f"History file should exist at {history_file}"

        # Verify content using V2 API
        storage = SessionStorage.load(history_file)
        records = storage.load_records()
        turns = [r for r in records if isinstance(r, Turn)]
        assert len(turns) >= 1
        # V2 stores messages and final_answer
        assert turns[0].final_answer is not None
        assert "Test response" in (turns[0].final_answer or "")

    def test_history_accumulates_across_messages(
        self, tmp_path, mock_workspace, mock_agent_config, mock_session_manager, monkeypatch
    ):
        """Multiple messages should accumulate in the same conversation history."""
        history_dir = tmp_path / "history"
        history_dir.mkdir()
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: history_dir)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test_machine")
        monkeypatch.setattr("tsugite.agent_runner.history_integration.get_history_dir", lambda: history_dir)

        conv_id = mock_session_manager.get_or_create_session("test_user")

        adapter = _StubAdapter("test_agent", mock_agent_config, mock_session_manager)

        responses = ["Response 1", "Response 2", "Response 3"]
        response_iter = iter(responses)

        def mock_run_agent(*args, **kwargs):
            return create_mock_result(next(response_iter), 50)

        with patch("tsugite.daemon.adapters.base.run_agent", side_effect=mock_run_agent):
            with patch("tsugite.daemon.adapters.base.resolve_agent_path", return_value=Path("/fake/agent.md")):
                channel_context = ChannelContext(
                    source="test", channel_id="ch1", user_id="test_user", reply_to="test:ch1"
                )

                asyncio.run(adapter.handle_message("test_user", "Message 1", channel_context))
                asyncio.run(adapter.handle_message("test_user", "Message 2", channel_context))
                asyncio.run(adapter.handle_message("test_user", "Message 3", channel_context))

        # Verify all turns are in history using V2 API
        storage = SessionStorage.load(history_dir / f"{conv_id}.jsonl")
        records = storage.load_records()
        turns = [r for r in records if isinstance(r, Turn)]
        assert len(turns) == 3


class TestMultiAdapterSessionContinuity:
    """Tests for session continuity across different adapters (Discord + CLI)."""

    def test_same_user_same_conversation_different_sources(self, tmp_path):
        """Same user should get same conversation regardless of source adapter."""
        sm = SessionManager("shared_agent", tmp_path)

        # Discord adapter gets session
        discord_conv = sm.get_or_create_session("user123")

        # CLI adapter gets session for same user
        cli_conv = sm.get_or_create_session("user123")

        assert discord_conv == cli_conv

    def test_history_shared_across_adapters(self, tmp_path, monkeypatch):
        """History saved by one adapter should be visible to another."""
        history_dir = tmp_path / "history"
        history_dir.mkdir()
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: history_dir)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test_machine")

        sm = SessionManager("shared_agent", tmp_path)
        sm.get_or_create_session("user123")

        # Create storage and save turns using V2 API
        storage = SessionStorage.create(agent_name="shared_agent", model="test")

        # Simulate Discord saving a turn
        storage.record_turn(
            messages=[
                {"role": "user", "content": "Hello from Discord"},
                {"role": "assistant", "content": "Hi Discord user!"},
            ],
            final_answer="Hi Discord user!",
            metadata={"source": "discord", "is_daemon_managed": True},
        )

        # Simulate CLI saving a turn to same session
        storage.record_turn(
            messages=[
                {"role": "user", "content": "Hello from CLI"},
                {"role": "assistant", "content": "Hi CLI user!"},
            ],
            final_answer="Hi CLI user!",
            metadata={"source": "cli", "is_daemon_managed": True},
        )

        # Both turns should be in the same conversation
        records = storage.load_records()
        turns = [r for r in records if isinstance(r, Turn)]
        assert len(turns) == 2

        sources = [t.metadata.get("source") for t in turns if t.metadata]
        assert "discord" in sources
        assert "cli" in sources

    def test_different_users_get_different_conversations(self, tmp_path):
        """Different users should have separate conversations."""
        sm = SessionManager("shared_agent", tmp_path)

        user1_conv = sm.get_or_create_session("user1")
        user2_conv = sm.get_or_create_session("user2")

        assert user1_conv != user2_conv


class TestHistoryLoadedOnContinue:
    """Tests for verifying history is loaded when continuing a conversation."""

    def test_run_agent_receives_history(self, tmp_path, monkeypatch):
        """run_agent should receive continue_conversation_id to load history."""
        history_dir = tmp_path / "history"
        history_dir.mkdir()
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: history_dir)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test_machine")

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        sm = SessionManager("test_agent", workspace)
        conv_id = sm.get_or_create_session("test_user")

        agent_config = AgentConfig(
            workspace_dir=workspace,
            agent_file="test.md",
        )

        adapter = _StubAdapter("test_agent", agent_config, sm)

        captured_kwargs = {}

        def capture_run_agent(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return create_mock_result("Response", 50)

        with patch("tsugite.daemon.adapters.base.run_agent", side_effect=capture_run_agent):
            with patch("tsugite.daemon.adapters.base.resolve_agent_path", return_value=Path("/fake/agent.md")):
                channel_context = ChannelContext(
                    source="test", channel_id="ch1", user_id="test_user", reply_to="test:ch1"
                )

                asyncio.run(adapter.handle_message("test_user", "Test message", channel_context))

        # Verify run_agent was called with continue_conversation_id
        assert "continue_conversation_id" in captured_kwargs
        assert captured_kwargs["continue_conversation_id"] == conv_id


class TestHistorySaveErrorHandling:
    """Tests for error handling during history save."""

    def test_history_save_failure_does_not_break_response(self, tmp_path, monkeypatch):
        """handle_message should still return response even if history save fails."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        sm = SessionManager("test_agent", workspace)
        agent_config = AgentConfig(
            workspace_dir=workspace,
            agent_file="test.md",
        )

        adapter = _StubAdapter("test_agent", agent_config, sm)

        mock_result = create_mock_result("Success response", 100)

        def failing_save(*args, **kwargs):
            raise RuntimeError("Simulated history save failure")

        with patch("tsugite.daemon.adapters.base.run_agent", return_value=mock_result):
            with patch("tsugite.daemon.adapters.base.resolve_agent_path", return_value=Path("/fake/agent.md")):
                with patch("tsugite.agent_runner.history_integration.save_run_to_history", side_effect=failing_save):
                    channel_context = ChannelContext(
                        source="test", channel_id="ch1", user_id="test_user", reply_to="test:ch1"
                    )

                    # Should not raise, should return the response
                    response = asyncio.run(adapter.handle_message("test_user", "Test message", channel_context))

        assert response == "Success response"


class TestDaemonMetadataInHistory:
    """Tests for daemon-specific metadata in saved history."""

    def test_daemon_metadata_saved_in_history(self, tmp_path, monkeypatch):
        """Verify channel metadata is correctly saved to history."""
        history_dir = tmp_path / "history"
        history_dir.mkdir()
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: history_dir)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test_machine")
        monkeypatch.setattr("tsugite.agent_runner.history_integration.get_history_dir", lambda: history_dir)

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        sm = SessionManager("test_agent", workspace)

        agent_config = AgentConfig(
            workspace_dir=workspace,
            agent_file="test.md",
        )

        adapter = _StubAdapter("test_agent", agent_config, sm)
        mock_result = create_mock_result("Response", 50)

        with patch("tsugite.daemon.adapters.base.run_agent", return_value=mock_result):
            with patch("tsugite.daemon.adapters.base.resolve_agent_path", return_value=Path("/fake/agent.md")):
                channel_context = ChannelContext(
                    source="discord",
                    channel_id="channel123",
                    user_id="test_user",
                    reply_to="discord:channel123",
                    metadata={"guild_id": "guild456"},
                )

                asyncio.run(adapter.handle_message("test_user", "Hello", channel_context))

        # Group chat resolves to composite key: "discord:channel123:test_user"
        resolved_user = adapter.resolve_user("test_user", channel_context)
        conv_id = sm.get_or_create_session(resolved_user)

        # Verify metadata in saved turn using V2 API
        storage = SessionStorage.load(history_dir / f"{conv_id}.jsonl")
        records = storage.load_records()
        turns = [r for r in records if isinstance(r, Turn)]
        assert len(turns) == 1

        metadata = turns[0].metadata
        assert metadata is not None
        assert metadata.get("is_daemon_managed") is True
        assert metadata.get("source") == "discord"
        assert metadata.get("channel_id") == "channel123"
        assert metadata.get("daemon_agent") == "test_agent"


class TestCompactionWithRetainedTurns:
    """Integration test: compaction keeps recent turns verbatim."""

    def test_compact_creates_summary_plus_retained_turns(self, tmp_path, monkeypatch):
        """After compaction, new session should have summary + retained turns."""
        history_dir = tmp_path / "history"
        history_dir.mkdir()
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: history_dir)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test_machine")

        storage = SessionStorage.create(
            agent_name="test_agent",
            model="openai:gpt-4o",
            session_path=history_dir / "old_session.jsonl",
        )

        for i in range(5):
            storage.record_turn(
                messages=[
                    {"role": "user", "content": f"Message {i}"},
                    {"role": "assistant", "content": f"Reply {i}"},
                ],
                final_answer=f"Reply {i}",
                tokens=50,
            )

        # Now simulate what _compact_session does: split, summarize, write
        from tsugite.daemon.memory import split_turns_for_compaction

        all_turns = [r for r in storage.load_records() if isinstance(r, Turn)]
        assert len(all_turns) == 5

        with patch("tsugite.daemon.memory._count_tokens", side_effect=lambda text, model: len(text) // 4):
            old_turns, recent_turns = split_turns_for_compaction(
                all_turns, "openai:gpt-4o-mini", retention_budget_tokens=100_000
            )

        # With huge budget, all fit -> no old turns
        assert old_turns == []
        assert recent_turns == all_turns

        # Now with tiny budget to force a split
        with patch("tsugite.daemon.memory._count_tokens", side_effect=lambda text, model: len(text) // 4):
            old_turns, recent_turns = split_turns_for_compaction(
                all_turns, "openai:gpt-4o-mini", retention_budget_tokens=1
            )

        assert len(old_turns) > 0
        assert len(recent_turns) >= 2

        # Create new session with summary + retained turns
        new_storage = SessionStorage.create(
            agent_name="test_agent",
            model="openai:gpt-4o",
            compacted_from="old_session",
            session_path=history_dir / "new_session.jsonl",
        )
        new_storage.record_compaction_summary(
            "## Current Task\nDiscussing messages",
            previous_turns=len(old_turns),
            retained_turns=len(recent_turns),
        )
        new_storage.write_turns(recent_turns)

        # Verify new session structure
        records = new_storage.load_records()
        summaries = [r for r in records if isinstance(r, CompactionSummary)]
        turns = [r for r in records if isinstance(r, Turn)]

        assert len(summaries) == 1
        assert summaries[0].retained_turns == len(recent_turns)
        assert len(turns) == len(recent_turns)

    def test_reconstruct_messages_with_retained_turns(self, tmp_path, monkeypatch):
        """reconstruct_messages should produce summary + retained turn messages in order."""
        history_dir = tmp_path / "history"
        history_dir.mkdir()
        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: history_dir)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test_machine")

        session_path = history_dir / "compacted.jsonl"
        storage = SessionStorage.create(
            agent_name="test_agent",
            model="openai:gpt-4o",
            compacted_from="prev_session",
            session_path=session_path,
        )

        storage.record_compaction_summary("## Current Task\nWorking on feature X", previous_turns=8, retained_turns=2)

        storage.record_turn(
            messages=[
                {"role": "user", "content": "Recent question"},
                {"role": "assistant", "content": "Recent answer"},
            ],
            final_answer="Recent answer",
        )
        storage.record_turn(
            messages=[
                {"role": "user", "content": "Latest question"},
                {"role": "assistant", "content": "Latest answer"},
            ],
            final_answer="Latest answer",
        )

        messages = reconstruct_messages(session_path)

        # Should have: summary user/assistant pair + 2 turns (2 msgs each) = 6 messages
        assert len(messages) == 6

        # First pair is the compaction summary
        assert "previous_conversation" in messages[0]["content"]
        assert "feature X" in messages[0]["content"]
        assert messages[1]["role"] == "assistant"

        # Then the retained turns
        assert messages[2]["content"] == "Recent question"
        assert messages[3]["content"] == "Recent answer"
        assert messages[4]["content"] == "Latest question"
        assert messages[5]["content"] == "Latest answer"
