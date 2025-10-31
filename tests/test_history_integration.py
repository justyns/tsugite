"""Tests for history integration with agent runs."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from tsugite.agent_runner.history_integration import (
    _extract_tool_calls,
    save_run_to_history,
)


class TestSaveRunToHistory:
    """Tests for save_run_to_history function."""

    def test_save_run_to_history_success(self, tmp_path, sample_agent_file):
        """Test successful save of run to history."""
        with (
            patch("tsugite.config.load_config") as mock_config,
            patch("tsugite.ui.chat_history.start_conversation") as mock_start,
            patch("tsugite.ui.chat_history.save_chat_turn") as mock_save_turn,
            patch("tsugite.md_agents.parse_agent_file") as mock_parse,
        ):
            # Setup mocks
            mock_config.return_value = MagicMock(history_enabled=True)
            mock_start.return_value = "test_conv_id"

            mock_agent = MagicMock()
            mock_agent.config = MagicMock(disable_history=False)
            mock_parse.return_value = mock_agent

            # Call save_run_to_history
            conv_id = save_run_to_history(
                agent_path=sample_agent_file,
                agent_name="test_agent",
                prompt="test prompt",
                result="test result",
                model="openai:gpt-4o",
            )

            # Verify it returned conversation ID
            assert conv_id == "test_conv_id"

            # Verify start_conversation was called
            mock_start.assert_called_once()
            call_kwargs = mock_start.call_args[1]
            assert call_kwargs["agent_name"] == "test_agent"
            assert call_kwargs["model"] == "openai:gpt-4o"

            # Verify save_chat_turn was called
            mock_save_turn.assert_called_once()
            call_kwargs = mock_save_turn.call_args[1]
            assert call_kwargs["conversation_id"] == "test_conv_id"
            assert call_kwargs["user_message"] == "test prompt"
            assert call_kwargs["agent_response"] == "test result"

    def test_save_run_to_history_with_token_count(self, tmp_path, sample_agent_file):
        """Test save with token count metadata."""
        with (
            patch("tsugite.config.load_config") as mock_config,
            patch("tsugite.ui.chat_history.start_conversation") as mock_start,
            patch("tsugite.ui.chat_history.save_chat_turn") as mock_save_turn,
            patch("tsugite.md_agents.parse_agent_file") as mock_parse,
        ):
            mock_config.return_value = MagicMock(history_enabled=True)
            mock_start.return_value = "test_conv_id"
            mock_agent = MagicMock()
            mock_agent.config = MagicMock(disable_history=False)
            mock_parse.return_value = mock_agent

            conv_id = save_run_to_history(
                agent_path=sample_agent_file,
                agent_name="test_agent",
                prompt="prompt",
                result="result",
                model="test_model",
                token_count=1234,
            )

            assert conv_id is not None
            call_kwargs = mock_save_turn.call_args[1]
            assert call_kwargs["token_count"] == 1234

    def test_save_run_to_history_with_cost(self, tmp_path, sample_agent_file):
        """Test save with cost metadata."""
        with (
            patch("tsugite.config.load_config") as mock_config,
            patch("tsugite.ui.chat_history.start_conversation") as mock_start,
            patch("tsugite.ui.chat_history.save_chat_turn") as mock_save_turn,
            patch("tsugite.md_agents.parse_agent_file") as mock_parse,
        ):
            mock_config.return_value = MagicMock(history_enabled=True)
            mock_start.return_value = "test_conv_id"
            mock_agent = MagicMock()
            mock_agent.config = MagicMock(disable_history=False)
            mock_parse.return_value = mock_agent

            conv_id = save_run_to_history(
                agent_path=sample_agent_file,
                agent_name="test_agent",
                prompt="prompt",
                result="result",
                model="test_model",
                cost=0.05,
            )

            assert conv_id is not None
            call_kwargs = mock_save_turn.call_args[1]
            assert call_kwargs["cost"] == 0.05

    def test_save_run_to_history_with_execution_steps(self, tmp_path, sample_agent_file):
        """Test save with execution steps."""
        with (
            patch("tsugite.config.load_config") as mock_config,
            patch("tsugite.ui.chat_history.start_conversation") as mock_start,
            patch("tsugite.ui.chat_history.save_chat_turn") as mock_save_turn,
            patch("tsugite.md_agents.parse_agent_file") as mock_parse,
        ):
            mock_config.return_value = MagicMock(history_enabled=True)
            mock_start.return_value = "test_conv_id"
            mock_agent = MagicMock()
            mock_agent.config = MagicMock(disable_history=False)
            mock_parse.return_value = mock_agent

            # Create mock execution steps with tools_called
            step1 = MagicMock()
            step1.tools_called = ["read_file", "write_file"]
            step2 = MagicMock()
            step2.tools_called = ["web_search"]

            conv_id = save_run_to_history(
                agent_path=sample_agent_file,
                agent_name="test_agent",
                prompt="prompt",
                result="result",
                model="test_model",
                execution_steps=[step1, step2],
            )

            assert conv_id is not None
            call_kwargs = mock_save_turn.call_args[1]
            # Tools should be extracted and sorted
            assert set(call_kwargs["tool_calls"]) == {"read_file", "web_search", "write_file"}

    def test_save_run_to_history_history_disabled_config(self, tmp_path, sample_agent_file):
        """Test that save returns None when history is disabled in config."""
        with patch("tsugite.config.load_config") as mock_config:
            mock_config.return_value = MagicMock(history_enabled=False)

            conv_id = save_run_to_history(
                agent_path=sample_agent_file,
                agent_name="test_agent",
                prompt="prompt",
                result="result",
                model="test_model",
            )

            assert conv_id is None

    def test_save_run_to_history_agent_disable_history(self, tmp_path, sample_agent_file):
        """Test that save returns None when agent has disable_history=True."""
        with (
            patch("tsugite.config.load_config") as mock_config,
            patch("tsugite.md_agents.parse_agent_file") as mock_parse,
        ):
            mock_config.return_value = MagicMock(history_enabled=True)

            # Agent has disable_history=True
            mock_agent = MagicMock()
            mock_agent.config = MagicMock(disable_history=True)
            mock_parse.return_value = mock_agent

            conv_id = save_run_to_history(
                agent_path=sample_agent_file,
                agent_name="test_agent",
                prompt="prompt",
                result="result",
                model="test_model",
            )

            assert conv_id is None

    def test_save_run_to_history_invalid_agent_path(self, tmp_path):
        """Test that invalid agent path is handled gracefully."""
        with (
            patch("tsugite.config.load_config") as mock_config,
            patch("tsugite.ui.chat_history.start_conversation") as mock_start,
            patch("tsugite.ui.chat_history.save_chat_turn") as mock_save_turn,
            patch("tsugite.md_agents.parse_agent_file") as mock_parse,
        ):
            mock_config.return_value = MagicMock(history_enabled=True)
            mock_start.return_value = "test_conv_id"

            # parse_agent_file raises exception
            mock_parse.side_effect = Exception("Invalid agent file")

            # Should still save (assumes history enabled if can't parse agent)
            conv_id = save_run_to_history(
                agent_path=Path("/nonexistent/agent.md"),
                agent_name="test_agent",
                prompt="prompt",
                result="result",
                model="test_model",
            )

            assert conv_id == "test_conv_id"
            mock_save_turn.assert_called_once()

    def test_save_run_to_history_error_handling(self, tmp_path, sample_agent_file, capsys):
        """Test that errors are caught and None is returned."""
        with (
            patch("tsugite.config.load_config") as mock_config,
            patch("tsugite.ui.chat_history.start_conversation") as mock_start,
        ):
            mock_config.return_value = MagicMock(history_enabled=True)

            # start_conversation raises exception
            mock_start.side_effect = Exception("Database error")

            conv_id = save_run_to_history(
                agent_path=sample_agent_file,
                agent_name="test_agent",
                prompt="prompt",
                result="result",
                model="test_model",
            )

            # Should return None and print warning
            assert conv_id is None

            # Verify warning was printed to stderr
            captured = capsys.readouterr()
            assert "Warning: Failed to save run to history" in captured.err
            assert "Database error" in captured.err


class TestExtractToolCalls:
    """Tests for _extract_tool_calls helper function."""

    def test_extract_tool_calls_with_steps(self):
        """Test extracting tool calls from execution steps."""
        step1 = MagicMock()
        step1.tools_called = ["read_file", "write_file"]
        step2 = MagicMock()
        step2.tools_called = ["web_search"]

        tools = _extract_tool_calls([step1, step2])

        # Should return sorted, unique list
        assert tools == ["read_file", "web_search", "write_file"]

    def test_extract_tool_calls_empty_steps(self):
        """Test with empty step list."""
        tools = _extract_tool_calls([])
        assert tools == []

    def test_extract_tool_calls_no_tools(self):
        """Test with steps that have no tools_called attribute."""
        step1 = MagicMock(spec=[])  # No tools_called attribute
        step2 = MagicMock()
        step2.tools_called = []

        tools = _extract_tool_calls([step1, step2])
        assert tools == []

    def test_extract_tool_calls_duplicate_tools(self):
        """Test that duplicate tool names are deduplicated."""
        step1 = MagicMock()
        step1.tools_called = ["read_file", "write_file"]
        step2 = MagicMock()
        step2.tools_called = ["read_file", "web_search"]  # Duplicate "read_file"

        tools = _extract_tool_calls([step1, step2])

        # Should only have unique tools
        assert tools == ["read_file", "web_search", "write_file"]

    def test_extract_tool_calls_preserves_order(self):
        """Test that tools are returned in sorted order."""
        step = MagicMock()
        step.tools_called = ["zebra", "aardvark", "monkey"]

        tools = _extract_tool_calls([step])

        # Should be alphabetically sorted
        assert tools == ["aardvark", "monkey", "zebra"]


class TestRunHistoryIntegration:
    """Integration tests for run history."""

    def test_run_creates_conversation(self, tmp_path, sample_agent_file):
        """Test that running an agent creates a conversation."""
        with (
            patch("tsugite.history.storage.get_history_dir", return_value=tmp_path),
            patch("tsugite.history.index.get_history_dir", return_value=tmp_path),
            patch("tsugite.ui.chat_history.get_machine_name", return_value="test_machine"),
            patch("tsugite.config.load_config") as mock_config,
            patch("tsugite.md_agents.parse_agent_file") as mock_parse,
        ):
            from tsugite.history import load_conversation

            mock_config.return_value = MagicMock(history_enabled=True)
            mock_agent = MagicMock()
            mock_agent.config = MagicMock(disable_history=False)
            mock_parse.return_value = mock_agent

            conv_id = save_run_to_history(
                agent_path=sample_agent_file,
                agent_name="test_agent",
                prompt="test prompt",
                result="test result",
                model="test_model",
            )

            assert conv_id is not None

            # Verify conversation file exists
            conv_file = tmp_path / f"{conv_id}.jsonl"
            assert conv_file.exists()

            # Verify conversation can be loaded
            turns = load_conversation(conv_id)
            assert len(turns) == 2  # metadata + turn

    def test_run_saves_single_turn(self, tmp_path, sample_agent_file):
        """Test that run saves exactly one turn."""
        with (
            patch("tsugite.history.storage.get_history_dir", return_value=tmp_path),
            patch("tsugite.history.index.get_history_dir", return_value=tmp_path),
            patch("tsugite.ui.chat_history.get_machine_name", return_value="test_machine"),
            patch("tsugite.config.load_config") as mock_config,
            patch("tsugite.md_agents.parse_agent_file") as mock_parse,
        ):
            from tsugite.history import load_conversation

            mock_config.return_value = MagicMock(history_enabled=True)
            mock_agent = MagicMock()
            mock_agent.config = MagicMock(disable_history=False)
            mock_parse.return_value = mock_agent

            conv_id = save_run_to_history(
                agent_path=sample_agent_file,
                agent_name="test_agent",
                prompt="hello",
                result="world",
                model="test_model",
            )

            turns = load_conversation(conv_id)

            # Should have metadata + 1 turn
            assert len(turns) == 2
            assert turns[0].type == "metadata"
            assert turns[1].type == "turn"
            assert turns[1].user == "hello"
            assert turns[1].assistant == "world"

    def test_run_index_updated(self, tmp_path, sample_agent_file):
        """Test that index is updated with run."""
        with (
            patch("tsugite.history.storage.get_history_dir", return_value=tmp_path),
            patch("tsugite.history.index.get_history_dir", return_value=tmp_path),
            patch("tsugite.ui.chat_history.get_machine_name", return_value="test_machine"),
            patch("tsugite.config.load_config") as mock_config,
            patch("tsugite.md_agents.parse_agent_file") as mock_parse,
        ):
            from tsugite.history import get_conversation_metadata

            mock_config.return_value = MagicMock(history_enabled=True)
            mock_agent = MagicMock()
            mock_agent.config = MagicMock(disable_history=False)
            mock_parse.return_value = mock_agent

            conv_id = save_run_to_history(
                agent_path=sample_agent_file,
                agent_name="test_agent",
                prompt="prompt",
                result="result",
                model="test_model",
                token_count=500,
                cost=0.01,
            )

            # Verify index was updated
            metadata = get_conversation_metadata(conv_id)
            assert metadata is not None
            assert metadata.agent == "test_agent"
            assert metadata.model == "test_model"
            assert metadata.turn_count == 1
            assert metadata.total_tokens == 500
            assert metadata.total_cost == 0.01

    def test_run_conversation_queryable(self, tmp_path, sample_agent_file):
        """Test that saved run is queryable from index."""
        with (
            patch("tsugite.history.storage.get_history_dir", return_value=tmp_path),
            patch("tsugite.history.index.get_history_dir", return_value=tmp_path),
            patch("tsugite.ui.chat_history.get_machine_name", return_value="test_machine"),
            patch("tsugite.config.load_config") as mock_config,
            patch("tsugite.md_agents.parse_agent_file") as mock_parse,
        ):
            from tsugite.history import query_index

            mock_config.return_value = MagicMock(history_enabled=True)
            mock_agent = MagicMock()
            mock_agent.config = MagicMock(disable_history=False)
            mock_parse.return_value = mock_agent

            conv_id = save_run_to_history(
                agent_path=sample_agent_file,
                agent_name="search_agent",
                prompt="prompt",
                result="result",
                model="test_model",
            )

            # Query for this agent
            results = query_index(agent="search_agent")

            assert len(results) == 1
            assert results[0]["conversation_id"] == conv_id
            assert results[0]["agent"] == "search_agent"
