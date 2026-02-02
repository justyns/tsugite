"""Tests for history integration with agent runs."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tsugite.agent_runner.history_integration import (
    _extract_functions_called,
    save_run_to_history,
)


@pytest.fixture
def sample_agent_file(tmp_path):
    """Create a sample agent file."""
    agent_file = tmp_path / "test_agent.md"
    agent_file.write_text("""---
name: test_agent
model: openai:gpt-4o
---
Test agent
""")
    return agent_file


class TestSaveRunToHistory:
    """Tests for save_run_to_history function."""

    def test_save_run_to_history_success(self, tmp_path, sample_agent_file):
        """Test successful save of run to history."""
        with (
            patch("tsugite.config.load_config") as mock_config,
            patch("tsugite.history.storage.get_history_dir", return_value=tmp_path),
            patch("tsugite.history.storage.get_machine_name", return_value="test_machine"),
            patch("tsugite.md_agents.parse_agent_file") as mock_parse,
        ):
            mock_config.return_value = MagicMock(history_enabled=True)
            mock_agent = MagicMock()
            mock_agent.config = MagicMock(disable_history=False)
            mock_parse.return_value = mock_agent

            conv_id = save_run_to_history(
                agent_path=sample_agent_file,
                agent_name="test_agent",
                prompt="test prompt",
                result="test result",
                model="openai:gpt-4o",
            )

            assert conv_id is not None

            # Verify conversation file exists
            conv_file = tmp_path / f"{conv_id}.jsonl"
            assert conv_file.exists()

    def test_save_run_to_history_with_token_count(self, tmp_path, sample_agent_file):
        """Test save with token count metadata."""
        with (
            patch("tsugite.config.load_config") as mock_config,
            patch("tsugite.history.storage.get_history_dir", return_value=tmp_path),
            patch("tsugite.history.storage.get_machine_name", return_value="test_machine"),
            patch("tsugite.md_agents.parse_agent_file") as mock_parse,
        ):
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
                token_count=1234,
            )

            assert conv_id is not None

            # Verify tokens saved
            from tsugite.history import SessionStorage

            storage = SessionStorage.load(tmp_path / f"{conv_id}.jsonl")
            assert storage.total_tokens == 1234

    def test_save_run_to_history_with_cost(self, tmp_path, sample_agent_file):
        """Test save with cost metadata."""
        with (
            patch("tsugite.config.load_config") as mock_config,
            patch("tsugite.history.storage.get_history_dir", return_value=tmp_path),
            patch("tsugite.history.storage.get_machine_name", return_value="test_machine"),
            patch("tsugite.md_agents.parse_agent_file") as mock_parse,
        ):
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
                cost=0.05,
            )

            assert conv_id is not None

            # Verify cost saved
            from tsugite.history import SessionStorage

            storage = SessionStorage.load(tmp_path / f"{conv_id}.jsonl")
            assert storage.total_cost == 0.05

    def test_save_run_to_history_with_execution_steps(self, tmp_path, sample_agent_file):
        """Test save with execution steps."""
        with (
            patch("tsugite.config.load_config") as mock_config,
            patch("tsugite.history.storage.get_history_dir", return_value=tmp_path),
            patch("tsugite.history.storage.get_machine_name", return_value="test_machine"),
            patch("tsugite.md_agents.parse_agent_file") as mock_parse,
        ):
            mock_config.return_value = MagicMock(history_enabled=True)
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

    def test_save_run_to_history_history_disabled_config(self, tmp_path, sample_agent_file):
        """Test that save returns None when history is disabled in config."""
        with patch("tsugite.agent_runner.history_integration.load_config") as mock_config:
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
            patch("tsugite.history.storage.get_history_dir", return_value=tmp_path),
            patch("tsugite.history.storage.get_machine_name", return_value="test_machine"),
            patch("tsugite.md_agents.parse_agent_file") as mock_parse,
        ):
            mock_config.return_value = MagicMock(history_enabled=True)

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

            assert conv_id is not None

    def test_save_run_to_history_error_handling(self, tmp_path, sample_agent_file, capsys):
        """Test that errors are caught and None is returned."""
        with (
            patch("tsugite.config.load_config") as mock_config,
            patch("tsugite.history.storage.get_history_dir") as mock_history_dir,
        ):
            mock_config.return_value = MagicMock(history_enabled=True)

            # get_history_dir raises exception
            mock_history_dir.side_effect = Exception("Database error")

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
    """Tests for _extract_functions_called helper function."""

    def test_extract_functions_called_with_steps(self):
        """Test extracting tool calls from execution steps."""
        step1 = MagicMock()
        step1.tools_called = ["read_file", "write_file"]
        step2 = MagicMock()
        step2.tools_called = ["web_search"]

        tools = _extract_functions_called([step1, step2])

        # Should return sorted, unique list
        assert tools == ["read_file", "web_search", "write_file"]

    def test_extract_functions_called_empty_steps(self):
        """Test with empty step list."""
        tools = _extract_functions_called([])
        assert tools == []

    def test_extract_functions_called_no_tools(self):
        """Test with steps that have no tools_called attribute."""
        step1 = MagicMock(spec=[])  # No tools_called attribute
        step2 = MagicMock()
        step2.tools_called = []

        tools = _extract_functions_called([step1, step2])
        assert tools == []

    def test_extract_functions_called_duplicate_tools(self):
        """Test that duplicate tool names are deduplicated."""
        step1 = MagicMock()
        step1.tools_called = ["read_file", "write_file"]
        step2 = MagicMock()
        step2.tools_called = ["read_file", "web_search"]  # Duplicate "read_file"

        tools = _extract_functions_called([step1, step2])

        # Should only have unique tools
        assert tools == ["read_file", "web_search", "write_file"]

    def test_extract_functions_called_preserves_order(self):
        """Test that tools are returned in sorted order."""
        step = MagicMock()
        step.tools_called = ["zebra", "aardvark", "monkey"]

        tools = _extract_functions_called([step])

        # Should be alphabetically sorted
        assert tools == ["aardvark", "monkey", "zebra"]


class TestRunHistoryIntegration:
    """Integration tests for run history."""

    def test_run_creates_conversation(self, tmp_path, sample_agent_file):
        """Test that running an agent creates a conversation."""
        with (
            patch("tsugite.history.storage.get_history_dir", return_value=tmp_path),
            patch("tsugite.history.storage.get_machine_name", return_value="test_machine"),
            patch("tsugite.config.load_config") as mock_config,
            patch("tsugite.md_agents.parse_agent_file") as mock_parse,
        ):
            from tsugite.history import SessionStorage, Turn

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
            storage = SessionStorage.load(conv_file)
            records = storage.load_records()
            turns = [r for r in records if isinstance(r, Turn)]
            assert len(turns) == 1

    def test_run_saves_single_turn(self, tmp_path, sample_agent_file):
        """Test that run saves exactly one turn."""
        with (
            patch("tsugite.history.storage.get_history_dir", return_value=tmp_path),
            patch("tsugite.history.storage.get_machine_name", return_value="test_machine"),
            patch("tsugite.config.load_config") as mock_config,
            patch("tsugite.md_agents.parse_agent_file") as mock_parse,
        ):
            from tsugite.history import SessionStorage, Turn

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

            storage = SessionStorage.load(tmp_path / f"{conv_id}.jsonl")
            records = storage.load_records()
            turns = [r for r in records if isinstance(r, Turn)]

            assert len(turns) == 1
            assert turns[0].final_answer == "world"

    def test_run_metadata_saved(self, tmp_path, sample_agent_file):
        """Test that metadata is saved with run."""
        with (
            patch("tsugite.history.storage.get_history_dir", return_value=tmp_path),
            patch("tsugite.history.storage.get_machine_name", return_value="test_machine"),
            patch("tsugite.config.load_config") as mock_config,
            patch("tsugite.md_agents.parse_agent_file") as mock_parse,
        ):
            from tsugite.history import SessionStorage

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

            # Verify metadata was saved
            storage = SessionStorage.load(tmp_path / f"{conv_id}.jsonl")
            assert storage.agent == "test_agent"
            assert storage.model == "test_model"
            assert storage.turn_count == 1
            assert storage.total_tokens == 500
            assert storage.total_cost == 0.01

    def test_run_conversation_listable(self, tmp_path, sample_agent_file):
        """Test that saved run is listable from session files."""
        with (
            patch("tsugite.history.storage.get_history_dir", return_value=tmp_path),
            patch("tsugite.history.storage.get_machine_name", return_value="test_machine"),
            patch("tsugite.config.load_config") as mock_config,
            patch("tsugite.md_agents.parse_agent_file") as mock_parse,
        ):
            from tsugite.history import list_session_files

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

            # Query for session files
            files = list_session_files()

            assert len(files) == 1
            assert conv_id in str(files[0])
