"""Tests for custom UI system."""

import pytest
from unittest.mock import MagicMock, patch
from io import StringIO
from pathlib import Path

from rich.console import Console
from smolagents.monitoring import LogLevel

from tsugite.custom_ui import (
    UIEvent,
    UIState,
    CustomUILogger,
    CustomUIHandler,
    custom_agent_ui,
    create_silent_logger,
)


class TestUIState:
    """Test UI state management."""

    def test_init_with_defaults(self):
        """Test UIState initialization with default values."""
        state = UIState()
        assert state.task is None
        assert state.current_step == 0
        assert state.total_steps is None
        assert state.current_action is None
        assert state.code_being_executed is None
        assert state.last_observation is None
        assert state.steps_history == []

    def test_init_with_values(self):
        """Test UIState initialization with custom values."""
        state = UIState(task="Test task", current_step=5)
        assert state.task == "Test task"
        assert state.current_step == 5
        assert state.steps_history == []


class TestCustomUILogger:
    """Test custom UI logger."""

    def test_init(self):
        """Test logger initialization."""
        ui_handler = MagicMock()
        logger = CustomUILogger(ui_handler)

        assert logger.ui_handler == ui_handler
        assert logger.level == LogLevel.OFF

    def test_log_task(self):
        """Test task logging."""
        ui_handler = MagicMock()
        logger = CustomUILogger(ui_handler)

        logger.log_task("Test task", "model-name", "Agent Title")

        ui_handler.handle_event.assert_called_once_with(
            UIEvent.TASK_START, {"task": "Test task", "model": "model-name", "title": "Agent Title"}
        )

    def test_log_rule_with_step(self):
        """Test step logging through log_rule."""
        ui_handler = MagicMock()
        logger = CustomUILogger(ui_handler)

        logger.log_rule("Step 3")

        ui_handler.handle_event.assert_called_once_with(UIEvent.STEP_START, {"step": 3, "title": "Step 3"})

    def test_log_code(self):
        """Test code execution logging."""
        ui_handler = MagicMock()
        logger = CustomUILogger(ui_handler)

        logger.log_code("Executing code", "print('hello')")

        ui_handler.handle_event.assert_called_once_with(
            UIEvent.CODE_EXECUTION, {"title": "Executing code", "code": "print('hello')"}
        )

    def test_log_tool_call(self):
        """Test tool call logging."""
        ui_handler = MagicMock()
        logger = CustomUILogger(ui_handler)

        logger.log("Calling tool: test_tool with args: {}")

        ui_handler.handle_event.assert_called_once_with(
            UIEvent.TOOL_CALL, {"content": "Calling tool: test_tool with args: {}"}
        )

    def test_log_observation(self):
        """Test observation logging."""
        ui_handler = MagicMock()
        logger = CustomUILogger(ui_handler)

        logger.log("Observations: Test observation")

        ui_handler.handle_event.assert_called_once_with(UIEvent.OBSERVATION, {"observation": "Test observation"})

    def test_log_final_answer(self):
        """Test final answer logging."""
        ui_handler = MagicMock()
        logger = CustomUILogger(ui_handler)

        logger.log("Final answer: 42")

        ui_handler.handle_event.assert_called_once_with(UIEvent.FINAL_ANSWER, {"answer": "42"})

    def test_log_markdown_llm_message(self):
        """Test LLM message logging through log_markdown."""
        ui_handler = MagicMock()
        logger = CustomUILogger(ui_handler)

        logger.log_markdown("I need to analyze this task", "Output message of the LLM:", LogLevel.DEBUG)

        ui_handler.handle_event.assert_called_once_with(
            UIEvent.LLM_MESSAGE,
            {"content": "I need to analyze this task", "title": "Output message of the LLM:", "level": LogLevel.DEBUG},
        )

    def test_log_execution_logs(self):
        """Test execution logs logging."""
        ui_handler = MagicMock()
        logger = CustomUILogger(ui_handler)

        logger.log("Execution logs: print('hello world')")

        ui_handler.handle_event.assert_called_once_with(
            UIEvent.EXECUTION_LOGS, {"content": "Execution logs: print('hello world')", "level": LogLevel.INFO}
        )


class TestCustomUIHandler:
    """Test custom UI handler."""

    def test_init(self):
        """Test handler initialization."""
        console = Console(file=StringIO())
        handler = CustomUIHandler(console)

        assert handler.console == console
        assert handler.state.task is None
        assert handler.show_code is True
        assert handler.show_observations is True

    def test_handle_task_start(self):
        """Test handling task start event."""
        console = Console(file=StringIO())
        handler = CustomUIHandler(console)

        handler.handle_event(UIEvent.TASK_START, {"task": "Test task", "model": "test-model", "title": "Test"})

        assert handler.state.task == "Test task"
        assert handler.state.current_step == 0
        assert handler.state.steps_history == []

        # Check output was generated
        output = console.file.getvalue()
        assert "Test task" in output

    def test_handle_step_start(self):
        """Test handling step start event."""
        console = Console(file=StringIO())
        handler = CustomUIHandler(console)

        handler.handle_event(UIEvent.STEP_START, {"step": 2})

        assert handler.state.current_step == 2
        assert len(handler.state.steps_history) == 1
        assert handler.state.steps_history[0]["step"] == 2
        assert handler.state.steps_history[0]["status"] == "in_progress"

    def test_handle_code_execution_with_show(self):
        """Test handling code execution with display enabled."""
        console = Console(file=StringIO())
        handler = CustomUIHandler(console, show_code=True)

        handler.handle_event(UIEvent.CODE_EXECUTION, {"code": 'print("hello")'})

        assert handler.state.code_being_executed == 'print("hello")'
        assert handler.state.current_action == "Executing code..."

        # Check output was generated
        output = console.file.getvalue()
        assert 'print("hello")' in output or "Executing Code" in output

    def test_handle_code_execution_without_show(self):
        """Test handling code execution with display disabled."""
        console = Console(file=StringIO())
        handler = CustomUIHandler(console, show_code=False)

        handler.handle_event(UIEvent.CODE_EXECUTION, {"code": 'print("hello")'})

        assert handler.state.code_being_executed == 'print("hello")'
        assert handler.state.current_action == "Executing code..."

        # Check no code output was generated
        output = console.file.getvalue()
        assert 'print("hello")' not in output

    def test_handle_observation_with_show(self):
        """Test handling observation with display enabled."""
        console = Console(file=StringIO())
        handler = CustomUIHandler(console, show_observations=True)

        # Add a step first
        handler.handle_event(UIEvent.STEP_START, {"step": 1})

        handler.handle_event(UIEvent.OBSERVATION, {"observation": "Test observation"})

        assert handler.state.last_observation == "Test observation"
        assert handler.state.steps_history[-1]["status"] == "completed"

        # Check output was generated
        output = console.file.getvalue()
        assert "Test observation" in output

    def test_handle_final_answer(self):
        """Test handling final answer event."""
        console = Console(file=StringIO())
        handler = CustomUIHandler(console)

        handler.handle_event(UIEvent.FINAL_ANSWER, {"answer": "42"})

        # Check output was generated
        output = console.file.getvalue()
        assert "42" in output
        assert "Final Answer" in output

    def test_handle_llm_message_with_show(self):
        """Test handling LLM message with display enabled."""
        console = Console(file=StringIO())
        handler = CustomUIHandler(console, show_llm_messages=True)

        handler.handle_event(
            UIEvent.LLM_MESSAGE, {"content": "I need to solve this step by step", "title": "Output message of the LLM:"}
        )

        output = console.file.getvalue()
        assert "I need to solve this step by step" in output
        assert "Agent Reasoning" in output

    def test_handle_llm_message_without_show(self):
        """Test handling LLM message with display disabled."""
        console = Console(file=StringIO())
        handler = CustomUIHandler(console, show_llm_messages=False)

        handler.handle_event(
            UIEvent.LLM_MESSAGE, {"content": "I need to solve this step by step", "title": "Output message of the LLM:"}
        )

        output = console.file.getvalue()
        assert "I need to solve this step by step" not in output

    def test_handle_execution_result(self):
        """Test handling execution result event."""
        console = Console(file=StringIO())
        handler = CustomUIHandler(console, show_execution_results=True)

        handler.handle_event(
            UIEvent.EXECUTION_RESULT, {"content": "Execution logs:\nprint statement output\nOut: Hello World"}
        )

        output = console.file.getvalue()
        assert "Hello World" in output or "Output:" in output

    def test_handle_execution_result_filters_none(self):
        """Test that None outputs are filtered out."""
        console = Console(file=StringIO())
        handler = CustomUIHandler(console, show_execution_results=True)

        handler.handle_event(UIEvent.EXECUTION_RESULT, {"content": "Out: None"})

        output = console.file.getvalue()
        # Should NOT contain "Output: None"
        assert "Output:" not in output
        assert "None" not in output

    def test_handle_execution_result_filters_null(self):
        """Test that null outputs are filtered out."""
        console = Console(file=StringIO())
        handler = CustomUIHandler(console, show_execution_results=True)

        handler.handle_event(UIEvent.EXECUTION_RESULT, {"content": "Out: null"})

        output = console.file.getvalue()
        # Should NOT contain "Output: null"
        assert "Output:" not in output

    def test_progress_context(self):
        """Test progress context manager."""
        console = Console(file=StringIO())
        handler = CustomUIHandler(console)

        with handler.progress_context():
            assert handler.progress is not None
            assert handler.task_id is not None

            # Test progress update
            handler.update_progress("New description")

        # After exiting context, progress should be stopped
        assert handler.progress is not None  # Still exists but stopped


class TestCustomAgentUI:
    """Test custom agent UI context manager."""

    def test_custom_agent_ui_context(self):
        """Test custom UI context manager."""
        console = Console(file=StringIO())

        with custom_agent_ui(console, show_progress=False) as logger:
            assert isinstance(logger, CustomUILogger)
            assert logger.level == LogLevel.OFF

    def test_custom_agent_ui_with_progress(self):
        """Test custom UI with progress enabled."""
        console = Console(file=StringIO())

        with patch("tsugite.custom_ui.CustomUIHandler.progress_context") as mock_context:
            mock_context.return_value.__enter__ = MagicMock()
            mock_context.return_value.__exit__ = MagicMock()

            with custom_agent_ui(console, show_progress=True) as logger:
                assert isinstance(logger, CustomUILogger)

            # Progress context should have been used
            mock_context.assert_called_once()

    def test_custom_agent_ui_flags(self):
        """Test custom UI with different flags."""
        console = Console(file=StringIO())

        with custom_agent_ui(console, show_code=False, show_observations=False, show_progress=False) as logger:
            assert logger.ui_handler.show_code is False
            assert logger.ui_handler.show_observations is False


class TestSilentLogger:
    """Test silent logger creation."""

    def test_create_silent_logger(self):
        """Test creating a completely silent logger."""
        from smolagents.monitoring import AgentLogger

        logger = create_silent_logger()

        assert isinstance(logger, AgentLogger)
        assert logger.level == LogLevel.OFF

        # Test that logger doesn't produce output
        # Since it writes to /dev/null, we can't capture output
        # but we can verify it doesn't raise errors
        logger.log("Test message")  # Should not raise
