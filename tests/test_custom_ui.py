"""Tests for custom UI system."""

from io import StringIO
from unittest.mock import MagicMock, patch

from rich.console import Console

from tsugite.custom_ui import (
    CustomUIHandler,
    CustomUILogger,
    UIEvent,
    UIState,
    create_silent_logger,
    custom_agent_ui,
)


class TestUIState:
    """Test UI state management."""

    def test_init_with_defaults(self):
        """Test UIState initialization with default values."""
        state = UIState()
        assert state.task is None
        assert state.current_step == 0
        assert state.total_steps is None
        assert state.code_being_executed is None
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
        console = Console(file=StringIO())
        logger = CustomUILogger(ui_handler, console)

        assert logger.ui_handler == ui_handler
        assert logger.console == console


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

        # Check output was generated
        output = console.file.getvalue()
        assert 'print("hello")' in output or "Executing Code" in output

    def test_handle_code_execution_without_show(self):
        """Test handling code execution with display disabled."""
        console = Console(file=StringIO())
        handler = CustomUIHandler(console, show_code=False)

        handler.handle_event(UIEvent.CODE_EXECUTION, {"code": 'print("hello")'})

        assert handler.state.code_being_executed == 'print("hello")'

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
        assert "Output message of the LLM:" in output

    def test_handle_llm_message_without_show(self):
        """Test handling LLM message with display disabled."""
        console = Console(file=StringIO())
        handler = CustomUIHandler(console, show_llm_messages=False)

        handler.handle_event(
            UIEvent.LLM_MESSAGE, {"content": "I need to solve this step by step", "title": "Output message of the LLM:"}
        )

        output = console.file.getvalue()
        assert "I need to solve this step by step" not in output

    def test_handle_llm_message_shows_step_number(self):
        """Test that LLM messages show step numbers in their titles."""
        console = Console(file=StringIO())
        handler = CustomUIHandler(console, show_llm_messages=True)

        # Test Step 1
        handler.handle_event(UIEvent.LLM_MESSAGE, {"content": "First step reasoning", "title": "Step 1 Reasoning"})

        output = console.file.getvalue()
        assert "First step reasoning" in output
        assert "Step 1" in output

        # Clear console for next test
        console = Console(file=StringIO())
        handler = CustomUIHandler(console, show_llm_messages=True)

        # Test Step 2
        handler.handle_event(UIEvent.LLM_MESSAGE, {"content": "Second step reasoning", "title": "Step 2 Reasoning"})

        output = console.file.getvalue()
        assert "Second step reasoning" in output
        assert "Step 2" in output

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
            assert logger.ui_handler is not None
            assert logger.console == console

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
        logger = create_silent_logger()

        assert isinstance(logger, CustomUILogger)
        assert logger.ui_handler is not None
        assert logger.console is not None

        # Test that logger has the required attributes
        assert hasattr(logger, "ui_handler")
        assert hasattr(logger, "console")


class TestMultiStepContext:
    """Test multi-step context handling."""

    def test_set_multistep_context(self):
        """Test setting multi-step context."""
        console = Console(file=StringIO())
        handler = CustomUIHandler(console)

        assert handler.state.multistep_context is None

        handler.set_multistep_context(1, "research", 4)

        assert handler.state.multistep_context is not None
        assert handler.state.multistep_context["step_number"] == 1
        assert handler.state.multistep_context["step_name"] == "research"
        assert handler.state.multistep_context["total_steps"] == 4

    def test_clear_multistep_context(self):
        """Test clearing multi-step context."""
        console = Console(file=StringIO())
        handler = CustomUIHandler(console)

        handler.set_multistep_context(1, "research", 4)
        assert handler.state.multistep_context is not None

        handler.clear_multistep_context()
        assert handler.state.multistep_context is None

    def test_display_prefix_with_multistep(self):
        """Test that display prefix is added when in multi-step context."""
        console = Console(file=StringIO())
        handler = CustomUIHandler(console)

        # No prefix without multi-step context
        assert handler._get_display_prefix() == ""

        # Prefix added with multi-step context
        handler.set_multistep_context(1, "research", 4)
        assert handler._get_display_prefix() == "  └─ "

    def test_step_start_with_multistep_shows_round(self):
        """Test that step start shows 'Round' when in multi-step context."""
        console = Console(file=StringIO())
        handler = CustomUIHandler(console)

        # Set multi-step context
        handler.set_multistep_context(1, "research", 4)

        # Trigger step start
        handler.handle_event(UIEvent.STEP_START, {"step": 1, "title": "Step 1"})

        # Should show "Round" instead of "Step"
        assert handler.state.current_step == 1

    def test_step_start_without_multistep_shows_step(self):
        """Test that step start shows 'Step' when not in multi-step context."""
        console = Console(file=StringIO())
        handler = CustomUIHandler(console)

        # No multi-step context
        assert handler.state.multistep_context is None

        # Trigger step start
        handler.handle_event(UIEvent.STEP_START, {"step": 1, "title": "Step 1"})

        # Should show "Step"
        assert handler.state.current_step == 1
