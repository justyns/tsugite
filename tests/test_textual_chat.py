"""Tests for Textual chat UI functionality."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from tsugite.ui.textual_handler import TextualUIHandler
from tsugite.ui.base import UIEvent
from tsugite.ui.widgets import MessageList, StatusBar


class TestTextualUIHandler:
    """Test TextualUIHandler for event handling."""

    def test_handler_initialization(self):
        """Test that handler initializes with callbacks."""
        status_callback = Mock()
        tool_callback = Mock()

        handler = TextualUIHandler(
            on_status_change=status_callback,
            on_tool_call=tool_callback,
        )

        assert handler.on_status_change == status_callback
        assert handler.on_tool_call == tool_callback
        assert handler.current_tools == []

    def test_task_start_resets_tools(self):
        """Test that starting a task resets the tools list."""
        status_callback = Mock()
        handler = TextualUIHandler(on_status_change=status_callback)

        # Add some tools
        handler.current_tools = ["tool1", "tool2"]

        # Handle task start
        handler.handle_event(UIEvent.TASK_START, {"task": "Test task"})

        # Tools should be reset
        assert handler.current_tools == []
        status_callback.assert_called_once_with("Starting task...")

    def test_tool_call_tracking(self):
        """Test that tool calls are tracked."""
        tool_callback = Mock()
        handler = TextualUIHandler(on_tool_call=tool_callback)

        # Handle tool call
        handler.handle_event(UIEvent.TOOL_CALL, {"content": "Tool: read_file"})

        # Tool should be added
        assert "read_file" in handler.current_tools
        tool_callback.assert_called_once_with("read_file")

    def test_multiple_tool_calls(self):
        """Test that multiple tool calls are all tracked."""
        handler = TextualUIHandler()

        handler.handle_event(UIEvent.TOOL_CALL, {"content": "Tool: read_file"})
        handler.handle_event(UIEvent.TOOL_CALL, {"content": "Tool: write_file"})
        handler.handle_event(UIEvent.TOOL_CALL, {"content": "Tool: web_search"})

        assert handler.current_tools == ["read_file", "write_file", "web_search"]

    def test_clear_tools(self):
        """Test clearing tools list."""
        handler = TextualUIHandler()
        handler.current_tools = ["tool1", "tool2", "tool3"]

        handler.clear_tools()

        assert handler.current_tools == []

    def test_get_tools_used_returns_copy(self):
        """Test that get_tools_used returns a copy, not reference."""
        handler = TextualUIHandler()
        handler.current_tools = ["tool1", "tool2"]

        tools_copy = handler.get_tools_used()
        tools_copy.append("tool3")

        # Original should not be modified
        assert handler.current_tools == ["tool1", "tool2"]

    def test_streaming_chunks(self):
        """Test streaming chunk handling."""
        chunk_callback = Mock()
        handler = TextualUIHandler(on_stream_chunk=chunk_callback)

        handler.handle_event(UIEvent.STREAM_CHUNK, {"chunk": "Hello "})
        handler.handle_event(UIEvent.STREAM_CHUNK, {"chunk": "world"})

        assert handler.streaming_content == "Hello world"
        assert handler.is_streaming is True
        assert chunk_callback.call_count == 2

    def test_stream_complete(self):
        """Test streaming completion."""
        complete_callback = Mock()
        handler = TextualUIHandler(on_stream_complete=complete_callback)

        # Add some streaming content
        handler.streaming_content = "Some content"
        handler.is_streaming = True

        handler.handle_event(UIEvent.STREAM_COMPLETE, {})

        assert handler.streaming_content == ""
        assert handler.is_streaming is False
        complete_callback.assert_called_once()


class TestMessageListWidget:
    """Test MessageList widget."""

    def test_initialization(self):
        """Test widget initializes with empty messages."""
        widget = MessageList()

        assert widget.messages == []
        assert widget.can_focus is False

    def test_add_user_message(self):
        """Test adding a user message."""
        widget = MessageList()

        widget.add_message("user", "Hello!")

        assert len(widget.messages) == 1
        assert widget.messages[0]["type"] == "user"
        assert widget.messages[0]["content"] == "Hello!"

    def test_add_agent_message(self):
        """Test adding an agent message."""
        widget = MessageList()

        widget.add_message("agent", "Hi there!")

        assert len(widget.messages) == 1
        assert widget.messages[0]["type"] == "agent"
        assert widget.messages[0]["content"] == "Hi there!"

    def test_add_status_message(self):
        """Test adding a status message."""
        widget = MessageList()

        widget.add_message("status", "System ready")

        assert len(widget.messages) == 1
        assert widget.messages[0]["type"] == "status"

    def test_add_separator(self):
        """Test adding a separator."""
        widget = MessageList()

        widget.add_separator()

        assert len(widget.messages) == 1
        assert widget.messages[0]["type"] == "separator"

    def test_message_order(self):
        """Test that messages are added in order."""
        widget = MessageList()

        widget.add_message("user", "Question")
        widget.add_message("agent", "Answer")
        widget.add_separator()
        widget.add_message("status", "Done")

        assert len(widget.messages) == 4
        assert widget.messages[0]["content"] == "Question"
        assert widget.messages[1]["content"] == "Answer"
        assert widget.messages[2]["type"] == "separator"
        assert widget.messages[3]["content"] == "Done"

    def test_clear_messages(self):
        """Test clearing all messages."""
        widget = MessageList()

        widget.add_message("user", "Message 1")
        widget.add_message("agent", "Message 2")

        # Clear by setting to empty list
        widget.messages = []

        assert len(widget.messages) == 0


class TestStatusBarWidget:
    """Test StatusBar widget."""

    def test_initialization(self):
        """Test widget initializes with defaults."""
        widget = StatusBar()

        assert widget.status == "Ready"
        assert widget.tools_used == []
        assert widget.is_streaming is False
        assert widget.code_executing is None
        assert widget.can_focus is False

    def test_status_change(self):
        """Test changing status."""
        widget = StatusBar()

        widget.status = "Processing..."

        assert widget.status == "Processing..."

    def test_tools_used_tracking(self):
        """Test tracking tools used."""
        widget = StatusBar()

        widget.tools_used = ["read_file", "write_file"]

        assert len(widget.tools_used) == 2
        assert "read_file" in widget.tools_used

    def test_streaming_state(self):
        """Test streaming state."""
        widget = StatusBar()

        widget.is_streaming = True

        assert widget.is_streaming is True

    def test_code_execution_state(self):
        """Test code execution state."""
        widget = StatusBar()

        widget.code_executing = "print('hello')"

        assert widget.code_executing == "print('hello')"


class TestChatAppIntegration:
    """Integration tests for ChatApp."""

    @pytest.fixture
    def test_agent(self, tmp_path):
        """Create a test agent for chat."""
        agent_content = """---
name: test_textual_chat
model: ollama:qwen2.5-coder:7b
tools: []
max_steps: 3
---

You are a test assistant for Textual UI testing.

{{ user_prompt }}
"""
        agent_path = tmp_path / "test_textual_chat.md"
        agent_path.write_text(agent_content)
        return agent_path

    def test_app_initialization(self, test_agent):
        """Test that ChatApp initializes without errors."""
        from tsugite.ui.textual_chat import ChatApp

        app = ChatApp(
            agent_path=test_agent,
            model_override=None,
            max_history=50,
            stream=False,
        )

        assert app.agent_path == test_agent
        assert app.agent_name == "test_textual_chat"
        assert app.model == "ollama:qwen2.5-coder:7b"
        assert app.max_history == 50
        assert app.stream_enabled is False
        assert app.turn_count == 0

    def test_app_with_model_override(self, test_agent):
        """Test ChatApp with model override."""
        from tsugite.ui.textual_chat import ChatApp

        app = ChatApp(
            agent_path=test_agent,
            model_override="ollama:custom-model",
            max_history=10,
            stream=True,
        )

        assert app.model == "ollama:custom-model"
        assert app.max_history == 10
        assert app.stream_enabled is True

    @pytest.mark.asyncio
    async def test_slash_command_help(self, test_agent):
        """Test /help command handling."""
        from tsugite.ui.textual_chat import ChatApp

        async with ChatApp(
            agent_path=test_agent,
            model_override=None,
            max_history=50,
            stream=False,
        ).run_test() as pilot:
            # Wait for app to mount
            await pilot.pause()

            # Get message list
            message_list = pilot.app.query_one(MessageList)
            initial_count = len(message_list.messages)

            # Simulate /help command
            await pilot.app.handle_command("/help")
            await pilot.pause()

            # Should have added help messages
            assert len(message_list.messages) > initial_count

            # Check that help content was added
            messages_text = " ".join([msg.get("content", "") for msg in message_list.messages])
            assert "/help" in messages_text
            assert "/clear" in messages_text
            assert "/stats" in messages_text

    @pytest.mark.asyncio
    async def test_slash_command_clear(self, test_agent):
        """Test /clear command handling."""
        from tsugite.ui.textual_chat import ChatApp

        async with ChatApp(
            agent_path=test_agent,
            model_override=None,
            max_history=50,
            stream=False,
        ).run_test() as pilot:
            await pilot.pause()

            # Add some messages
            message_list = pilot.app.query_one(MessageList)
            message_list.add_message("user", "Test message")
            message_list.add_message("agent", "Test response")

            # Add some history to manager
            pilot.app.manager.add_turn("Test", "Response")

            assert len(message_list.messages) > 0
            assert len(pilot.app.manager.conversation_history) > 0

            # Clear
            await pilot.app.handle_command("/clear")
            await pilot.pause()

            # History should be cleared (only status message remains)
            assert len(pilot.app.manager.conversation_history) == 0
            # Messages list should only have the "cleared" status message
            assert any("cleared" in msg.get("content", "").lower() for msg in message_list.messages)

    @pytest.mark.asyncio
    async def test_slash_command_stats(self, test_agent):
        """Test /stats command handling."""
        from tsugite.ui.textual_chat import ChatApp

        async with ChatApp(
            agent_path=test_agent,
            model_override=None,
            max_history=50,
            stream=False,
        ).run_test() as pilot:
            await pilot.pause()

            # Add some history
            pilot.app.manager.add_turn("Question 1", "Answer 1", token_count=100)
            pilot.app.manager.add_turn("Question 2", "Answer 2", token_count=150)

            message_list = pilot.app.query_one(MessageList)
            initial_count = len(message_list.messages)

            # Get stats
            await pilot.app.handle_command("/stats")
            await pilot.pause()

            # Should have added stats messages
            assert len(message_list.messages) > initial_count

            # Check that stats were displayed
            messages_text = " ".join([msg.get("content", "") for msg in message_list.messages])
            assert "Total Turns: 2" in messages_text
            assert "250" in messages_text  # Total tokens

    @pytest.mark.asyncio
    async def test_ui_handler_creation(self, test_agent):
        """Test that UI handler is created with proper callbacks."""
        from tsugite.ui.textual_chat import ChatApp

        async with ChatApp(
            agent_path=test_agent,
            model_override=None,
            max_history=50,
            stream=False,
        ).run_test() as pilot:
            # After mount, ui_handler should exist
            assert pilot.app.ui_handler is not None
            assert pilot.app.manager is not None

            # Check callbacks are set
            assert pilot.app.ui_handler.on_status_change is not None
            assert pilot.app.ui_handler.on_tool_call is not None
            assert pilot.app.ui_handler.on_stream_chunk is not None
            assert pilot.app.ui_handler.on_stream_complete is not None


class TestRunTextualChat:
    """Test the run_textual_chat function."""

    @pytest.fixture
    def test_agent(self, tmp_path):
        """Create a test agent."""
        agent_content = """---
name: test_runner
model: ollama:qwen2.5-coder:7b
tools: []
---

Test agent for runner.

{{ user_prompt }}
"""
        agent_path = tmp_path / "test_runner.md"
        agent_path.write_text(agent_content)
        return agent_path

    def test_run_textual_chat_creates_app(self, test_agent):
        """Test that run_textual_chat creates ChatApp correctly."""
        from tsugite.ui.textual_chat import ChatApp

        # We can't actually run the app (it would block), but we can create it
        with patch.object(ChatApp, 'run') as mock_run:
            from tsugite.ui.textual_chat import run_textual_chat

            run_textual_chat(
                agent_path=test_agent,
                model_override="custom-model",
                max_history=25,
                stream=True,
            )

            # Verify run was called
            mock_run.assert_called_once()
