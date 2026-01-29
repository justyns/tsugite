"""Tests for Discord progress handler."""

import asyncio
import sys
from unittest.mock import MagicMock

import pytest

# Mock discord modules BEFORE importing anything that needs them
# This is necessary because pytest discovers and imports test files before fixtures run
_original_discord = sys.modules.get("discord")
_original_discord_ext = sys.modules.get("discord.ext")
_original_discord_ext_commands = sys.modules.get("discord.ext.commands")

sys.modules["discord"] = MagicMock()
sys.modules["discord.ext"] = MagicMock()
sys.modules["discord.ext.commands"] = MagicMock()

from tsugite.daemon.adapters.discord import DiscordProgressHandler
from tsugite.events import FinalAnswerEvent, ReasoningContentEvent, ToolCallEvent


@pytest.fixture(autouse=True, scope="module")
def cleanup_discord_mocks():
    """Cleanup discord mocks after module tests complete."""
    yield
    # Restore original modules after all tests in this module
    if _original_discord is not None:
        sys.modules["discord"] = _original_discord
    else:
        sys.modules.pop("discord", None)

    if _original_discord_ext is not None:
        sys.modules["discord.ext"] = _original_discord_ext
    else:
        sys.modules.pop("discord.ext", None)

    if _original_discord_ext_commands is not None:
        sys.modules["discord.ext.commands"] = _original_discord_ext_commands
    else:
        sys.modules.pop("discord.ext.commands", None)


class MockChannel:
    """Mock Discord channel for testing."""

    def __init__(self):
        self.messages = []
        self.edits = []
        self.typing_triggered = 0

    async def send(self, content):
        """Mock send."""
        msg = MockMessage(content)
        self.messages.append(msg)
        return msg

    async def trigger_typing(self):
        """Mock trigger_typing."""
        self.typing_triggered += 1


class MockMessage:
    """Mock Discord message."""

    def __init__(self, content):
        self.content = content
        self.edit_history = [content]

    async def edit(self, content):
        """Mock edit."""
        self.content = content
        self.edit_history.append(content)


@pytest.mark.asyncio
async def test_progress_handler_tool_calls():
    """Test progress handler with tool calls."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel)

    # First tool call
    await handler.handle_event(ToolCallEvent(tool="read_file", args={"path": "test.txt"}))
    assert len(channel.messages) == 1
    assert "read_file" in channel.messages[0].content
    assert "🤔 Working..." in channel.messages[0].content

    # Second tool call (marks previous as complete)
    await handler.handle_event(ToolCallEvent(tool="write_file", args={"path": "out.txt", "content": "test"}))
    assert len(channel.messages[0].edit_history) >= 2
    assert "✓" in channel.messages[0].content  # Previous tool marked complete
    assert "write_file" in channel.messages[0].content

    # Final answer
    await handler.handle_event(FinalAnswerEvent(answer="Done!", turns=2, tokens=100, cost=0.01))
    assert "✅ Done" in channel.messages[0].content
    assert handler.done


@pytest.mark.asyncio
async def test_progress_handler_reasoning():
    """Test progress handler with reasoning events."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel)

    # Reasoning event
    await handler.handle_event(ReasoningContentEvent(content="Thinking about the problem..."))
    assert len(channel.messages) == 1
    assert "Thinking" in channel.messages[0].content
    assert "💭" in channel.messages[0].content


@pytest.mark.asyncio
async def test_progress_handler_emoji_mapping():
    """Test emoji mapping for different tool types."""
    handler = DiscordProgressHandler(MockChannel())

    assert handler._get_tool_emoji("search_web") == "🔍"
    assert handler._get_tool_emoji("read_file") == "📖"
    assert handler._get_tool_emoji("write_file") == "✍️"
    assert handler._get_tool_emoji("execute_code") == "⚙️"
    assert handler._get_tool_emoji("unknown_tool") == "🔧"


@pytest.mark.asyncio
async def test_progress_handler_truncation():
    """Test progress handler truncates long tool lists."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel)

    # Add 15 tool calls (more than MAX_DISPLAY_STEPS)
    for i in range(15):
        await handler.handle_event(ToolCallEvent(tool=f"tool_{i}", args={}))

    # Check that only recent steps are shown
    content = channel.messages[0].content
    assert "earlier steps" in content
    assert "tool_14" in content  # Last tool
    assert "tool_0" not in content  # First tool should be truncated


@pytest.mark.asyncio
async def test_progress_handler_typing_loop():
    """Test typing loop keeps indicator active."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel)

    # Start typing loop
    await handler.start_typing_loop()

    # Wait for a few typing triggers
    await asyncio.sleep(0.5)
    assert channel.typing_triggered >= 1

    # Cleanup should cancel typing
    await handler.cleanup(success=True)
    initial_count = channel.typing_triggered

    # Wait and verify typing stopped
    await asyncio.sleep(0.5)
    assert channel.typing_triggered == initial_count  # No new triggers


@pytest.mark.asyncio
async def test_progress_handler_error_state():
    """Test progress handler shows error state."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel)

    # Add a tool call
    await handler.handle_event(ToolCallEvent(tool="failing_tool", args={}))

    # Cleanup with failure
    await handler.cleanup(success=False)

    # Check error state
    assert "❌ Error" in channel.messages[0].content
    assert "failing_tool" in channel.messages[0].content


@pytest.mark.asyncio
async def test_progress_handler_summary():
    """Test progress handler summary counts tools correctly."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel)

    # Add multiple tool calls (some repeated)
    await handler.handle_event(ToolCallEvent(tool="read_file", args={}))
    await handler.handle_event(ToolCallEvent(tool="read_file", args={}))
    await handler.handle_event(ToolCallEvent(tool="write_file", args={}))
    await handler.handle_event(ToolCallEvent(tool="read_file", args={}))

    # Final answer
    await handler.handle_event(FinalAnswerEvent(answer="Done!", turns=1, tokens=50, cost=0.005))

    # Check summary shows counts
    content = channel.messages[0].content
    assert "✅ Done" in content
    assert "4 steps" in content  # Total count
    assert "read_file ×3" in content  # Most used tool with count
