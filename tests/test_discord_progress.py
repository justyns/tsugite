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

from tsugite.daemon.adapters.discord import DiscordAdapter, DiscordProgressHandler  # noqa: E402
from tsugite.events import FinalAnswerEvent, ReasoningContentEvent, ToolCallEvent  # noqa: E402


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

    async def typing(self):
        """Mock typing indicator."""
        self.typing_triggered += 1

    async def trigger_typing(self):
        """Mock trigger_typing (alias)."""
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
    handler = DiscordProgressHandler(channel, asyncio.get_running_loop())

    # First tool call (use _handle_event_async directly for testing)
    await handler._handle_event_async(ToolCallEvent(tool="read_file", args={"path": "test.txt"}))
    assert len(channel.messages) == 1
    assert "read_file" in channel.messages[0].content
    assert "🤔 Working..." in channel.messages[0].content

    # Second tool call (marks previous as complete)
    await handler._handle_event_async(ToolCallEvent(tool="write_file", args={"path": "out.txt", "content": "test"}))
    assert len(channel.messages[0].edit_history) >= 2
    assert "✓" in channel.messages[0].content  # Previous tool marked complete
    assert "write_file" in channel.messages[0].content

    # Final answer
    await handler._handle_event_async(FinalAnswerEvent(answer="Done!", turns=2, tokens=100, cost=0.01))
    assert "✅ Done" in channel.messages[0].content
    assert handler.done


@pytest.mark.asyncio
async def test_progress_handler_reasoning():
    """Test progress handler with reasoning events."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel, asyncio.get_running_loop())

    # Reasoning event (use _handle_event_async directly for testing)
    await handler._handle_event_async(ReasoningContentEvent(content="Thinking about the problem..."))
    assert len(channel.messages) == 1
    assert "Thinking" in channel.messages[0].content
    assert "💭" in channel.messages[0].content


@pytest.mark.asyncio
async def test_progress_handler_emoji_mapping():
    """Test emoji mapping for different tool types."""
    handler = DiscordProgressHandler(MockChannel(), asyncio.get_running_loop())

    assert handler._get_tool_emoji("search_web") == "🔍"
    assert handler._get_tool_emoji("read_file") == "📖"
    assert handler._get_tool_emoji("write_file") == "✍️"
    assert handler._get_tool_emoji("execute_code") == "⚙️"
    assert handler._get_tool_emoji("unknown_tool") == "🔧"


@pytest.mark.asyncio
async def test_progress_handler_truncation():
    """Test progress handler truncates long tool lists."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel, asyncio.get_running_loop())

    # Add 15 tool calls (more than MAX_DISPLAY_STEPS)
    for i in range(15):
        await handler._handle_event_async(ToolCallEvent(tool=f"tool_{i}", args={}))

    # Check that only recent steps are shown
    content = channel.messages[0].content
    assert "earlier steps" in content
    assert "tool_14" in content  # Last tool
    assert "tool_0" not in content  # First tool should be truncated


@pytest.mark.asyncio
async def test_progress_handler_typing_loop():
    """Test typing loop keeps indicator active."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel, asyncio.get_running_loop())

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
    handler = DiscordProgressHandler(channel, asyncio.get_running_loop())

    # Add a tool call
    await handler._handle_event_async(ToolCallEvent(tool="failing_tool", args={}))

    # Cleanup with failure
    await handler.cleanup(success=False)

    # Check error state
    assert "❌ Error" in channel.messages[0].content
    assert "failing_tool" in channel.messages[0].content


@pytest.mark.asyncio
async def test_progress_handler_summary():
    """Test progress handler summary counts tools correctly."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel, asyncio.get_running_loop())

    # Add multiple tool calls (some repeated)
    await handler._handle_event_async(ToolCallEvent(tool="read_file", args={}))
    await handler._handle_event_async(ToolCallEvent(tool="read_file", args={}))
    await handler._handle_event_async(ToolCallEvent(tool="write_file", args={}))
    await handler._handle_event_async(ToolCallEvent(tool="read_file", args={}))

    # Final answer
    await handler._handle_event_async(FinalAnswerEvent(answer="Done!", turns=1, tokens=50, cost=0.005))

    # Check summary shows counts
    content = channel.messages[0].content
    assert "✅ Done" in content
    assert "4 steps" in content  # Total count
    assert "read_file ×3" in content  # Most used tool with count


class TestCodeBlockChunking:
    """Tests for code block aware message chunking."""

    @pytest.fixture
    def adapter(self):
        """Create adapter instance for testing chunking method."""
        adapter = object.__new__(DiscordAdapter)
        return adapter

    def test_short_message_no_split(self, adapter):
        """Short message should not be split."""
        text = "Hello world"
        chunks = adapter._split_respecting_code_blocks(text, 2000)
        assert chunks == ["Hello world"]

    def test_long_text_splits_on_newlines(self, adapter):
        """Long text without code blocks splits on newlines."""
        lines = ["Line " + str(i) for i in range(100)]
        text = "\n".join(lines)
        chunks = adapter._split_respecting_code_blocks(text, 200)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 200

    def test_code_block_kept_whole(self, adapter):
        """Code block that fits should not be split."""
        text = "Before\n```python\ndef foo():\n    pass\n```\nAfter"
        chunks = adapter._split_respecting_code_blocks(text, 2000)
        assert len(chunks) == 1
        assert "```python" in chunks[0]
        assert "```" in chunks[0]

    def test_code_block_starts_new_chunk(self, adapter):
        """Code block that doesn't fit with text starts new chunk."""
        prefix = "x" * 100
        code = "```python\ndef foo():\n    pass\n```"
        text = prefix + "\n" + code
        # limit=120 means: prefix (100) fits, but prefix+code (135) doesn't
        # code alone (35) fits, so code should start new chunk
        chunks = adapter._split_respecting_code_blocks(text, 120)
        assert len(chunks) >= 2
        code_chunk = [c for c in chunks if "```python" in c][0]
        assert code_chunk.startswith("```python")
        assert code_chunk.endswith("```")

    def test_long_code_block_split_with_continuation(self, adapter):
        """Code block exceeding limit gets split with continuation markers."""
        lines = [f"    line{i} = {i}" for i in range(50)]
        code = "```python\n" + "\n".join(lines) + "\n```"
        chunks = adapter._split_respecting_code_blocks(code, 300)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.startswith("```python")
            assert chunk.endswith("```")
            assert len(chunk) <= 300

    def test_language_identifier_preserved(self, adapter):
        """Language identifier preserved when splitting code blocks."""
        lines = [f"const x{i} = {i};" for i in range(50)]
        code = "```javascript\n" + "\n".join(lines) + "\n```"
        chunks = adapter._split_respecting_code_blocks(code, 300)
        for chunk in chunks:
            assert chunk.startswith("```javascript")

    def test_multiple_code_blocks(self, adapter):
        """Multiple code blocks handled correctly."""
        text = "Text1\n```python\ncode1\n```\nText2\n```bash\ncode2\n```\nText3"
        chunks = adapter._split_respecting_code_blocks(text, 2000)
        assert len(chunks) == 1
        assert "```python" in chunks[0]
        assert "```bash" in chunks[0]

    def test_empty_chunks_skipped(self, adapter):
        """Empty chunks should not appear in output."""
        text = "\n\n\nHello\n\n\n"
        chunks = adapter._split_respecting_code_blocks(text, 2000)
        for chunk in chunks:
            assert chunk.strip()
