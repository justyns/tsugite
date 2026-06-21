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

from tsugite_discord import DiscordAdapter, DiscordProgressHandler  # noqa: E402

from tsugite.events import (  # noqa: E402
    CodeExecutionEvent,
    ErrorEvent,
    FinalAnswerEvent,
    InfoEvent,
    LLMMessageEvent,
    LLMWaitProgressEvent,
    ObservationEvent,
    ReasoningContentEvent,
    StepStartEvent,
    ToolCallEvent,
    ToolResultEvent,
    WarningEvent,
)  # noqa: E402


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
async def test_progress_handler_turns():
    """Test progress handler with turn and execution events."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel, asyncio.get_running_loop())

    # First turn (use _handle_event_async directly for testing)
    await handler._handle_event_async(StepStartEvent(step=1, max_turns=10))
    assert len(channel.messages) == 1
    assert "Turn 1/10" in channel.messages[0].content
    assert "🤔 Working..." in channel.messages[0].content

    # Code execution
    await handler._handle_event_async(CodeExecutionEvent(code="print('hello')"))
    assert "Running code" in channel.messages[0].content
    assert "⚙️" in channel.messages[0].content

    # Observation marks previous complete
    await handler._handle_event_async(ObservationEvent(observation="hello", success=True))
    assert "✓" in channel.messages[0].content

    # Second turn (marks previous as complete)
    await handler._handle_event_async(StepStartEvent(step=2, max_turns=10))
    assert "Turn 2/10" in channel.messages[0].content

    # Final answer
    await handler._handle_event_async(FinalAnswerEvent(answer="Done!", turns=2, tokens=100, cost=0.01))
    assert "✅ Done (2 turns)" in channel.messages[0].content
    assert handler.done


@pytest.mark.asyncio
async def test_progress_handler_reasoning():
    """ReasoningContentEvent shows 'Reasoning' (matches web UI status_text)."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel, asyncio.get_running_loop())

    await handler._handle_event_async(ReasoningContentEvent(content="Thinking about the problem..."))
    assert len(channel.messages) == 1
    assert "Reasoning" in channel.messages[0].content
    assert "💭" in channel.messages[0].content


@pytest.mark.asyncio
async def test_progress_handler_thought_buffered_until_code():
    """LLMMessageEvent buffers content; CodeExecutionEvent flushes it as a standalone message."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel, asyncio.get_running_loop())

    await handler._handle_event_async(LLMMessageEvent(content="Let me consider..."))
    assert len(channel.messages) == 0

    await handler._handle_event_async(CodeExecutionEvent(code="x = 1"))
    thought_msgs = [m for m in channel.messages if "Let me consider" in m.content]
    assert len(thought_msgs) == 1
    assert thought_msgs[0].content.startswith("💭 ")


@pytest.mark.asyncio
async def test_progress_handler_thought_flushed_before_tool_call():
    """ToolCallEvent flushes a buffered thought too (tool-calling agents without code)."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel, asyncio.get_running_loop())

    await handler._handle_event_async(LLMMessageEvent(content="Need to grep first"))
    assert len(channel.messages) == 0

    await handler._handle_event_async(ToolCallEvent(tool_name="grep"))
    thought_msgs = [m for m in channel.messages if "Need to grep first" in m.content]
    assert len(thought_msgs) == 1


@pytest.mark.asyncio
async def test_progress_handler_thought_discarded_on_final_answer():
    """LLMMessageEvent followed by FinalAnswerEvent discards the buffer (avoids duplicating the answer)."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel, asyncio.get_running_loop())

    await handler._handle_event_async(LLMMessageEvent(content="Final reasoning"))
    await handler._handle_event_async(FinalAnswerEvent(answer="Final reasoning", turns=1, tokens=10, cost=0.0))

    thought_msgs = [m for m in channel.messages if "Final reasoning" in m.content]
    assert len(thought_msgs) == 0


@pytest.mark.asyncio
async def test_progress_handler_llm_message_empty_buffers_nothing():
    """Empty LLMMessageEvent neither sends nor buffers."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel, asyncio.get_running_loop())

    await handler._handle_event_async(LLMMessageEvent(content=""))
    await handler._handle_event_async(CodeExecutionEvent(code="x = 1"))

    assert all("💭" not in m.content for m in channel.messages)


@pytest.mark.asyncio
async def test_progress_handler_long_thought_truncated():
    """Thoughts exceeding Discord's 2000-char limit are truncated, not dropped."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel, asyncio.get_running_loop())

    long_text = "x" * 2500
    await handler._handle_event_async(LLMMessageEvent(content=long_text))
    await handler._handle_event_async(CodeExecutionEvent(code="y = 1"))

    thought_msgs = [m for m in channel.messages if m.content.startswith("💭 ")]
    assert len(thought_msgs) == 1
    assert len(thought_msgs[0].content) <= 2000
    assert "(truncated)" in thought_msgs[0].content


@pytest.mark.asyncio
async def test_progress_handler_tool_call_shows_tool_name():
    """ToolCallEvent appends a 'Tool: <name>' step."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel, asyncio.get_running_loop())

    await handler._handle_event_async(StepStartEvent(step=1, max_turns=10))
    await handler._handle_event_async(ToolCallEvent(tool_name="read_file", arguments={"path": "/foo"}))
    assert "Tool: read_file" in channel.messages[0].content
    assert "🔧" in channel.messages[0].content


@pytest.mark.asyncio
async def test_progress_handler_tool_result_marks_complete():
    """ToolResultEvent marks the in-flight tool step as completed."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel, asyncio.get_running_loop())

    await handler._handle_event_async(ToolCallEvent(tool_name="read_file"))
    content_before = channel.messages[0].content
    assert "✓" not in content_before.split("Tool: read_file", 1)[1].split("\n", 1)[0]

    await handler._handle_event_async(ToolResultEvent(tool_name="read_file", success=True))
    content_after = channel.messages[0].content
    assert "Tool: read_file ✓" in content_after


@pytest.mark.asyncio
async def test_progress_handler_multiple_tool_calls_per_turn():
    """Code agents may call multiple tools per turn; each gets its own line."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel, asyncio.get_running_loop())

    await handler._handle_event_async(StepStartEvent(step=1, max_turns=10))
    await handler._handle_event_async(CodeExecutionEvent(code="x"))
    await handler._handle_event_async(ToolCallEvent(tool_name="read_file"))
    await handler._handle_event_async(ToolCallEvent(tool_name="grep"))
    content = channel.messages[0].content
    assert "Running code" in content
    assert "Tool: read_file" in content
    assert "Tool: grep" in content


@pytest.mark.asyncio
async def test_progress_handler_llm_wait_updates_in_place():
    """Repeated LLMWaitProgressEvent updates the same step rather than appending."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel, asyncio.get_running_loop())

    await handler._handle_event_async(LLMWaitProgressEvent(elapsed_seconds=5))
    after_first = channel.messages[0].content
    assert "Waiting on LLM (5s)" in after_first
    assert "⏳" in after_first

    await handler._handle_event_async(LLMWaitProgressEvent(elapsed_seconds=12))
    after_second = channel.messages[0].content
    assert "Waiting on LLM (12s)" in after_second
    assert "Waiting on LLM (5s)" not in after_second
    assert after_second.count("⏳") == 1


@pytest.mark.asyncio
async def test_progress_handler_llm_wait_cleared_by_other_event():
    """A non-wait event ends the wait so the next wait appends fresh."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel, asyncio.get_running_loop())

    await handler._handle_event_async(LLMWaitProgressEvent(elapsed_seconds=5))
    await handler._handle_event_async(StepStartEvent(step=1, max_turns=10))
    await handler._handle_event_async(LLMWaitProgressEvent(elapsed_seconds=3))
    content = channel.messages[0].content
    assert content.count("⏳") == 2


@pytest.mark.asyncio
async def test_progress_handler_custom_header():
    """Custom header_text replaces the default '🤔 Working...' first line."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel, asyncio.get_running_loop(), header_text="🤔 #general · refactor-auth")

    await handler._handle_event_async(StepStartEvent(step=1, max_turns=10))
    content = channel.messages[0].content
    assert content.startswith("🤔 #general · refactor-auth")
    assert "🤔 Working..." not in content


@pytest.mark.asyncio
async def test_progress_handler_compacting_active():
    """compacting event renders 'Compacting history' with 📦."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel, asyncio.get_running_loop())

    handler._emit("compacting", {})
    await asyncio.sleep(0.05)
    content = channel.messages[0].content
    assert "Compacting history" in content
    assert "📦" in content


@pytest.mark.asyncio
async def test_progress_handler_compacting_waiting_distinct_label():
    """compacting_waiting renders 'Waiting for compaction' with ⌛, separate from active."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel, asyncio.get_running_loop())

    handler._emit("compacting_waiting", {})
    await asyncio.sleep(0.05)
    content = channel.messages[0].content
    assert "Waiting for compaction" in content
    assert "⌛" in content
    assert "Compacting history" not in content


@pytest.mark.asyncio
async def test_progress_handler_compacted_marks_either_complete():
    """compacted marks the in-flight compaction step done whether active or waiting."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel, asyncio.get_running_loop())

    handler._emit("compacting_waiting", {})
    await asyncio.sleep(0.05)
    handler._emit("compacted", {})
    await asyncio.sleep(0.05)
    content = channel.messages[0].content
    assert "Waiting for compaction ✓" in content


@pytest.mark.asyncio
async def test_progress_handler_warning_and_error():
    """Test progress handler with warning and error events."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel, asyncio.get_running_loop())

    await handler._handle_event_async(StepStartEvent(step=1, max_turns=10))
    await handler._handle_event_async(WarningEvent(message="Rate limited"))
    assert "⚠️" in channel.messages[0].content
    assert "Retrying" in channel.messages[0].content

    await handler._handle_event_async(ErrorEvent(error="Fatal error"))
    assert "❌" in channel.messages[0].content


@pytest.mark.asyncio
async def test_progress_handler_truncation():
    """Test progress handler truncates long step lists."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel, asyncio.get_running_loop())

    # Add 15 turns (more than MAX_DISPLAY_STEPS)
    for i in range(1, 16):
        await handler._handle_event_async(StepStartEvent(step=i, max_turns=20))

    # Check that only recent steps are shown
    content = channel.messages[0].content
    assert "earlier steps" in content
    assert "Turn 15/20" in content  # Last turn
    assert "Turn 1/20" not in content  # First turn should be truncated


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

    # Add a turn
    await handler._handle_event_async(StepStartEvent(step=1, max_turns=10))

    # Cleanup with failure
    await handler.cleanup(success=False)

    # Check error state
    assert "❌ Error after 1 turn" in channel.messages[0].content


@pytest.mark.asyncio
async def test_progress_handler_summary():
    """Test progress handler summary counts turns correctly."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel, asyncio.get_running_loop())

    # Add multiple turns
    await handler._handle_event_async(StepStartEvent(step=1, max_turns=10))
    await handler._handle_event_async(CodeExecutionEvent(code="x = 1"))
    await handler._handle_event_async(ObservationEvent(observation="done", success=True))
    await handler._handle_event_async(StepStartEvent(step=2, max_turns=10))
    await handler._handle_event_async(CodeExecutionEvent(code="x = 2"))
    await handler._handle_event_async(ObservationEvent(observation="done", success=True))
    await handler._handle_event_async(StepStartEvent(step=3, max_turns=10))

    # Final answer
    await handler._handle_event_async(FinalAnswerEvent(answer="Done!", turns=3, tokens=50, cost=0.005))

    # Check summary shows turn count
    content = channel.messages[0].content
    assert "✅ Done (3 turns)" in content


@pytest.mark.asyncio
async def test_progress_handler_info_event():
    """Test InfoEvent sends standalone message to channel."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel, asyncio.get_running_loop())

    await handler._handle_event_async(StepStartEvent(step=1, max_turns=10))
    assert len(channel.messages) == 1  # progress message

    await handler._handle_event_async(InfoEvent(message="Processing step 1 of 5..."))
    assert len(channel.messages) == 2  # progress + info message
    assert channel.messages[1].content == "Processing step 1 of 5..."


@pytest.mark.asyncio
async def test_progress_handler_info_event_empty():
    """Test empty InfoEvent is silently ignored."""
    channel = MockChannel()
    handler = DiscordProgressHandler(channel, asyncio.get_running_loop())

    await handler._handle_event_async(InfoEvent(message=""))
    assert len(channel.messages) == 0


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
