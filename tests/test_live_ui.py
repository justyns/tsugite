"""Tests for LiveUIHandler (three-region scrollback + persistent status footer)."""

from io import StringIO

import pytest
from rich.console import Console

from tsugite.events import (
    CostSummaryEvent,
    FinalAnswerEvent,
    LLMWaitProgressEvent,
    ObservationEvent,
    StepStartEvent,
    TaskStartEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from tsugite.ui.live import LiveUIHandler
from tsugite.ui.plain import PlainUIHandler


class _FakeClock:
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def fake_clock():
    return _FakeClock()


@pytest.fixture
def handler(fake_clock):
    return LiveUIHandler(clock=fake_clock)


def test_status_initial_state(handler, fake_clock):
    """Fresh handler shows turn 0, idle, zero tokens/cost, elapsed 0:00."""
    text = handler._render_status_text()
    assert "Turn 0" in text
    assert "idle" in text
    assert "tokens 0" in text
    assert "$0.00" in text
    assert "0:00" in text


def test_step_start_advances_turn(handler):
    """StepStartEvent updates the displayed turn number."""
    handler.handle_event(StepStartEvent(step=3))
    assert "Turn 3" in handler._render_status_text()


def test_tool_call_sets_current_tool(handler):
    """ToolCallEvent populates the tool name and a short desc from the first arg."""
    handler.handle_event(ToolCallEvent(tool_name="read_file", arguments={"path": "/tmp/x"}))
    text = handler._render_status_text()
    assert "read_file" in text
    assert "/tmp/x" in text
    assert "idle" not in text


def test_tool_result_returns_to_idle(handler):
    """ToolResultEvent clears the active tool so the footer reads idle again."""
    handler.handle_event(ToolCallEvent(tool_name="http_request", arguments={"url": "https://x"}))
    handler.handle_event(ToolResultEvent(tool_name="http_request", success=True, result_summary="200"))
    text = handler._render_status_text()
    assert "idle" in text
    assert "http_request" not in text


def test_cost_summary_updates_tokens_and_cost(handler):
    """CostSummaryEvent updates cumulative tokens and cost."""
    handler.handle_event(CostSummaryEvent(tokens=25000, cost=0.02, model="claude-opus-4-7"))
    text = handler._render_status_text()
    assert "tokens 25.0k" in text or "tokens 25k" in text
    assert "$0.02" in text


def test_llm_wait_progress_shows_waiting(handler):
    """LLMWaitProgressEvent renders a waiting label with elapsed seconds."""
    handler.handle_event(LLMWaitProgressEvent(elapsed_seconds=12))
    text = handler._render_status_text()
    assert "waiting" in text.lower()
    assert "12" in text


def test_elapsed_formats_mm_ss(handler, fake_clock):
    """Elapsed timer uses M:SS format and advances with the injected clock."""
    fake_clock.advance(95)
    text = handler._render_status_text()
    assert "1:35" in text

    fake_clock.advance(60 * 4 + 5)
    text = handler._render_status_text()
    assert "5:40" in text


def test_short_desc_truncated_to_terminal_width(handler):
    """Very long tool args get truncated so the footer fits the terminal width."""
    long_path = "/very/long/path/" + "a" * 500
    handler.handle_event(ToolCallEvent(tool_name="read_file", arguments={"path": long_path}))
    text = handler._render_status_text(width=80)
    assert len(text) <= 80
    assert "read_file" in text


def test_scrollback_matches_plain_output():
    """LiveUIHandler's scrollback emits identical lines to PlainUIHandler given the same events.

    DRY contract: the only difference between modes is the status footer; the log content
    above it must come from the same code path.
    """
    plain_buf = StringIO()
    plain_console = Console(file=plain_buf, force_terminal=False, no_color=True, width=80)
    plain = PlainUIHandler()
    plain.console = plain_console
    plain.show_panels = False

    live_buf = StringIO()
    live_console = Console(file=live_buf, force_terminal=False, no_color=True, width=80)
    live = LiveUIHandler()
    live.console = live_console
    live.show_panels = False

    events = [
        TaskStartEvent(task="do the thing", model="claude-opus-4-7"),
        StepStartEvent(step=1),
        ToolCallEvent(tool_name="read_file", arguments={"path": "/tmp/x"}),
        ObservationEvent(observation="contents", tool="read_file", success=True),
        ToolResultEvent(tool_name="read_file", success=True, result_summary="ok"),
        FinalAnswerEvent(answer="done"),
    ]
    for ev in events:
        plain.handle_event(ev)
        live.handle_event(ev)

    plain_lines = [line.rstrip() for line in plain_buf.getvalue().splitlines() if line.strip()]
    live_lines = [line.rstrip() for line in live_buf.getvalue().splitlines() if line.strip()]
    assert plain_lines == live_lines


def test_show_reasoning_propagates(handler):
    """When show_reasoning=True, the underlying show_llm_messages is enabled."""
    reasoning_handler = LiveUIHandler(show_reasoning=True)
    assert reasoning_handler.show_llm_messages is True

    quiet_handler = LiveUIHandler(show_reasoning=False)
    assert quiet_handler.show_llm_messages is False


def test_live_context_returns_context_manager(handler):
    """live_context() yields a context manager that can be entered and exited."""
    with handler.live_context() as ctx:
        assert ctx is None or ctx is not None  # just ensure no exception


def test_progress_context_is_noop(handler):
    """progress_context() must not start Rich's Progress when Live owns the screen."""
    with handler.progress_context():
        assert handler.progress is None


def test_handle_event_threadsafe_smoke(handler):
    """Driving many events does not raise. Smoke test for the lock and dispatch."""
    for i in range(1, 20):
        handler.handle_event(StepStartEvent(step=i))
        handler.handle_event(ToolCallEvent(tool_name=f"tool_{i}", arguments={"i": i}))
        handler.handle_event(ToolResultEvent(tool_name=f"tool_{i}", success=True, result_summary="ok"))
    text = handler._render_status_text()
    assert "Turn 19" in text


def test_terminal_short_footer_collapses(handler):
    """When terminal height is small the status panel falls back to a single line."""
    handler.console.size = (80, 8)
    panel = handler._render_status_panel()
    assert panel.height == 1


def test_terminal_tall_footer_three_lines(handler):
    """When terminal height has room, status panel uses up to 3 rows."""
    handler.console.size = (80, 40)
    panel = handler._render_status_panel()
    assert panel.height == 3
