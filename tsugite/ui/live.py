"""Live UI handler: plain scrollback above, persistent status footer below.

Three-region layout per the cleaner CLI run UX feature: scrollback log lines
flow normally above a 1-3 line status panel pinned at the bottom of the
terminal that always shows the current turn, active tool, cumulative
tokens/cost, and elapsed time.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Optional

from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from tsugite.console import get_stderr_console
from tsugite.events import (
    BaseEvent,
    CostSummaryEvent,
    LLMWaitProgressEvent,
    StepStartEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from tsugite.ui.plain import PlainUIHandler


@dataclass
class _StatusState:
    """Mutable state rendered into the persistent footer.

    Turn number lives on the inherited ``UIState.current_step`` and is not duplicated here.
    """

    current_tool: Optional[str] = None
    short_desc: str = ""
    tokens: int = 0
    cost: float = 0.0
    wait_elapsed: Optional[int] = None
    start_time: float = field(default_factory=time.monotonic)


class LiveUIHandler(PlainUIHandler):
    """Plain-style scrollback plus a persistent Rich Live status footer."""

    def __init__(
        self,
        show_reasoning: bool = False,
        clock: Optional[Callable[[], float]] = None,
    ):
        super().__init__(show_reasoning=show_reasoning)
        self.console = get_stderr_console(no_color=False)
        self._clock = clock or time.monotonic
        self._status = _StatusState(start_time=self._clock())

    def handle_event(self, event: BaseEvent) -> None:
        super().handle_event(event)
        self._update_status_state(event)

    def _update_status_state(self, event: BaseEvent) -> None:
        if isinstance(event, StepStartEvent):
            self._status.wait_elapsed = None
        elif isinstance(event, ToolCallEvent):
            self._status.current_tool = event.tool_name
            self._status.short_desc = self._format_arguments(event.arguments)
            self._status.wait_elapsed = None
        elif isinstance(event, ToolResultEvent):
            if self._status.current_tool == event.tool_name:
                self._status.current_tool = None
                self._status.short_desc = ""
        elif isinstance(event, CostSummaryEvent):
            if event.tokens is not None:
                self._status.tokens = event.tokens
            if event.cost is not None:
                self._status.cost = event.cost
        elif isinstance(event, LLMWaitProgressEvent):
            self._status.wait_elapsed = event.elapsed_seconds

    @staticmethod
    def _format_arguments(arguments: dict[str, Any]) -> str:
        if not arguments:
            return ""
        first_key = next(iter(arguments))
        value = arguments[first_key]
        if isinstance(value, (dict, list)):
            return f"{first_key}=..."
        return str(value)

    def _format_tokens(self, tokens: int) -> str:
        if tokens >= 1000:
            return f"tokens {tokens / 1000:.1f}k"
        return f"tokens {tokens}"

    def _format_elapsed(self) -> str:
        elapsed = max(0, int(self._clock() - self._status.start_time))
        minutes, seconds = divmod(elapsed, 60)
        return f"{minutes}:{seconds:02d}"

    def _render_status_text(self, width: Optional[int] = None) -> str:
        """Render the status as a single line. Used by tests and as the compact form."""
        if self._status.wait_elapsed is not None:
            action = f"waiting on LLM ({self._status.wait_elapsed}s)"
        elif self._status.current_tool:
            desc = self._status.short_desc
            action = f"{self._status.current_tool} {desc}".rstrip()
        else:
            action = "idle"

        parts = [
            f"Turn {self.state.current_step}",
            action,
            self._format_tokens(self._status.tokens),
            f"${self._status.cost:.2f}",
            self._format_elapsed(),
        ]
        text = " | ".join(parts)
        if width is not None and len(text) > width:
            text = text[: max(0, width - 1)] + "…"
        return text

    def _render_status_panel(self) -> Panel:
        height = getattr(self.console.size, "height", 24)
        width = getattr(self.console.size, "width", None)
        compact = height < 10
        body = Text(self._render_status_text(width=width), style="dim")
        return Panel(body, padding=(0, 1), height=1 if compact else 3, border_style="dim")

    def __rich__(self) -> Panel:
        """Rich render hook: called by ``Live`` on every auto-refresh tick."""
        return self._render_status_panel()

    @contextmanager
    def progress_context(self) -> Generator[None, None, None]:
        """No-op: the Rich Progress spinner cannot coexist with the Live footer."""
        self.progress = None
        yield

    @contextmanager
    def live_context(self) -> Generator[None, None, None]:
        """Open the persistent status footer. Scrollback log lines scroll above it.

        ``Live`` calls ``self.__rich__`` on each auto-refresh tick (default 4Hz),
        so state mutations from ``handle_event`` surface within ~250ms without
        per-event manual updates.
        """
        self._status.start_time = self._clock()
        with Live(self, console=self.console, refresh_per_second=4, transient=True):
            yield
