"""Context menu provider: attach a terminal's recent output as chat context.

Registered under the ``tsugite.context_providers`` entry point (``terminal``). It
reuses the daemon-wired PtyManager + TerminalSessionStore - the very singletons
the @terminal tools drive - via ``get_terminal_runtime``, so nothing new has to
be wired: ``choices`` lists the current session's terminals and ``capture``
snapshots the picked terminal's recent output.
"""

from __future__ import annotations

from tsugite.attachments.base import Attachment
from tsugite.context import ContextChoice, ContextProvider, register_context_provider
from tsugite_pty.tools import get_terminal_runtime

# A tail of recent output is enough context; the live ring buffer is larger.
_MAX_VALUE_CHARS = 2000


def _terminal_label(terminal) -> str:
    """A short human label for a terminal (its command, else its id)."""
    return (terminal.cmd or "").strip() or terminal.id


def terminal_choices(context: dict) -> list[ContextChoice]:
    """The user's terminals as pickable submenu options, the current session's own
    first, then the rest, each newest-first.

    A terminal opened from the terminals view is not parented to the chat (only
    agent-spawned terminals record a ``parent_session_id``), so filtering strictly
    by the session would leave the submenu empty in the common case. Listing all of
    them, session-first, is what makes "attach a terminal's output" actually work.
    """
    _manager, store, _cb = get_terminal_runtime()
    if store is None:
        return []
    session_id = context.get("session_id")
    terminals = sorted(store.list_all(), key=lambda t: t.created_at, reverse=True)
    own = [t for t in terminals if t.parent_session_id == session_id]
    rest = [t for t in terminals if t.parent_session_id != session_id]
    return [ContextChoice(value=t.id, label=_terminal_label(t)) for t in own + rest]


def _recent_output(manager, store, terminal) -> str:
    """Recent output for a terminal: the live ring buffer while the PTY is up,
    else the output log persisted on exit, else the one-line last_line snapshot."""
    proc = manager.get(terminal.id) if manager is not None else None
    if proc is not None:
        raw = proc.buffer
    else:
        try:
            raw = store.log_path(terminal.id).read_bytes()
        except OSError:
            raw = (terminal.last_line or "").encode("utf-8", errors="replace")
    return raw.decode("utf-8", errors="replace")


def capture_terminal(arg: str | None, context: dict) -> list[Attachment]:
    """Snapshot the picked terminal's recent output as a single context item.

    ``arg`` is the terminal id from ``terminal_choices``. An unknown id or an
    empty buffer attaches nothing.
    """
    manager, store, _cb = get_terminal_runtime()
    if store is None or not arg:
        return []
    terminal = store.get(arg)
    if terminal is None:
        return []
    text = _recent_output(manager, store, terminal).strip()
    if not text:
        return []
    return [Attachment.context(key=f"terminal:{arg}", label=_terminal_label(terminal), value=text[-_MAX_VALUE_CHARS:])]


register_context_provider(
    ContextProvider(
        key="terminal",
        label="Terminal output",
        icon="term",
        choices=terminal_choices,
        capture=capture_terminal,
    )
)
