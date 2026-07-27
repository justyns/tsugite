"""Tests for the tsugite-pty terminal-output context provider.

The provider reaches the live PtyManager + TerminalSessionStore through the same
``set_terminal_runtime`` seam the @terminal tools use, so these tests wire fake
manager/store objects into that seam rather than spawning real PTYs.
"""

from __future__ import annotations

import pytest
from tsugite_pty import context as pty_ctx
from tsugite_pty import tools as terminal_tools
from tsugite_pty.terminal_store import TerminalSession

from tsugite.context import get_context_provider, reset_context_providers


class _FakeProc:
    def __init__(self, buffer: bytes):
        self.buffer = buffer


class _FakeManager:
    def __init__(self, procs: dict[str, _FakeProc]):
        self._procs = procs

    def get(self, terminal_id: str):
        return self._procs.get(terminal_id)


class _FakeStore:
    def __init__(self, terminals: list[TerminalSession], logs: dict[str, object] | None = None):
        self._by_id = {t.id: t for t in terminals}
        self._logs = logs or {}

    def list_all(self) -> list[TerminalSession]:
        return list(self._by_id.values())

    def get(self, terminal_id: str):
        return self._by_id.get(terminal_id)

    def log_path(self, terminal_id: str):
        return self._logs.get(terminal_id, "/nonexistent/does-not-exist.log")


@pytest.fixture
def wire():
    """Wire fake (manager, store) into the tools seam; drop them on teardown."""

    def _wire(manager, store):
        terminal_tools.set_terminal_runtime(manager, store, None)

    try:
        yield _wire
    finally:
        terminal_tools.set_terminal_runtime(None, None, None)


def _term(tid: str, cmd: str, session: str) -> TerminalSession:
    return TerminalSession(id=tid, cmd=cmd, parent_session_id=session)


def test_choices_lists_all_terminals_session_own_first(wire):
    store = _FakeStore(
        [
            _term("term-1", "htop", "sess-a"),
            _term("term-2", "psql", "sess-a"),
            _term("term-3", "vim", "sess-b"),
        ]
    )
    wire(_FakeManager({}), store)

    choices = pty_ctx.terminal_choices({"session_id": "sess-a"})

    # Every terminal is pickable; this session's own come before the rest.
    assert {c.value for c in choices} == {"term-1", "term-2", "term-3"}
    assert {choices[0].value, choices[1].value} == {"term-1", "term-2"}
    assert choices[2].value == "term-3"


def test_choices_includes_unparented_terminals(wire):
    # The bug: a terminal opened from the terminals view has no parent session, so
    # the old strict ``== session_id`` filter hid it and the submenu was empty.
    store = _FakeStore([_term("term-x", "bash", None)])
    wire(_FakeManager({}), store)

    choices = pty_ctx.terminal_choices({"session_id": "sess-a"})

    assert [c.value for c in choices] == ["term-x"]


def test_choices_empty_when_runtime_unset(wire):
    wire(None, None)
    assert pty_ctx.terminal_choices({"session_id": "sess-a"}) == []


def test_capture_returns_decoded_buffer_item(wire):
    store = _FakeStore([_term("term-1", "htop", "sess-a")])
    manager = _FakeManager({"term-1": _FakeProc(b"cpu 42%\nmem 30%\n")})
    wire(manager, store)

    items = pty_ctx.capture_terminal("term-1", {"session_id": "sess-a"})

    assert len(items) == 1
    assert items[0].key == "terminal:term-1"
    assert items[0].label == "htop"
    assert items[0].value == "cpu 42%\nmem 30%"


def test_capture_unknown_terminal_returns_empty(wire):
    wire(_FakeManager({}), _FakeStore([]))
    assert pty_ctx.capture_terminal("term-nope", {"session_id": "sess-a"}) == []


def test_capture_none_arg_returns_empty(wire):
    wire(_FakeManager({}), _FakeStore([_term("term-1", "htop", "sess-a")]))
    assert pty_ctx.capture_terminal(None, {"session_id": "sess-a"}) == []


def test_capture_tail_caps_recent_output(wire):
    buffer = b"A" * 3000 + b"B" * 2000
    store = _FakeStore([_term("term-1", "yes", "sess-a")])
    wire(_FakeManager({"term-1": _FakeProc(buffer)}), store)

    items = pty_ctx.capture_terminal("term-1", {"session_id": "sess-a"})

    assert items[0].value == "B" * 2000


def test_capture_falls_back_to_persisted_log_after_eviction(wire, tmp_path):
    log = tmp_path / "term-1.log"
    log.write_bytes(b"persisted output\n")
    store = _FakeStore([_term("term-1", "make", "sess-a")], logs={"term-1": log})
    wire(_FakeManager({}), store)  # no live proc -> replay the log

    items = pty_ctx.capture_terminal("term-1", {"session_id": "sess-a"})

    assert items[0].value == "persisted output"


def test_registers_terminal_menu_provider():
    import importlib

    reset_context_providers()
    importlib.reload(pty_ctx)

    provider = get_context_provider("terminal")
    assert provider is not None
    assert provider.in_menu
    assert provider.choices is not None
    assert (provider.label, provider.icon) == ("Terminal output", "term")
