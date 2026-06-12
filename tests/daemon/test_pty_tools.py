"""End-to-end tests for the @terminal agent tools (pty_create, pty_send_keys,
pty_capture, pty_kill, pty_list).

These exercise the real PtyManager + TerminalSessionStore wired via
`set_terminal_runtime`, so the tools and their underlying runtime get covered
in one pass.
"""

from __future__ import annotations

import sys
import time

import pytest

from tsugite.daemon.pty_manager import PtyManager
from tsugite.daemon.terminal_store import TerminalSessionStore, TerminalState
from tsugite.tools import terminal as terminal_tools

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="PTY support is POSIX-only")


@pytest.fixture
def runtime(tmp_path):
    """Stand up a real PtyManager + TerminalSessionStore and wire the tools to it.

    Teardown drops the runtime references so the next test starts clean and
    kills any straggler PTYs.
    """
    mgr = PtyManager()
    store = TerminalSessionStore(tmp_path / "terminal_sessions.json")
    terminal_tools.set_terminal_runtime(mgr, store, None)
    try:
        yield mgr, store
    finally:
        mgr.shutdown()
        terminal_tools.set_terminal_runtime(None, None, None)


def _wait_until_terminal_state(store: TerminalSessionStore, terminal_id: str, timeout: float = 5.0):
    """Block until the terminal reaches a sink state, or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        t = store.get(terminal_id)
        if t and t.state in {
            TerminalState.SUCCEEDED.value,
            TerminalState.FAILED.value,
            TerminalState.CANCELLED.value,
            TerminalState.STREAM_LOST.value,
        }:
            return t
        time.sleep(0.05)
    return store.get(terminal_id)


def _wait_for_output(terminal_id: str, marker: str, timeout: float = 5.0) -> bool:
    """Block until `marker` appears in the captured buffer, or timeout.

    Seeing output proves the child has exec'd and the PTY is flowing both ways,
    so the slave is the controlling terminal with the child in the foreground
    group - the precondition for a written Ctrl+C byte to raise SIGINT. Polling
    for this beats a fixed sleep, which slow CI runners lose the race on.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker in terminal_tools.pty_capture(terminal_id)["text"]:
            return True
        time.sleep(0.05)
    return False


def test_pty_create_spawns_real_process(runtime):
    mgr, store = runtime
    result = terminal_tools.pty_create("echo hi")
    assert "terminal_id" in result
    assert "pid" in result
    assert result["cmd"] == "echo hi"
    assert result["pid"] > 0

    _wait_until_terminal_state(store, result["terminal_id"])

    captured = terminal_tools.pty_capture(result["terminal_id"])
    assert "hi" in captured["text"]


def test_pty_send_keys_writes_to_stdin(runtime):
    mgr, store = runtime
    created = terminal_tools.pty_create("cat")
    tid = created["terminal_id"]

    # Give cat a moment to exec before we write into it.
    time.sleep(0.1)

    sent = terminal_tools.pty_send_keys(tid, "hello")
    assert sent["bytes_written"] > 0

    # Cat echoes the line back; poll the buffer for it.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        cap = terminal_tools.pty_capture(tid)
        if "hello" in cap["text"]:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"never saw 'hello' echoed back: {cap['text']!r}")

    terminal_tools.pty_kill(tid)


def test_pty_send_keys_ctrl_c_interrupts(runtime):
    mgr, store = runtime
    created = terminal_tools.pty_create("echo READY; sleep 30")
    tid = created["terminal_id"]

    assert _wait_for_output(tid, "READY"), "child never produced output; PTY not ready for Ctrl+C"
    sent = terminal_tools.pty_send_keys(tid, "\x03", enter=False)
    assert sent["bytes_written"] > 0

    final = _wait_until_terminal_state(store, tid, timeout=5.0)
    assert final is not None
    # Sending Ctrl+C kills sleep; the shell parent reports the signal. The
    # daemon sees a non-zero exit (FAILED) unless kill() was called too.
    assert final.state in {
        TerminalState.FAILED.value,
        TerminalState.CANCELLED.value,
        TerminalState.SUCCEEDED.value,
    }


def test_pty_capture_tail_returns_last_n_lines(runtime):
    mgr, store = runtime
    # Spew 20 lines; tail=5 should return the last 5.
    created = terminal_tools.pty_create("for i in $(seq 1 20); do echo line-$i; done")
    tid = created["terminal_id"]

    _wait_until_terminal_state(store, tid)

    captured = terminal_tools.pty_capture(tid, lines=5, tail=True)
    lines = captured["text"].splitlines()
    assert 1 <= len(lines) <= 5
    # Confirm we got the trailing lines, not the leading ones.
    assert "line-20" in captured["text"]
    assert "line-1\n" not in captured["text"]


def test_pty_capture_head_returns_first_n_lines(runtime):
    mgr, store = runtime
    created = terminal_tools.pty_create("for i in $(seq 1 20); do echo line-$i; done")
    tid = created["terminal_id"]

    _wait_until_terminal_state(store, tid)

    captured = terminal_tools.pty_capture(tid, lines=3, tail=False)
    lines = captured["text"].splitlines()
    assert 1 <= len(lines) <= 3
    assert "line-1" in captured["text"]


def test_pty_kill_term_transitions_to_cancelled(runtime):
    mgr, store = runtime
    created = terminal_tools.pty_create("sleep 30")
    tid = created["terminal_id"]

    time.sleep(0.1)
    killed = terminal_tools.pty_kill(tid, signal="TERM")
    assert killed["signal"] == "TERM"

    final = _wait_until_terminal_state(store, tid, timeout=5.0)
    assert final is not None
    assert final.state == TerminalState.CANCELLED.value


def test_pty_kill_int_does_not_cancel(runtime):
    """pty_kill(signal="INT") sends Ctrl+C via stdin and never calls kill(), so
    `killed` stays False and the on-exit hook classifies by exit code.

    Load-bearing: a regression that routed INT through PtyManager.kill() (which
    flips `killed` True → CANCELLED) or dropped the stdin write would change the
    documented outcome. We assert the result is a signal-classified terminal
    state, explicitly NOT CANCELLED.
    """
    mgr, store = runtime
    # A foreground program that catches SIGINT itself (no shell-only trap).
    created = terminal_tools.pty_create("echo READY; sleep 30")
    tid = created["terminal_id"]

    assert _wait_for_output(tid, "READY"), "child never produced output; PTY not ready for INT"
    result = terminal_tools.pty_kill(tid, signal="INT")
    assert result["signal"] == "INT"

    final = _wait_until_terminal_state(store, tid, timeout=5.0)
    assert final is not None
    # killed never flipped, so the exit-code classifier runs: a SIGINT death is
    # non-zero (FAILED); a shell that exits 0 on interrupt would be SUCCEEDED.
    # The one outcome INT must NOT produce is CANCELLED (that requires kill()).
    assert final.state != TerminalState.CANCELLED.value
    assert final.state in {
        TerminalState.FAILED.value,
        TerminalState.SUCCEEDED.value,
    }


def test_pty_kill_then_kill_escalates(runtime):
    mgr, store = runtime
    # Script that ignores SIGTERM so we need SIGKILL to take it down.
    created = terminal_tools.pty_create("trap '' TERM; sleep 30")
    tid = created["terminal_id"]

    time.sleep(0.2)
    terminal_tools.pty_kill(tid, signal="TERM")
    time.sleep(0.1)
    terminal_tools.pty_kill(tid, signal="KILL")

    final = _wait_until_terminal_state(store, tid, timeout=5.0)
    assert final is not None
    assert final.state == TerminalState.CANCELLED.value


def test_pty_list_filters_by_state(runtime):
    mgr, store = runtime
    # Spawn one that finishes quickly, and one that stays running.
    quick = terminal_tools.pty_create("echo done")
    long_running = terminal_tools.pty_create("sleep 30")
    try:
        _wait_until_terminal_state(store, quick["terminal_id"])

        running = terminal_tools.pty_list(state="running")
        running_ids = {t["terminal_id"] for t in running}
        assert long_running["terminal_id"] in running_ids
        assert quick["terminal_id"] not in running_ids

        all_terms = terminal_tools.pty_list()
        all_ids = {t["terminal_id"] for t in all_terms}
        assert long_running["terminal_id"] in all_ids
        assert quick["terminal_id"] in all_ids
    finally:
        terminal_tools.pty_kill(long_running["terminal_id"])


def test_pty_list_returns_empty_when_no_terminals(runtime):
    mgr, store = runtime
    listed = terminal_tools.pty_list()
    assert listed == []


def test_pty_create_outside_daemon_errors():
    """When the runtime hasn't been wired, tools return an error dict."""
    # Explicitly clear runtime to simulate non-daemon mode.
    terminal_tools.set_terminal_runtime(None, None, None)

    result = terminal_tools.pty_create("echo hi")
    assert "error" in result
    assert "not available" in result["error"] or "daemon" in result["error"]


def test_pty_send_keys_outside_daemon_errors():
    terminal_tools.set_terminal_runtime(None, None, None)
    result = terminal_tools.pty_send_keys("term-x", "hi")
    assert "error" in result


def test_pty_capture_outside_daemon_errors():
    terminal_tools.set_terminal_runtime(None, None, None)
    result = terminal_tools.pty_capture("term-x")
    assert "error" in result


def test_pty_kill_unknown_terminal_returns_error(runtime):
    mgr, store = runtime
    result = terminal_tools.pty_kill("term-does-not-exist")
    assert "error" in result


def test_pty_send_keys_unknown_terminal_returns_error(runtime):
    mgr, store = runtime
    result = terminal_tools.pty_send_keys("term-does-not-exist", "hi")
    assert "error" in result


def test_pty_kill_unsupported_signal_returns_error(runtime):
    mgr, store = runtime
    created = terminal_tools.pty_create("sleep 30")
    try:
        result = terminal_tools.pty_kill(created["terminal_id"], signal="HUP")
        assert "error" in result
        assert "HUP" in result["error"] or "Unsupported" in result["error"]
    finally:
        terminal_tools.pty_kill(created["terminal_id"])


def test_pty_send_keys_appends_newline_when_enter_true(runtime):
    mgr, store = runtime
    created = terminal_tools.pty_create("cat")
    tid = created["terminal_id"]
    try:
        time.sleep(0.1)
        # Without enter=True the bytes_written is len("hi") = 2; with it 3.
        result = terminal_tools.pty_send_keys(tid, "hi", enter=True)
        assert result["bytes_written"] == 3
    finally:
        terminal_tools.pty_kill(tid)


def test_pty_send_keys_no_newline_when_enter_false(runtime):
    mgr, store = runtime
    created = terminal_tools.pty_create("cat")
    tid = created["terminal_id"]
    try:
        time.sleep(0.1)
        result = terminal_tools.pty_send_keys(tid, "hi", enter=False)
        assert result["bytes_written"] == 2
    finally:
        terminal_tools.pty_kill(tid)


def test_pty_capture_reads_persisted_log_after_eviction(runtime):
    """After the exit-grace eviction the PtyProcess (and its ring buffer) are
    gone, but the exit hook persisted the full output log. pty_capture must
    replay that log instead of degrading to the one-line last_line snapshot."""
    mgr, store = runtime
    created = terminal_tools.pty_create("printf 'line1\\nline2\\nline3\\n'")
    tid = created["terminal_id"]
    _wait_until_terminal_state(store, tid)

    log_path = store.log_path(tid)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not log_path.exists():
        time.sleep(0.05)
    assert log_path.exists(), "exit hook should have persisted the output log"

    mgr.remove(tid)  # simulate the EVICT_GRACE_SECONDS eviction

    captured = terminal_tools.pty_capture(tid)
    assert "line1" in captured["text"]
    assert "line3" in captured["text"]
