"""Tests for PtyProcess + PtyManager: spawn → read → exit-code lifecycle."""

from __future__ import annotations

import sys
import time

import pytest

from tsugite.daemon.pty_manager import (
    DEFAULT_BUFFER_CAP,
    PtyManager,
    PtyProcess,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="PTY support is POSIX-only")


# ── PtyProcess ──


def test_spawn_echo_captures_output_and_exit_zero():
    p = PtyProcess.spawn(["/bin/sh", "-c", "echo hello"])
    output = _drain(p, timeout=3.0)
    assert "hello" in output
    assert p.exit_code == 0
    assert p.pid > 0


def test_spawn_nonzero_exit_captured():
    p = PtyProcess.spawn(["/bin/sh", "-c", "exit 7"])
    _drain(p, timeout=3.0)
    assert p.exit_code == 7


def test_spawn_with_cwd(tmp_path):
    (tmp_path / "marker.txt").write_text("hi")
    p = PtyProcess.spawn(["/bin/sh", "-c", "ls"], cwd=str(tmp_path))
    output = _drain(p, timeout=3.0)
    assert "marker.txt" in output
    assert p.exit_code == 0


def test_spawn_with_env_var():
    p = PtyProcess.spawn(["/bin/sh", "-c", "echo $TEST_VAR"], env={"TEST_VAR": "tsugite-test"})
    output = _drain(p, timeout=3.0)
    assert "tsugite-test" in output


def test_kill_marks_process_terminated():
    p = PtyProcess.spawn(["/bin/sh", "-c", "sleep 30"])
    # Give the shell a moment to actually exec sleep so SIGTERM lands on it.
    time.sleep(0.1)
    p.kill()
    _wait_for_exit(p, timeout=5.0)
    # SIGTERM kills the shell; sleep inherits and dies. Exit code is non-zero.
    assert p.exit_code is not None
    assert p.exit_code != 0


def test_double_kill_escalates_to_sigkill():
    # A trap-ignoring shell ensures the SIGTERM grace period elapses, forcing SIGKILL.
    script = "trap '' TERM; sleep 30"
    p = PtyProcess.spawn(["/bin/sh", "-c", script])
    time.sleep(0.2)
    p.kill()  # SIGTERM ignored
    time.sleep(0.1)
    p.kill()  # second call escalates to SIGKILL
    _wait_for_exit(p, timeout=5.0)
    assert p.exit_code is not None


def test_subscribe_receives_output_chunks():
    """A subscriber callback fires on chunks read off the PTY fd.

    Uses `cat` and writes stdin AFTER subscribing: a fast one-shot command can
    emit (and the reader thread dispatch) its output before subscribe() lands,
    which made this flake under parallel test load. Real consumers that need
    the pre-subscribe output use snapshot_and_subscribe().
    """
    p = PtyProcess.spawn(["/bin/cat"])
    received: list[bytes] = []
    p.subscribe(lambda chunk: received.append(chunk))
    p.write_stdin(b"one two three\n")
    deadline = time.time() + 3.0
    while time.time() < deadline and b"one two three" not in b"".join(received):
        time.sleep(0.05)
    p.kill()
    _wait_for_exit(p, timeout=3.0)
    p.wait_drain(timeout=1.0)
    assert b"one two three" in b"".join(received)


def test_on_exit_unsubscribe_prevents_callback():
    """The callable returned by on_exit removes the callback so it never fires.

    Load-bearing: pre-fix on_exit returned None, so callers (the SSE generator's
    `finally`) had no way to detach and the exit hook fired into a torn-down
    stream. We register on a still-running proc, immediately unregister, then let
    it exit and assert the callback was NOT invoked.
    """
    p = PtyProcess.spawn(["/bin/sh", "-c", "sleep 0.3"])
    fired: list[PtyProcess] = []
    unregister = p.on_exit(lambda proc: fired.append(proc))
    assert callable(unregister)
    unregister()
    _wait_for_exit(p, timeout=3.0)
    p.wait_drain(timeout=1.0)
    assert fired == []


def test_on_exit_on_already_exited_fires_synchronously_and_returns_noop():
    """on_exit on an already-exited proc fires the callback synchronously and
    returns a no-op callable.

    Load-bearing: the SSE handler registers on_exit AFTER snapshot/subscribe, so
    a proc that exited in the meantime must still deliver its exit event. The
    returned unregister must be safe to call (no-op) since nothing was queued.
    """
    p = PtyProcess.spawn(["/bin/sh", "-c", "echo done"])
    _wait_for_exit(p, timeout=3.0)
    p.wait_drain(timeout=1.0)
    assert p.exit_code is not None  # confirm it's already terminal

    fired: list[PtyProcess] = []
    unregister = p.on_exit(lambda proc: fired.append(proc))
    # Fired synchronously, in-line, before on_exit returned.
    assert fired == [p]
    assert callable(unregister)
    # No-op: must not raise, must not re-fire.
    unregister()
    assert fired == [p]


def test_snapshot_and_subscribe_no_chunk_lost_across_the_boundary():
    """snapshot_and_subscribe() returns the buffer AND registers the subscriber
    under one lock, so a chunk produced right around the call is either in the
    snapshot or delivered to the subscriber - never dropped.

    Load-bearing: pins the atomicity the SSE handler depends on. A regression
    that snapshotted and subscribed in two separate lock acquisitions could lose
    a chunk landing between them. We subscribe while the process is provably
    mid-stream (slow per-line emitter), so some lines are buffered (snapshot) and
    the rest arrive live (delivered). The union must reconstruct L001..L200 with
    no gap; we also assert delivery actually happened so the boundary is exercised.
    """
    # Slow emitter: ~10ms/line keeps output pending for ~2s, guaranteeing the
    # reader is actively appending when we subscribe partway through. Monotonic
    # line numbers let us detect a lost chunk as a missing line.
    p = PtyProcess.spawn(["/bin/sh", "-c", "for i in $(seq 1 200); do printf 'L%03d\\n' $i; sleep 0.01; done"])
    # Let the first lines accumulate into the buffer before we snapshot+subscribe.
    time.sleep(0.2)

    delivered: list[bytes] = []
    snapshot, unsub = p.snapshot_and_subscribe(lambda chunk: delivered.append(chunk))

    _wait_for_exit(p, timeout=8.0)
    p.wait_drain(timeout=2.0)
    unsub()

    # Both halves must be non-empty, else the boundary wasn't actually crossed
    # under load and the atomicity guarantee went untested.
    assert snapshot, "snapshot was empty - subscribed before any output buffered"
    assert delivered, "no chunks delivered live - subscribed after the process finished"

    combined = snapshot + b"".join(delivered)
    # The PTY line discipline rewrites \n as \r\n (ONLCR), so lines land as
    # `L001\r\n`. A chunk lost on the boundary would drop a line here.
    for i in range(1, 201):
        assert (b"L%03d\r\n" % i) in combined, f"line L{i:03d} lost across snapshot/subscribe boundary"
    # The snapshot is a true prefix of the full stream (no interleaving bug).
    expected = b"".join(b"L%03d\r\n" % i for i in range(1, 201))
    assert expected.startswith(snapshot), "snapshot was not a prefix of the output stream"


def test_buffer_truncation_marks_truncated():
    """When output exceeds the ring-buffer cap, total bytes_out keeps climbing
    but the buffer caps and `truncated` flips True."""
    p = PtyProcess.spawn(
        ["/bin/sh", "-c", "head -c 200000 /dev/urandom | base64"],
        buffer_cap=8192,
    )
    _wait_for_exit(p, timeout=10.0)
    p.wait_drain(timeout=2.0)
    assert p.bytes_out > 8192, f"expected more than the buffer cap; got {p.bytes_out}"
    assert p.truncated is True
    assert len(p.buffer) <= 8192


def test_buffer_no_truncation_under_cap():
    p = PtyProcess.spawn(["/bin/sh", "-c", "echo small"])
    _wait_for_exit(p, timeout=3.0)
    p.wait_drain(timeout=1.0)
    assert p.truncated is False
    assert p.bytes_out > 0


# ── PtyManager ──


def test_manager_spawn_and_get():
    mgr = PtyManager()
    try:
        p = mgr.spawn("term-1", ["/bin/sh", "-c", "echo manager"])
        assert mgr.get("term-1") is p
        _wait_for_exit(p, timeout=3.0)
    finally:
        mgr.shutdown()


def test_manager_spawn_duplicate_raises():
    mgr = PtyManager()
    try:
        mgr.spawn("term-dup", ["/bin/sh", "-c", "sleep 5"])
        with pytest.raises(ValueError):
            mgr.spawn("term-dup", ["/bin/sh", "-c", "echo no"])
    finally:
        mgr.shutdown()


def test_manager_kill_known_terminal():
    mgr = PtyManager()
    try:
        p = mgr.spawn("term-k", ["/bin/sh", "-c", "sleep 30"])
        time.sleep(0.1)
        mgr.kill("term-k")
        _wait_for_exit(p, timeout=5.0)
        assert p.exit_code is not None
    finally:
        mgr.shutdown()


def test_manager_kill_unknown_is_noop():
    mgr = PtyManager()
    try:
        # Must not raise - caller may have raced state vs. PTY exit.
        mgr.kill("does-not-exist")
    finally:
        mgr.shutdown()


def test_manager_subscribe_dispatches_chunks():
    """Same race-free shape as test_subscribe_receives_output_chunks: output is
    produced after the subscription, so the dispatch path is what's tested."""
    mgr = PtyManager()
    try:
        received: list[bytes] = []
        p = mgr.spawn("term-sub", ["/bin/cat"])
        mgr.subscribe("term-sub", lambda chunk: received.append(chunk))
        mgr.write_stdin("term-sub", b"subscribed!\n")
        deadline = time.time() + 3.0
        while time.time() < deadline and b"subscribed!" not in b"".join(received):
            time.sleep(0.05)
        mgr.kill("term-sub")
        _wait_for_exit(p, timeout=3.0)
        p.wait_drain(timeout=1.0)
        assert b"subscribed!" in b"".join(received)
    finally:
        mgr.shutdown()


def test_manager_write_stdin_round_trips():
    """`cat` echoes whatever we send to stdin. Verifies stdin write path."""
    mgr = PtyManager()
    try:
        received: list[bytes] = []
        p = mgr.spawn("term-cat", ["/bin/cat"])
        mgr.subscribe("term-cat", lambda chunk: received.append(chunk))
        # Tiny pause so cat is actually exec'd before we write.
        time.sleep(0.1)
        mgr.write_stdin("term-cat", b"ping\n")
        # Give cat a moment to read + write back.
        deadline = time.time() + 2.0
        while time.time() < deadline and b"ping" not in b"".join(received):
            time.sleep(0.05)
        mgr.kill("term-cat")
        _wait_for_exit(p, timeout=3.0)
        assert b"ping" in b"".join(received)
    finally:
        mgr.shutdown()


def test_exited_pty_is_evicted_from_manager(tmp_path, monkeypatch):
    """After a PTY exits (plus the eviction grace), the manager must drop the
    PtyProcess so its ~1 MB ring buffer is freed.

    Load-bearing: pre-fix the exit hook never removed the entry, so exited procs
    were retained forever. We override the grace to ~0 to avoid the 30s timer and
    drive the real exit hook, then assert the id is gone (manager.remove ran).
    """
    from tsugite.daemon import terminal_runtime
    from tsugite.daemon.terminal_store import TerminalSession, TerminalSessionStore, TerminalState

    monkeypatch.setattr(terminal_runtime, "EVICT_GRACE_SECONDS", 0.0)

    store = TerminalSessionStore(tmp_path / "terminal_sessions.json")
    mgr = PtyManager()
    try:
        session = store.add(TerminalSession(id="", cmd="echo bye", cwd=None, parent_session_id=None))
        proc = mgr.spawn(session.id, ["/bin/sh", "-c", "echo bye"])
        store.update_state(session.id, TerminalState.RUNNING.value)
        _wait_for_exit(proc, timeout=3.0)
        proc.wait_drain(timeout=1.0)
        assert mgr.get(session.id) is proc  # still retained until the hook runs

        # Drive the real exit hook with grace=0; it schedules manager.remove via a
        # Timer, which fires near-immediately. Poll for the eviction.
        terminal_runtime._on_pty_exit(proc, store, mgr, session.id)
        deadline = time.time() + 3.0
        while time.time() < deadline and mgr.get(session.id) is not None:
            time.sleep(0.02)
        assert mgr.get(session.id) is None, "exited PTY was not evicted from the manager"
        # The persisted record survives the eviction (final counts already saved).
        assert store.get(session.id) is not None
    finally:
        mgr.shutdown()


def test_default_buffer_cap_is_reasonable():
    """Sanity check: default cap is the documented 1MB-ish, not zero or insanely high."""
    assert 256 * 1024 <= DEFAULT_BUFFER_CAP <= 16 * 1024 * 1024


def test_exit_hook_persists_buffer_to_log_file(tmp_path, monkeypatch):
    """The output buffer must be written to disk before the in-memory PtyProcess
    is evicted, so re-opening an old terminal still shows what it printed.
    Regression: the UI claimed `output preserved` but only for the 30s grace
    window; after eviction the SSE stream returned an empty terminal."""
    from tsugite.daemon import terminal_runtime
    from tsugite.daemon.terminal_store import TerminalSession, TerminalSessionStore, TerminalState

    monkeypatch.setattr(terminal_runtime, "EVICT_GRACE_SECONDS", 60.0)  # don't race the write

    store = TerminalSessionStore(tmp_path / "terminal_sessions.json")
    mgr = PtyManager()
    try:
        session = store.add(TerminalSession(id="", cmd="echo hi", cwd=None, parent_session_id=None))
        proc = mgr.spawn(session.id, ["/bin/sh", "-c", "echo hi"])
        store.update_state(session.id, TerminalState.RUNNING.value)
        _wait_for_exit(proc, timeout=3.0)
        proc.wait_drain(timeout=1.0)

        terminal_runtime._on_pty_exit(proc, store, mgr, session.id)

        log_path = store.log_path(session.id)
        assert log_path.is_file(), "exit hook must write the captured buffer to disk"
        assert b"hi" in log_path.read_bytes()
    finally:
        mgr.shutdown()


def test_exit_hook_skips_log_when_buffer_is_empty(tmp_path, monkeypatch):
    """No point writing a zero-byte file: skip when the PTY produced nothing."""
    from tsugite.daemon import terminal_runtime
    from tsugite.daemon.terminal_store import TerminalSession, TerminalSessionStore, TerminalState

    monkeypatch.setattr(terminal_runtime, "EVICT_GRACE_SECONDS", 60.0)

    store = TerminalSessionStore(tmp_path / "terminal_sessions.json")
    mgr = PtyManager()
    try:
        session = store.add(TerminalSession(id="", cmd="true", cwd=None, parent_session_id=None))
        proc = mgr.spawn(session.id, ["/bin/sh", "-c", "true"])
        store.update_state(session.id, TerminalState.RUNNING.value)
        _wait_for_exit(proc, timeout=3.0)
        proc.wait_drain(timeout=1.0)

        terminal_runtime._on_pty_exit(proc, store, mgr, session.id)

        assert not store.log_path(session.id).exists(), "empty buffer must not produce a log file"
    finally:
        mgr.shutdown()


# ── helpers ──


def _drain(p: PtyProcess, timeout: float) -> str:
    """Wait for the process to exit and return decoded buffer output."""
    _wait_for_exit(p, timeout)
    p.wait_drain(timeout=1.0)
    return p.buffer.decode("utf-8", errors="replace")


def _wait_for_exit(p: PtyProcess, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if p.exit_code is not None:
            return
        time.sleep(0.02)
    raise AssertionError(f"PTY did not exit within {timeout}s; pid={p.pid}")
