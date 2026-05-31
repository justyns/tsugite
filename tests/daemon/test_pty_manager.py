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
    """A subscriber callback fires on every chunk read off the PTY fd."""
    p = PtyProcess.spawn(["/bin/sh", "-c", "printf 'one two three'"])
    received: list[bytes] = []
    p.subscribe(lambda chunk: received.append(chunk))
    _wait_for_exit(p, timeout=3.0)
    p.wait_drain(timeout=1.0)
    joined = b"".join(received)
    assert b"one two three" in joined


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
        # Must not raise — caller may have raced state vs. PTY exit.
        mgr.kill("does-not-exist")
    finally:
        mgr.shutdown()


def test_manager_subscribe_dispatches_chunks():
    mgr = PtyManager()
    try:
        received: list[bytes] = []
        p = mgr.spawn("term-sub", ["/bin/sh", "-c", "printf 'subscribed!'"])
        mgr.subscribe("term-sub", lambda chunk: received.append(chunk))
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


def test_default_buffer_cap_is_reasonable():
    """Sanity check: default cap is the documented 1MB-ish, not zero or insanely high."""
    assert 256 * 1024 <= DEFAULT_BUFFER_CAP <= 16 * 1024 * 1024


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
