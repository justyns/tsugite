"""Daemon-managed PTY processes for the terminal viewer.

Uses stdlib `os.openpty` + `subprocess.Popen` (no external dep). One background
thread per PTY drains the master fd into a ring buffer and dispatches each chunk
to any subscribed callbacks. Subscribers (e.g. the SSE handler) get raw bytes
including ANSI escapes; encoding/JSON-framing is left to the caller.

The ring buffer is capped (default 1 MB). Output beyond the cap drops from the
buffer but `bytes_out` keeps counting, and a `truncated` flag flips True so the
UI can show "+47 MB truncated" without us holding 47 MB of memory.

State + persistence lives in `terminal_store.py`. This module is the runtime
side: spawn, read, write stdin, kill.
"""

from __future__ import annotations

import errno
import logging
import os
import pty
import signal
import subprocess
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_BUFFER_CAP = 1024 * 1024  # 1 MB
SIGKILL_GRACE_SECONDS = 2.0
_READ_CHUNK_SIZE = 8192


class PtyProcess:
    """A single PTY-backed subprocess with a ring-buffered output stream.

    Construct via `PtyProcess.spawn(...)`. The reader thread starts immediately
    so subscribers attached after spawn still get every chunk after subscription;
    chunks emitted before subscription are NOT replayed (the buffer is for that).
    """

    def __init__(
        self,
        proc: subprocess.Popen,
        master_fd: int,
        cmd: list[str],
        buffer_cap: int = DEFAULT_BUFFER_CAP,
    ):
        self._proc = proc
        self._master_fd = master_fd
        self.cmd = cmd
        self._buffer_cap = buffer_cap
        # `deque` with a maxlen would auto-evict, but we need byte-granularity
        # eviction (not per-chunk), so we maintain a flat bytearray and trim.
        self._buffer = bytearray()
        self.bytes_out = 0
        self.lines_out = 0
        self.last_line = ""
        self.truncated = False
        self.killed = False  # True once kill() has been called at least once
        self.exit_code: Optional[int] = None
        self._subscribers: list[Callable[[bytes], None]] = []
        self._exit_callbacks: list[Callable[["PtyProcess"], None]] = []
        self._subscribers_lock = threading.Lock()
        self._buffer_lock = threading.Lock()
        self._closed = threading.Event()
        self._first_kill_at: Optional[float] = None
        self._reader_thread = threading.Thread(target=self._reader_loop, name=f"pty-{proc.pid}", daemon=True)
        self._reader_thread.start()

    @classmethod
    def spawn(
        cls,
        cmd: list[str],
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
        buffer_cap: int = DEFAULT_BUFFER_CAP,
    ) -> "PtyProcess":
        """Allocate a PTY pair and exec `cmd` inside the slave.

        env defaults to the daemon's env if not provided. We merge user-provided
        env on top so callers can override individual vars without losing PATH.
        """
        master_fd, slave_fd = pty.openpty()
        try:
            full_env = dict(os.environ)
            if env:
                full_env.update(env)
            # `TERM=xterm-256color` gives the child program a sane default for
            # ANSI-aware output; callers can override via env.
            full_env.setdefault("TERM", "xterm-256color")
            proc = subprocess.Popen(
                cmd,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                cwd=cwd,
                env=full_env,
                # Putting the child in its own process group lets us kill the
                # whole tree (a shell + its children) with one signal to -pgid.
                preexec_fn=os.setsid,
                close_fds=True,
            )
        except Exception:
            os.close(master_fd)
            os.close(slave_fd)
            raise
        finally:
            # The slave end is owned by the child process now; the parent must
            # close its copy or the master read will never see EOF.
            try:
                os.close(slave_fd)
            except OSError:
                pass
        return cls(proc, master_fd, cmd, buffer_cap=buffer_cap)

    @property
    def pid(self) -> int:
        return self._proc.pid

    @property
    def buffer(self) -> bytes:
        """Snapshot of the current ring-buffer contents (oldest dropped if over cap)."""
        with self._buffer_lock:
            return bytes(self._buffer)

    def subscribe(self, callback: Callable[[bytes], None]) -> Callable[[], None]:
        """Register a callback fired on every output chunk. Returns an unsubscribe fn."""
        with self._subscribers_lock:
            self._subscribers.append(callback)
        return self._make_unsubscribe(callback)

    def snapshot_and_subscribe(self, callback: Callable[[bytes], None]) -> tuple[bytes, Callable[[], None]]:
        """Atomically snapshot the buffer AND register a chunk subscriber.

        Holding `_buffer_lock` across both ensures no chunk slips between the
        snapshot and the subscription: the reader appends under `_buffer_lock`
        before dispatching under `_subscribers_lock`, so any chunk not in the
        returned snapshot is guaranteed to reach `callback`. A boundary chunk may
        be delivered both ways (in the snapshot and to the callback); duplicating
        is the safe failure mode, dropping is not.
        """
        with self._buffer_lock:
            snapshot = bytes(self._buffer)
            with self._subscribers_lock:
                self._subscribers.append(callback)
        return snapshot, self._make_unsubscribe(callback)

    def _make_unsubscribe(self, callback: Callable[[bytes], None]) -> Callable[[], None]:
        def _unsubscribe() -> None:
            with self._subscribers_lock:
                try:
                    self._subscribers.remove(callback)
                except ValueError:
                    pass

        return _unsubscribe

    def on_exit(self, callback: Callable[["PtyProcess"], None]) -> Callable[[], None]:
        """Register a callback fired once the PTY exits. Returns an unregister fn.

        If the process has already exited by the time on_exit is called, the
        callback fires synchronously - callers can register late without racing
        the reader thread's exit cleanup. In that case the returned unregister fn
        is a no-op (the callback already ran and was never queued).
        """
        with self._subscribers_lock:
            if self.exit_code is not None:
                fire_now = True
            else:
                self._exit_callbacks.append(callback)
                fire_now = False
        if fire_now:
            try:
                callback(self)
            except Exception:
                logger.exception("PtyProcess on_exit callback failed (late registration)")
            return lambda: None

        def _unregister() -> None:
            with self._subscribers_lock:
                try:
                    self._exit_callbacks.remove(callback)
                except ValueError:
                    pass

        return _unregister

    def write_stdin(self, data: bytes) -> int:
        """Write bytes to the PTY master. Returns count written. No-op after exit.

        Serialized with fd teardown under `_buffer_lock` so we never write to a
        closed (and possibly OS-reused) fd while the reader thread is closing it.
        """
        with self._buffer_lock:
            if self.exit_code is not None or self._master_fd < 0:
                return 0
            try:
                return os.write(self._master_fd, data)
            except OSError as e:
                if e.errno in (errno.EIO, errno.EBADF):
                    # Slave closed / fd torn down. Treat as no-op like exit case.
                    return 0
                raise

    def kill(self) -> None:
        """Send SIGTERM the first time; SIGKILL on subsequent calls or after grace.

        Signals the child's process group so the whole tree dies (shell + nested).
        Safe to call multiple times and after the process has already exited.
        """
        if self.exit_code is not None:
            return
        now = time.monotonic()
        try:
            pgid = os.getpgid(self.pid)
        except OSError:
            return  # Already gone.

        if self._first_kill_at is None:
            self._first_kill_at = now
            sig = signal.SIGTERM
        elif now - self._first_kill_at >= SIGKILL_GRACE_SECONDS:
            sig = signal.SIGKILL
        else:
            sig = signal.SIGKILL  # explicit second call = escalate immediately
        self.killed = True
        try:
            os.killpg(pgid, sig)
        except OSError as e:
            if e.errno != errno.ESRCH:  # already dead
                raise

    def wait_drain(self, timeout: float = 1.0) -> None:
        """Block until the reader thread finishes (or timeout). Tests use this
        to make sure all pending PTY output has landed in the buffer before
        asserting. Production code should never need to block on this."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._closed.is_set() and not self._reader_thread.is_alive():
                return
            time.sleep(0.02)

    # ── internals ──

    def _reader_loop(self) -> None:
        try:
            while True:
                try:
                    chunk = os.read(self._master_fd, _READ_CHUNK_SIZE)
                except OSError as e:
                    # EIO is the canonical "PTY slave is gone" indicator on Linux.
                    if e.errno in (errno.EIO, errno.EBADF):
                        break
                    raise
                if not chunk:
                    break
                self._append(chunk)
                self._dispatch(chunk)
        finally:
            try:
                # Reap the child if it's done so exit_code populates.
                self.exit_code = self._proc.wait()
            except Exception:
                logger.exception("PtyProcess: error waiting on child pid=%s", self.pid)
                self.exit_code = self._proc.returncode
            with self._buffer_lock:
                fd, self._master_fd = self._master_fd, -1
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
            with self._subscribers_lock:
                callbacks = list(self._exit_callbacks)
                self._exit_callbacks.clear()
            for cb in callbacks:
                try:
                    cb(self)
                except Exception:
                    logger.exception("PtyProcess on_exit callback failed")
            self._closed.set()

    def _append(self, chunk: bytes) -> None:
        with self._buffer_lock:
            self.bytes_out += len(chunk)
            self.lines_out += chunk.count(b"\n")
            self._buffer.extend(chunk)
            if len(self._buffer) > self._buffer_cap:
                drop = len(self._buffer) - self._buffer_cap
                del self._buffer[:drop]
                self.truncated = True
            # Most recent line: scan the bytearray from the end for the last
            # newline (skipping a trailing one) and slice only that tail. Avoids
            # the ~1 MB memcpy that `bytes(self._buffer)` would do per chunk.
            end = len(self._buffer)
            if end and self._buffer[end - 1] == 0x0A:
                end -= 1
            nl = self._buffer.rfind(b"\n", 0, end)
            last = bytes(self._buffer[nl + 1 : end])  # noqa: E203
            self.last_line = last.decode("utf-8", errors="replace")[-200:]

    def _dispatch(self, chunk: bytes) -> None:
        with self._subscribers_lock:
            subs = list(self._subscribers)
        for cb in subs:
            try:
                cb(chunk)
            except Exception:
                logger.exception("PtyProcess subscriber failed")


class PtyManager:
    """Tracks all live PtyProcess instances by terminal_id.

    A "singleton" in practice (one per daemon, wired through gateway), but the
    class is plain so tests can construct throw-away instances.
    """

    def __init__(self):
        self._procs: dict[str, PtyProcess] = {}
        self._lock = threading.Lock()

    def spawn(
        self,
        terminal_id: str,
        cmd: list[str],
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
        buffer_cap: int = DEFAULT_BUFFER_CAP,
    ) -> PtyProcess:
        """Spawn a PTY for `terminal_id`. Raises ValueError on duplicate id."""
        with self._lock:
            if terminal_id in self._procs:
                raise ValueError(f"Terminal already exists: {terminal_id}")
        proc = PtyProcess.spawn(cmd, cwd=cwd, env=env, buffer_cap=buffer_cap)
        with self._lock:
            self._procs[terminal_id] = proc
        return proc

    def get(self, terminal_id: str) -> Optional[PtyProcess]:
        return self._procs.get(terminal_id)

    def kill(self, terminal_id: str) -> None:
        """Kill a tracked terminal. No-op if unknown or already gone."""
        proc = self._procs.get(terminal_id)
        if proc is None:
            return
        proc.kill()

    def write_stdin(self, terminal_id: str, data: bytes) -> int:
        proc = self._procs.get(terminal_id)
        if proc is None:
            return 0
        return proc.write_stdin(data)

    def subscribe(self, terminal_id: str, callback: Callable[[bytes], None]) -> Optional[Callable[[], None]]:
        """Subscribe to chunk callbacks. Returns the unsubscribe fn, or None if unknown."""
        proc = self._procs.get(terminal_id)
        if proc is None:
            return None
        return proc.subscribe(callback)

    def remove(self, terminal_id: str) -> None:
        """Drop the entry. Caller is responsible for kill-and-drain semantics."""
        with self._lock:
            self._procs.pop(terminal_id, None)

    def shutdown(self) -> None:
        """Kill every tracked PTY. Used at daemon stop and in test teardown."""
        with self._lock:
            procs = list(self._procs.values())
            self._procs.clear()
        for p in procs:
            try:
                p.kill()
            except Exception:
                logger.exception("PtyManager.shutdown: kill failed")
