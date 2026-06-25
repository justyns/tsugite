"""Connection lifecycle for the SQLite history battery.

History has genuine concurrent writers across both threads and processes: the daemon
runs agent turns on worker threads (asyncio.to_thread), and the scheduler and CLI are
separate processes. SQLite connection objects are not safe to share across threads, so
we keep one connection per (db_path, thread) and let WAL + busy_timeout serialize
writers. ``isolation_level=None`` puts the module in autocommit so the backend controls
transactions explicitly (``BEGIN IMMEDIATE``).
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from .sqlite_schema import apply_migrations

logger = logging.getLogger(__name__)

BUSY_TIMEOUT_MS = 5000

# SQLite + WAL is unsafe on these: POSIX advisory locks are unreliable over the network
# and WAL's shared-memory file assumes a local mmap. Concurrent access corrupts the db.
_NETWORK_FS = {
    "nfs",
    "nfs4",
    "cifs",
    "smbfs",
    "smb3",
    "fuse.sshfs",
    "fuse.glusterfs",
    "glusterfs",
    "9p",
    "afs",
    "ncpfs",
}

_conns: dict[tuple[str, int], sqlite3.Connection] = {}
_lock = threading.Lock()
_warned_paths: set[str] = set()


def connect_history_db(path: Path) -> sqlite3.Connection:
    """Return this thread's connection to the history db, creating + migrating on first use."""
    key = (str(path), threading.get_ident())
    with _lock:
        conn = _conns.get(key)
        if conn is not None:
            return conn
        path.parent.mkdir(parents=True, exist_ok=True)
        _warn_if_network_fs(path)
        conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        apply_migrations(conn)
        _conns[key] = conn
        return conn


def close_all() -> None:
    """Close every cached connection (used by tests and history backend resets)."""
    with _lock:
        for conn in _conns.values():
            with contextlib.suppress(Exception):
                conn.close()
        _conns.clear()


def _warn_if_network_fs(path: Path) -> None:
    key = str(path)
    if key in _warned_paths:
        return
    _warned_paths.add(key)
    fstype = _filesystem_type(path)
    if fstype and fstype.lower() in _NETWORK_FS:
        logger.warning(
            "History database %s is on a network filesystem (%s). SQLite + WAL is unsafe over "
            "NFS/SMB and concurrent access can corrupt it. Use a local-disk path (or the future "
            "postgres backend) for multi-machine setups.",
            path,
            fstype,
        )


def _filesystem_type(path: Path) -> Optional[str]:
    """Best-effort filesystem type of the mount containing ``path`` (Linux /proc/mounts)."""
    try:
        target = path.resolve()
        best_fstype: Optional[str] = None
        best_len = -1
        for line in Path("/proc/mounts").read_text().splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            mount_point, fstype = parts[1], parts[2]
            try:
                target.relative_to(mount_point)
            except ValueError:
                continue
            if len(mount_point) > best_len:
                best_fstype, best_len = fstype, len(mount_point)
        return best_fstype
    except Exception:
        return None
