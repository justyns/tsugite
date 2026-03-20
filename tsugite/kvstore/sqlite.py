"""SQLite KV store backend."""

import sqlite3
import time

from tsugite.config import get_xdg_data_path


class SqliteKVBackend:
    def __init__(self, db_path=None):
        self._db_path = db_path or (get_xdg_data_path("kvstore") / "kv.db")
        self._conn = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS kv ("
                "namespace TEXT, key TEXT, value TEXT, expires_at INTEGER, "
                "PRIMARY KEY(namespace, key))"
            )
            self._conn.commit()
        return self._conn

    def get(self, namespace: str, key: str) -> str | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value FROM kv WHERE namespace=? AND key=? AND (expires_at IS NULL OR expires_at > ?)",
            (namespace, key, int(time.time())),
        ).fetchone()
        return row[0] if row else None

    def set(self, namespace: str, key: str, value: str, ttl_seconds: int | None = None) -> None:
        conn = self._get_conn()
        expires_at = int(time.time()) + ttl_seconds if ttl_seconds is not None else None
        conn.execute(
            "INSERT OR REPLACE INTO kv (namespace, key, value, expires_at) VALUES (?, ?, ?, ?)",
            (namespace, key, value, expires_at),
        )
        conn.commit()

    def delete(self, namespace: str, key: str) -> bool:
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM kv WHERE namespace=? AND key=?", (namespace, key))
        conn.commit()
        return cursor.rowcount > 0

    def list_keys(self, namespace: str, prefix: str = "") -> list[str]:
        conn = self._get_conn()
        now = int(time.time())
        if prefix:
            escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            rows = conn.execute(
                "SELECT key FROM kv WHERE namespace=? AND key LIKE ? ESCAPE '\\' "
                "AND (expires_at IS NULL OR expires_at > ?) ORDER BY key",
                (namespace, escaped + "%", now),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT key FROM kv WHERE namespace=? AND (expires_at IS NULL OR expires_at > ?) ORDER BY key",
                (namespace, now),
            ).fetchall()
        return [r[0] for r in rows]

    def list_namespaces(self) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT DISTINCT namespace FROM kv WHERE expires_at IS NULL OR expires_at > ? ORDER BY namespace",
            (int(time.time()),),
        ).fetchall()
        return [r[0] for r in rows]

    def get_with_metadata(self, namespace: str, key: str) -> dict | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value, expires_at FROM kv WHERE namespace=? AND key=? AND (expires_at IS NULL OR expires_at > ?)",
            (namespace, key, int(time.time())),
        ).fetchone()
        if not row:
            return None
        return {"value": row[0], "expires_at": row[1]}
