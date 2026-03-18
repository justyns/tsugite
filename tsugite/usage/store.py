"""SQLite-backed usage tracking store.

Provides efficient aggregation queries for token usage and costs
across agents, schedules, and time periods.
"""

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from .models import UsageRecord

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS usage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    agent TEXT NOT NULL,
    model TEXT NOT NULL,
    session_id TEXT,
    schedule_id TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    cost REAL NOT NULL DEFAULT 0.0,
    duration_ms INTEGER
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_usage_agent ON usage_log(agent);",
    "CREATE INDEX IF NOT EXISTS idx_usage_schedule ON usage_log(schedule_id);",
    "CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_log(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_usage_model ON usage_log(model);",
]


class UsageStore:
    """SQLite-backed store for agent usage tracking.

    Thread-safe. DB is created lazily on first write.
    Read operations return empty results if the DB doesn't exist.
    """

    def __init__(self, db_path: Path):
        self._path = db_path
        self._lock = threading.Lock()
        self._initialized = False

    def _ensure_db(self) -> sqlite3.Connection:
        """Create DB and schema if needed. Returns a connection."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._path))
        conn.row_factory = sqlite3.Row
        if not self._initialized:
            conn.execute(_CREATE_TABLE)
            for idx_sql in _CREATE_INDEXES:
                conn.execute(idx_sql)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
            self._initialized = True
        return conn

    def _connect_readonly(self) -> Optional[sqlite3.Connection]:
        """Connect for reads. Returns None if DB doesn't exist."""
        if not self._path.exists():
            return None
        conn = sqlite3.connect(str(self._path))
        conn.row_factory = sqlite3.Row
        return conn

    def record(self, record: UsageRecord) -> None:
        """Insert a usage record."""
        with self._lock:
            conn = self._ensure_db()
            try:
                conn.execute(
                    """INSERT INTO usage_log
                       (timestamp, agent, model, session_id, schedule_id,
                        input_tokens, output_tokens, total_tokens, cached_tokens,
                        cost, duration_ms)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record.timestamp,
                        record.agent,
                        record.model,
                        record.session_id,
                        record.schedule_id,
                        record.input_tokens,
                        record.output_tokens,
                        record.total_tokens,
                        record.cached_tokens,
                        record.cost,
                        record.duration_ms,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def query(
        self,
        agent: Optional[str] = None,
        schedule_id: Optional[str] = None,
        model: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 100,
    ) -> list[UsageRecord]:
        """Query usage records with optional filters.

        Args:
            agent: Filter by agent name
            schedule_id: Filter by schedule ID
            model: Filter by model name
            since: ISO datetime — only records after this time
            until: ISO datetime — only records before this time
            limit: Max records to return (default 100)

        Returns:
            List of UsageRecord, ordered by timestamp descending
        """
        conn = self._connect_readonly()
        if not conn:
            return []
        try:
            where, params = self._build_where(agent, schedule_id, model, since, until)
            sql = f"SELECT * FROM usage_log {where} ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_record(row) for row in rows]
        finally:
            conn.close()

    def summary(
        self,
        agent: Optional[str] = None,
        schedule_id: Optional[str] = None,
        model: Optional[str] = None,
        since: Optional[str] = None,
    ) -> dict:
        """Get aggregate usage summary.

        Returns:
            Dict with total_tokens, input_tokens, output_tokens, total_cost,
            run_count, avg_cost_per_run. All zeroes if no matching records.
        """
        conn = self._connect_readonly()
        empty = {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "total_cost": 0.0,
            "run_count": 0,
            "avg_cost_per_run": 0.0,
        }
        if not conn:
            return empty
        try:
            where, params = self._build_where(agent, schedule_id, model, since)
            sql = f"""SELECT
                        COUNT(*) as run_count,
                        COALESCE(SUM(total_tokens), 0) as total_tokens,
                        COALESCE(SUM(input_tokens), 0) as input_tokens,
                        COALESCE(SUM(output_tokens), 0) as output_tokens,
                        COALESCE(SUM(cached_tokens), 0) as cached_tokens,
                        COALESCE(SUM(cost), 0.0) as total_cost
                      FROM usage_log {where}"""
            row = conn.execute(sql, params).fetchone()
            if not row or row["run_count"] == 0:
                return empty
            run_count = row["run_count"]
            total_cost = row["total_cost"]
            return {
                "total_tokens": row["total_tokens"],
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "cached_tokens": row["cached_tokens"],
                "total_cost": total_cost,
                "run_count": run_count,
                "avg_cost_per_run": total_cost / run_count if run_count > 0 else 0.0,
            }
        finally:
            conn.close()

    def aggregate(
        self,
        group_by: str = "day",
        agent: Optional[str] = None,
        schedule_id: Optional[str] = None,
        model: Optional[str] = None,
        since: Optional[str] = None,
    ) -> list[dict]:
        """Aggregate usage by time period.

        Args:
            group_by: "day", "week", or "month"
            agent: Filter by agent name
            schedule_id: Filter by schedule ID
            model: Filter by model name
            since: ISO datetime — only records after this time

        Returns:
            List of dicts with period, run_count, total_tokens, input_tokens,
            output_tokens, total_cost. Ordered by period ascending.
        """
        conn = self._connect_readonly()
        if not conn:
            return []

        fmt_map = {
            "day": "%Y-%m-%d",
            "week": "%Y-W%W",
            "month": "%Y-%m",
        }
        fmt = fmt_map.get(group_by, "%Y-%m-%d")

        try:
            where, params = self._build_where(agent, schedule_id, model, since)
            sql = f"""SELECT
                        strftime('{fmt}', timestamp) as period,
                        COUNT(*) as run_count,
                        COALESCE(SUM(total_tokens), 0) as total_tokens,
                        COALESCE(SUM(input_tokens), 0) as input_tokens,
                        COALESCE(SUM(output_tokens), 0) as output_tokens,
                        COALESCE(SUM(cached_tokens), 0) as cached_tokens,
                        COALESCE(SUM(cost), 0.0) as total_cost
                      FROM usage_log {where}
                      GROUP BY period
                      ORDER BY period ASC"""
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def _build_where(
        self,
        agent: Optional[str] = None,
        schedule_id: Optional[str] = None,
        model: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> tuple[str, list]:
        """Build WHERE clause and params from filters."""
        clauses = []
        params: list = []
        if agent:
            clauses.append("agent = ?")
            params.append(agent)
        if schedule_id:
            clauses.append("schedule_id = ?")
            params.append(schedule_id)
        if model:
            clauses.append("model = ?")
            params.append(model)
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until:
            clauses.append("timestamp <= ?")
            params.append(until)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        return where, params

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> UsageRecord:
        """Convert a sqlite3.Row to a UsageRecord."""
        return UsageRecord(
            id=row["id"],
            timestamp=row["timestamp"],
            agent=row["agent"],
            model=row["model"],
            session_id=row["session_id"],
            schedule_id=row["schedule_id"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            total_tokens=row["total_tokens"],
            cached_tokens=row["cached_tokens"],
            cost=row["cost"],
            duration_ms=row["duration_ms"],
        )


# ── Module-level singleton ──

_store: Optional[UsageStore] = None
_store_lock = threading.Lock()


def get_usage_store(db_path: Optional[Path] = None) -> UsageStore:
    """Get or create the global UsageStore singleton.

    Args:
        db_path: Override DB path (mainly for testing). If None, uses
                 the default XDG data path.
    """
    global _store
    if _store is not None and db_path is None:
        return _store
    with _store_lock:
        if _store is not None and db_path is None:
            return _store
        if db_path is None:
            from tsugite.config import get_xdg_data_path

            db_path = get_xdg_data_path() / "usage.db"
        store = UsageStore(db_path)
        if db_path is None or _store is None:
            _store = store
        return store


def reset_usage_store() -> None:
    """Reset the global singleton (for testing)."""
    global _store
    with _store_lock:
        _store = None
