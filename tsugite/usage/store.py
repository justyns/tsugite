"""SQLite-backed usage tracking store."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from tsugite.config import get_xdg_data_path

_DEFAULT_DB_DIR = "usage"
_DEFAULT_DB_NAME = "usage.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    session_id TEXT,
    agent TEXT,
    model TEXT,
    source TEXT,
    schedule_name TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    cost_usd REAL,
    duration_ms INTEGER,
    cache_creation_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage(timestamp);
CREATE INDEX IF NOT EXISTS idx_usage_agent ON usage(agent, timestamp);
CREATE INDEX IF NOT EXISTS idx_usage_model ON usage(model, timestamp);
CREATE INDEX IF NOT EXISTS idx_usage_session ON usage(session_id);
"""

_instance: UsageStore | None = None


def get_usage_store() -> UsageStore:
    """Get the singleton UsageStore instance."""
    global _instance
    if _instance is None:
        _instance = UsageStore()
    return _instance


class UsageStore:
    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_path = os.getenv("TSUGITE_USAGE_DB")
        if db_path is None:
            db_path = get_xdg_data_path(_DEFAULT_DB_DIR) / _DEFAULT_DB_NAME

        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
        return self._conn

    def record(
        self,
        *,
        session_id: str | None = None,
        agent: str | None = None,
        model: str | None = None,
        source: str | None = None,
        schedule_name: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        cost_usd: float | None = None,
        duration_ms: int | None = None,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
        timestamp: str | None = None,
    ) -> None:
        """Insert a usage record."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        conn = self._get_conn()
        conn.execute(
            """INSERT INTO usage (
                timestamp, session_id, agent, model, source, schedule_name,
                input_tokens, output_tokens, total_tokens, cost_usd,
                duration_ms, cache_creation_tokens, cache_read_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                timestamp, session_id, agent, model, source, schedule_name,
                input_tokens, output_tokens, total_tokens, cost_usd,
                duration_ms, cache_creation_tokens, cache_read_tokens,
            ),
        )
        conn.commit()

    def _build_where(
        self,
        *,
        agent: str | None = None,
        model: str | None = None,
        source: str | None = None,
        session_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> tuple[str, list]:
        conditions: list[str] = []
        params: list = []
        for col, val in [("agent", agent), ("model", model), ("source", source), ("session_id", session_id)]:
            if val is not None:
                conditions.append(f"{col} = ?")
                params.append(val)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        if until:
            conditions.append("timestamp < ?")
            params.append(until)
        clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        return clause, params

    def query(
        self,
        *,
        agent: str | None = None,
        model: str | None = None,
        source: str | None = None,
        session_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        where, params = self._build_where(
            agent=agent, model=model, source=source, session_id=session_id, since=since, until=until,
        )
        sql = f"SELECT * FROM usage {where} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        return [dict(row) for row in self._get_conn().execute(sql, params).fetchall()]

    def summary(
        self,
        *,
        agent: str | None = None,
        period: str = "day",
        since: str | None = None,
    ) -> list[dict]:
        """Aggregate usage by time period ('day', 'week', 'month')."""
        date_exprs = {"day": "date(timestamp)", "week": "strftime('%Y-W%W', timestamp)", "month": "strftime('%Y-%m', timestamp)"}
        date_expr = date_exprs.get(period, "date(timestamp)")

        where, params = self._build_where(agent=agent, since=since)
        sql = f"""
            SELECT {date_expr} as period,
                   COUNT(*) as runs,
                   SUM(total_tokens) as total_tokens,
                   SUM(cost_usd) as total_cost,
                   SUM(input_tokens) as input_tokens,
                   SUM(output_tokens) as output_tokens,
                   SUM(duration_ms) as total_duration_ms
            FROM usage {where}
            GROUP BY {date_expr}
            ORDER BY period DESC
        """
        return [dict(row) for row in self._get_conn().execute(sql, params).fetchall()]

    def _top_by(self, column: str, *, since: str | None = None, limit: int = 10) -> list[dict]:
        assert column in ("agent", "model")
        where, params = self._build_where(since=since)
        sql = f"""
            SELECT {column}, COUNT(*) as runs,
                   SUM(total_tokens) as total_tokens, SUM(cost_usd) as total_cost
            FROM usage {where}
            GROUP BY {column} ORDER BY total_cost DESC LIMIT ?
        """
        params.append(limit)
        return [dict(row) for row in self._get_conn().execute(sql, params).fetchall()]

    def top_agents(self, *, since: str | None = None, limit: int = 10) -> list[dict]:
        return self._top_by("agent", since=since, limit=limit)

    def top_models(self, *, since: str | None = None, limit: int = 10) -> list[dict]:
        return self._top_by("model", since=since, limit=limit)

    def total(self, *, since: str | None = None) -> dict:
        where, params = self._build_where(since=since)
        sql = f"""
            SELECT COUNT(*) as runs,
                   COALESCE(SUM(total_tokens), 0) as total_tokens,
                   COALESCE(SUM(cost_usd), 0) as total_cost,
                   COALESCE(SUM(input_tokens), 0) as input_tokens,
                   COALESCE(SUM(output_tokens), 0) as output_tokens
            FROM usage {where}
        """
        row = self._get_conn().execute(sql, params).fetchone()
        return dict(row) if row else {"runs": 0, "total_tokens": 0, "total_cost": 0.0}
