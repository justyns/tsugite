"""SQLite implementation of the history battery (base.py protocols).

The event log is one ``events`` row per Event (``id`` = rowid gives ordering +
identity); a ``sessions`` row holds metadata, maintained aggregates, and lineage.
``record``/``record_many`` are the public transaction owners; ``_insert_event`` and
``_fold_event`` are transaction-agnostic so compaction/branching can reuse them under
a single ``BEGIN IMMEDIATE`` (no nested transactions).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional

from .models import Event, _parse_iso, dedup_model_request_data, iso_utc
from .sqlite_conn import connect_history_db
from .storage import SessionSummary, generate_session_id, get_history_dir


class SessionAlreadyExistsError(Exception):
    """create() was called with an explicit session_id that already exists."""


def _db_path(explicit: str | Path | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    env = os.getenv("TSUGITE_HISTORY_DB")
    if env:
        return Path(env)
    return get_history_dir() / "history.db"


def _dump_data(data: dict) -> str:
    return json.dumps({k: v for k, v in data.items() if v is not None}, default=str)


def _snippet(text: str, query: str, width: int = 80) -> str:
    lo = text.lower().find(query.lower())
    if lo < 0:
        return text[:width]
    start = max(0, lo - width // 2)
    return text[start : start + width]


def _as_iso(value) -> str:
    return iso_utc(value) if isinstance(value, datetime) else value


_MAX_WRITE_ATTEMPTS = 3
_RETRY_BASE_SECONDS = 0.05


def _is_locked(exc: sqlite3.OperationalError) -> bool:
    msg = str(exc).lower()
    return "locked" in msg or "busy" in msg


def _run_write(conn: sqlite3.Connection, body, *, attempts: int = _MAX_WRITE_ATTEMPTS, sleep=time.sleep):
    """Run ``body()`` inside one ``BEGIN IMMEDIATE``, retrying on a locked/busy database.

    The single transaction owner: compaction/branching pass a body that calls the
    txn-agnostic insert/fold helpers, so there's never a nested ``BEGIN``. The retry is a
    second line of defense past ``busy_timeout`` so writes don't fail under contention.
    """
    for attempt in range(attempts):
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if _is_locked(exc) and attempt < attempts - 1:
                sleep(_RETRY_BASE_SECONDS * (attempt + 1))
                continue
            raise
        try:
            result = body()
            conn.execute("COMMIT")
            return result
        except sqlite3.OperationalError as exc:
            conn.execute("ROLLBACK")
            if _is_locked(exc) and attempt < attempts - 1:
                sleep(_RETRY_BASE_SECONDS * (attempt + 1))
                continue
            raise
        except Exception:
            conn.execute("ROLLBACK")
            raise


def _insert_event(conn: sqlite3.Connection, session_id: str, event: Event) -> int:
    """Insert one event (caller owns the transaction). Never preserves event.id."""
    cur = conn.execute(
        "INSERT INTO events(session_id, type, ts, data) VALUES (?, ?, ?, ?)",
        (session_id, event.type, iso_utc(event.ts), _dump_data(event.data)),
    )
    return int(cur.lastrowid)


def _fold_event(conn: sqlite3.Connection, session_id: str, event: Event, now_iso: str) -> None:
    """Fold one event into the sessions aggregates/metadata (caller owns the transaction).

    Incremental mirror of SessionSummary.from_events. ``updated_at`` is always
    wall-clock now (sidebar recency); ``last_event_ts`` tracks the max event ts (event
    semantics, legitimately old after copy-forward).
    """
    d = event.data
    t = event.type
    ts_iso = iso_utc(event.ts)
    conn.execute(
        "UPDATE sessions SET updated_at = ?, last_event_ts = MAX(COALESCE(last_event_ts, ''), ?) WHERE session_id = ?",
        (now_iso, ts_iso, session_id),
    )
    if t == "session_start":
        conn.execute(
            "UPDATE sessions SET agent=?, model=?, workspace=?, parent_session=?, created_at=? WHERE session_id=?",
            (
                d.get("agent"),
                d.get("model"),
                d.get("workspace"),
                d.get("parent_session"),
                ts_iso,
                session_id,
            ),
        )
    elif t == "user_input":
        conn.execute("UPDATE sessions SET turn_count = turn_count + 1 WHERE session_id = ?", (session_id,))
    elif t == "model_response":
        usage = d.get("usage") or {}
        tokens = int(usage.get("total_tokens") or 0) if isinstance(usage, dict) else 0
        conn.execute(
            "UPDATE sessions SET total_tokens = total_tokens + ?, total_cost = total_cost + ? WHERE session_id = ?",
            (tokens, float(d.get("cost") or 0.0), session_id),
        )
    elif t in ("code_execution", "tool_invocation"):
        conn.execute(
            "UPDATE sessions SET total_duration_ms = total_duration_ms + ? WHERE session_id = ?",
            (int(d.get("duration_ms") or 0), session_id),
        )
    elif t == "session_end":
        conn.execute(
            "UPDATE sessions SET status = ?, error_message = ?, ended_at = ? WHERE session_id = ?",
            (d.get("status"), d.get("error_message"), ts_iso, session_id),
        )


def _copy_session_events(
    conn: sqlite3.Connection,
    src_id: str,
    dst_id: str,
    *,
    min_id: Optional[int] = None,
    max_id: Optional[int] = None,
    scrub_state_delta: bool = True,
    now_iso: Optional[str] = None,
) -> None:
    """Copy events from ``src_id`` into ``dst_id`` (caller owns the transaction).

    Shared by branching (copy the head: ``max_id=cut``) and compaction (copy the tail:
    ``min_id=cut``). Strips ``state_delta`` from copied ``model_response`` events so the
    destination starts a fresh provider session instead of ``--resume``-ing into the
    source's, and drops ``session_end`` so a copied-forward session stays active.
    Destination rows get fresh ids; source ids are never preserved.
    """
    now_iso = now_iso or iso_utc()
    clauses = ["session_id = ?"]
    params: list[Any] = [src_id]
    if min_id is not None:
        clauses.append("id >= ?")
        params.append(min_id)
    if max_id is not None:
        clauses.append("id <= ?")
        params.append(max_id)
    sql = f"SELECT type, ts, data FROM events WHERE {' AND '.join(clauses)} ORDER BY id"
    for row in conn.execute(sql, params).fetchall():
        if row["type"] == "session_end":
            continue
        data = json.loads(row["data"])
        if scrub_state_delta and row["type"] == "model_response":
            data.pop("state_delta", None)
        event = Event(type=row["type"], ts=row["ts"], data=data)
        _insert_event(conn, dst_id, event)
        _fold_event(conn, dst_id, event, now_iso)


class SqliteSession:
    """Read/write handle for one conversation backed by SQLite."""

    def __init__(self, backend: "SqliteHistoryBackend", session_id: str):
        self.backend = backend
        self.session_id = session_id

    def record(self, type: str, *, ts: Optional[datetime] = None, **data: Any) -> None:
        self.record_many([Event(type=type, ts=ts or datetime.now(timezone.utc), data=data)])

    def record_many(self, events: Iterable[Event]) -> None:
        events = list(events)
        if not events:
            return
        conn = self.backend._conn()
        now_iso = iso_utc()

        def body():
            for event in events:
                _insert_event(conn, self.session_id, event)
                _fold_event(conn, self.session_id, event, now_iso)

        _run_write(conn, body)

    def iter_events(self, types: Optional[Iterable[str]] = None) -> Iterator[Event]:
        conn = self.backend._conn()
        if types is not None:
            types = list(types)
            placeholders = ",".join("?" * len(types))
            sql = f"SELECT id, type, ts, data FROM events WHERE session_id=? AND type IN ({placeholders}) ORDER BY id"
            rows = conn.execute(sql, (self.session_id, *types)).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, type, ts, data FROM events WHERE session_id=? ORDER BY id", (self.session_id,)
            ).fetchall()
        for row in rows:
            yield Event(id=row["id"], type=row["type"], ts=row["ts"], data=json.loads(row["data"]))

    def load_events(self) -> List[Event]:
        return list(self.iter_events())

    def summary(self) -> SessionSummary:
        conn = self.backend._conn()
        row = conn.execute("SELECT * FROM sessions WHERE session_id=?", (self.session_id,)).fetchone()
        s = SessionSummary()
        if row is None:
            return s
        s.agent = row["agent"]
        s.model = row["model"]
        s.workspace = row["workspace"]
        s.created_at = _parse_iso(row["created_at"]) if row["created_at"] else None
        s.parent_session = row["parent_session"]
        s.status = row["status"]
        s.error_message = row["error_message"]
        s.turn_count = row["turn_count"]
        s.total_tokens = row["total_tokens"]
        s.total_cost = row["total_cost"]
        s.total_duration_ms = row["total_duration_ms"]
        # functions_called / last_response_text aren't columns: a targeted read fills them.
        for event in self.iter_events(types=["code_execution", "tool_invocation", "model_response"]):
            if event.type == "code_execution":
                for fn in event.data.get("tools_called") or []:
                    s.functions_called.add(fn)
            elif event.type == "tool_invocation":
                name = event.data.get("name")
                if name:
                    s.functions_called.add(name)
            elif event.type == "model_response":
                s.last_response_text = event.data.get("raw_content", s.last_response_text)
        return s


class SqliteHistoryBackend:
    """Default history backend: a single SQLite database for all sessions."""

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = _db_path(db_path)

    def _conn(self) -> sqlite3.Connection:
        return connect_history_db(self._db_path)

    def _resolve_new_session_id(
        self, conn: sqlite3.Connection, *, explicit: Optional[str], agent: str, ts: Optional[datetime] = None
    ) -> str:
        """The id to insert: the explicit one (raise if taken) or a freshly generated, collision-free one.

        Caller owns the transaction so the existence checks see uncommitted siblings.
        """
        if explicit is not None:
            if self._row_exists(conn, explicit):
                raise SessionAlreadyExistsError(explicit)
            return explicit
        sid = generate_session_id(agent, ts)
        suffix = 1
        while self._row_exists(conn, sid):
            suffix += 1
            sid = f"{generate_session_id(agent, ts)}_{suffix}"
        return sid

    def create(
        self,
        agent_name: str,
        model: str,
        *,
        workspace: Optional[str] = None,
        parent_session: Optional[str] = None,
        session_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> SqliteSession:
        ts = timestamp or datetime.now(timezone.utc)
        conn = self._conn()
        sid: Optional[str] = None

        def body():
            nonlocal sid
            sid = self._resolve_new_session_id(conn, explicit=session_id, agent=agent_name, ts=ts)
            conn.execute("INSERT INTO sessions(session_id, created_at) VALUES (?, ?)", (sid, iso_utc(ts)))
            start = Event(
                type="session_start",
                ts=ts,
                data={
                    "agent": agent_name,
                    "model": model,
                    "workspace": workspace,
                    "parent_session": parent_session,
                },
            )
            _insert_event(conn, sid, start)
            _fold_event(conn, sid, start, iso_utc())

        _run_write(conn, body)
        return SqliteSession(self, sid)

    def load(self, session_id: str) -> SqliteSession:
        if not self.exists(session_id):
            raise FileNotFoundError(f"Session not found: {session_id}")
        return SqliteSession(self, session_id)

    def exists(self, session_id: str) -> bool:
        return self._row_exists(self._conn(), session_id)

    def get_meta(self, session_id: str) -> Optional[Event]:
        row = (
            self._conn()
            .execute(
                "SELECT id, type, ts, data FROM events WHERE session_id=? AND type='session_start' ORDER BY id LIMIT 1",
                (session_id,),
            )
            .fetchone()
        )
        if row is None:
            return None
        return Event(id=row["id"], type=row["type"], ts=row["ts"], data=json.loads(row["data"]))

    def list_sessions(
        self,
        *,
        workspace: Optional[str] = None,
        agent: Optional[str] = None,
        status: Optional[str] = None,
        since: Optional[datetime] = None,
        before: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        clauses: list[str] = []
        params: list[Any] = []
        for col, val in (("workspace", workspace), ("agent", agent), ("status", status)):
            if val is not None:
                clauses.append(f"{col} = ?")
                params.append(val)
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(_as_iso(since))
        if before is not None:
            clauses.append("created_at < ?")
            params.append(_as_iso(before))
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT session_id FROM sessions{where} ORDER BY COALESCE(updated_at, created_at) DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [r["session_id"] for r in self._conn().execute(sql, params).fetchall()]

    def count_events(self, session_id: str, *, type: Optional[str] = None) -> int:
        conn = self._conn()
        if type is None:
            return conn.execute("SELECT COUNT(*) FROM events WHERE session_id=?", (session_id,)).fetchone()[0]
        return conn.execute("SELECT COUNT(*) FROM events WHERE session_id=? AND type=?", (session_id, type)).fetchone()[
            0
        ]

    def ensure_session(self, session_id: str) -> SqliteSession:
        """Get-or-create a bare session row (no session_start), for telemetry targets.

        Never overwrites existing metadata - a later session_start folds it in.
        """
        conn = self._conn()
        if not self._row_exists(conn, session_id):

            def body():
                if not self._row_exists(conn, session_id):
                    conn.execute("INSERT INTO sessions(session_id, created_at) VALUES (?, ?)", (session_id, iso_utc()))

            _run_write(conn, body)
        return SqliteSession(self, session_id)

    def search(self, query: str, *, agent: Optional[str] = None, limit: int = 50) -> List[dict]:
        """Find sessions whose user/model/tool text matches ``query`` (SQL LIKE).

        Returns at most ``limit`` per-session hits, recency-ordered, each with a snippet.
        FTS seam: a future migration swaps the LIKE for an FTS5/tsvector match here.
        """
        like = f"%{query}%"
        sql = (
            "SELECT e.session_id, e.type, e.data FROM events e JOIN sessions s ON s.session_id = e.session_id "
            "WHERE ("
            "  json_extract(e.data, '$.text') LIKE ?"
            "  OR json_extract(e.data, '$.raw_content') LIKE ?"
            "  OR json_extract(e.data, '$.name') LIKE ?"
            ")"
        )
        params: list[Any] = [like, like, like]
        if agent is not None:
            sql += " AND s.agent = ?"
            params.append(agent)
        sql += " ORDER BY e.id DESC"
        hits: dict[str, dict] = {}
        for row in self._conn().execute(sql, params):
            sid = row["session_id"]
            if sid in hits:
                continue
            data = json.loads(row["data"])
            text = data.get("text") or data.get("raw_content") or data.get("name") or ""
            hits[sid] = {"session_id": sid, "type": row["type"], "snippet": _snippet(text, query)}
            if len(hits) >= limit:
                break
        return list(hits.values())

    def purge(self, *, older_than: Optional[datetime] = None) -> int:
        """Delete sessions last written before ``older_than`` (events cascade). Returns count."""
        if older_than is None:
            return 0
        cutoff = _as_iso(older_than)
        conn = self._conn()
        removed = 0

        def body():
            nonlocal removed
            cur = conn.execute("DELETE FROM sessions WHERE COALESCE(updated_at, created_at) < ?", (cutoff,))
            removed = cur.rowcount

        _run_write(conn, body)
        return removed

    def delete_session(self, session_id: str) -> bool:
        """Delete one session and its events (cascade). Returns True if a row was removed."""
        conn = self._conn()
        removed = 0

        def body():
            nonlocal removed
            cur = conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
            removed = cur.rowcount

        _run_write(conn, body)
        return removed > 0

    def export_jsonl(self, session_id: str) -> Iterator[str]:
        """Yield one JSONL line per event, byte-identical to the legacy file format.

        ``id`` is excluded so exported logs match pre-SQLite JSONL (which had no id).
        """
        for event in self.load(session_id).iter_events():
            yield event.model_dump_json(exclude={"id"}, exclude_none=True)

    def import_jsonl(self, paths: Iterable[Path], *, dry_run: bool = False) -> dict:
        """Import legacy per-session JSONL files into the database. Idempotent.

        Skips sessions already imported (by stem). Reads via the existing SessionStorage
        reader, which tolerates malformed lines. A file with no session_start still
        imports (a bare sessions row is synthesized from the first event's timestamp).
        """
        from .storage import SessionStorage

        report = {"imported": 0, "skipped": 0, "no_session_start": 0}
        for path in paths:
            path = Path(path)
            sid = path.stem
            if self.exists(sid):
                report["skipped"] += 1
                continue
            events = list(SessionStorage(path).iter_events())
            if not events:
                continue
            has_start = any(e.type == "session_start" for e in events)
            if not dry_run:
                conn = self._conn()

                def body(sid=sid, events=events):
                    conn.execute(
                        "INSERT INTO sessions(session_id, created_at) VALUES (?, ?)", (sid, iso_utc(events[0].ts))
                    )
                    now_iso = iso_utc()
                    for event in events:
                        if event.type == "model_request":
                            # Strip the legacy full messages array (the on-disk de-dup); reconstruction
                            # never reads it, so this shrinks an imported db dramatically.
                            event = Event(type=event.type, ts=event.ts, data=dedup_model_request_data(event.data))
                        _insert_event(conn, sid, event)
                        _fold_event(conn, sid, event, now_iso)
                    # Recency reflects the imported conversation's real last activity, not import
                    # time, so list/sidebar ordering stays chronological after a migration.
                    conn.execute("UPDATE sessions SET updated_at = last_event_ts WHERE session_id = ?", (sid,))

                _run_write(conn, body)
            report["imported"] += 1
            if not has_start:
                report["no_session_start"] += 1
        return report

    def create_branch(
        self,
        source_id: str,
        *,
        at_event_id: int,
        new_session_id: Optional[str] = None,
    ) -> str:
        """Fork ``source_id`` at ``at_event_id`` into an independent branch. Returns its id.

        Copies the head (events ``id <= at_event_id``) with provider state scrubbed; the
        source is untouched. Lineage is stored in the tree columns (branched_from_session_id /
        branch_point_event_id), distinct from the linear compaction chain.
        """
        conn = self._conn()
        cut = conn.execute("SELECT session_id, type FROM events WHERE id=?", (at_event_id,)).fetchone()
        if cut is None or cut["session_id"] != source_id:
            raise ValueError(f"Event {at_event_id} does not belong to session {source_id}")
        if cut["type"] == "session_start":
            raise ValueError("Cannot branch at session_start (nothing to continue)")
        meta = self.get_meta(source_id)
        agent = (meta.data.get("agent") if meta else None) or "agent"
        new_id: Optional[str] = None

        def body():
            nonlocal new_id
            new_id = self._resolve_new_session_id(conn, explicit=new_session_id, agent=agent)
            conn.execute(
                "INSERT INTO sessions(session_id, created_at, branched_from_session_id, branch_point_event_id) "
                "VALUES (?, ?, ?, ?)",
                (new_id, iso_utc(), source_id, at_event_id),
            )
            _copy_session_events(conn, source_id, new_id, max_id=at_event_id, scrub_state_delta=True)

        _run_write(conn, body)
        return new_id

    @staticmethod
    def _row_exists(conn: sqlite3.Connection, session_id: str) -> bool:
        return conn.execute("SELECT 1 FROM sessions WHERE session_id=? LIMIT 1", (session_id,)).fetchone() is not None
