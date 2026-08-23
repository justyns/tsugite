"""ActivityMixin: the recent-activity feed.

Merges the history event log (finished session runs, compactions), the job store
and the scheduler into one reverse-chronological list.

A one-off schedule is dropped from the store once its run finishes, so its run
never reaches this feed.
"""

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from tsugite.renderer import parse_iso_utc
from tsugite_daemon.session_store import SESSION_END_EVENT_TYPES

if TYPE_CHECKING:
    from tsugite.history.models import Event
    from tsugite_daemon.session_store import Session

_ACTIVITY_TYPES = ("session", "job", "schedule", "compaction")

_DEFAULT_LIMIT = 50
_MAX_LIMIT = 200

_EPOCH = datetime.min.replace(tzinfo=timezone.utc)

# `session_end` is absent: it carries its own status payload, mapped by
# _SESSION_END_STATUS.
_EVENT_STATUS = {
    "session_complete": "ok",
    "final_result": "ok",
    "session_error": "error",
    "error": "error",
    "session_cancelled": "cancelled",
    "cancelled": "cancelled",
}
# "interrupted" is max_turns being hit, not a user cancel, so it reads as a failure.
_SESSION_END_STATUS = {"success": "ok", "error": "error", "cancelled": "cancelled", "interrupted": "error"}
_STATUS_LABEL = {"ok": "completed", "error": "failed", "cancelled": "cancelled"}

_JOB_STATUS = {"done": "ok", "cancelled": "cancelled", "stuck": "error", "errored": "error"}
_RUN_STATUS = {"success": "ok", "error": "error", "skipped": "skipped"}


def _one_line(text: Optional[str], width: int) -> str:
    lines = (text or "").strip().splitlines()
    return lines[0][:width] if lines else ""


def _entry(
    *,
    kind: str,
    key: str,
    timestamp: str,
    title: str,
    summary: str = "",
    status: Optional[str] = None,
    label: str = "",
    session_id: Optional[str] = None,
    job_id: Optional[str] = None,
    schedule_id: Optional[str] = None,
) -> dict:
    return {
        "id": f"{kind}:{key}",
        "type": kind,
        "timestamp": timestamp,
        "title": title,
        "summary": summary,
        "status": status,
        "label": label,
        "session_id": session_id,
        "job_id": job_id,
        "schedule_id": schedule_id,
    }


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    # parse_iso_utc does not force a tz; a naive result would raise on its first
    # comparison against an aware one.
    ts = parse_iso_utc(value) if value else None
    if ts is None:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _sort_key(entry: dict) -> datetime:
    return _parse_ts(entry["timestamp"]) or _EPOCH


def _session_title(session: "Optional[Session]", session_id: str) -> str:
    if session is None:
        return session_id
    return session.title or _one_line(session.prompt, 80) or session.agent or session_id


def _session_entry(session_id: str, session: "Optional[Session]", event: "Event") -> dict:
    data = event.data
    if event.type == "session_end":
        status = _SESSION_END_STATUS.get(data.get("status"))
    else:
        status = _EVENT_STATUS.get(event.type)
    detail = data.get("error") or data.get("error_message") or data.get("result_preview") or ""
    return _entry(
        kind="session",
        key=f"{session_id}:{event.id}",
        timestamp=event.ts.isoformat(),
        title=_session_title(session, session_id),
        summary=_one_line(str(detail), 200),
        status=status,
        label=_STATUS_LABEL.get(status, "finished"),
        session_id=session_id,
    )


def _compaction_entry(session_id: str, session: "Optional[Session]", event: "Event") -> dict:
    replaced = event.data.get("replaced_count")
    retained = event.data.get("retained_count")
    summary = f"{replaced} turns compacted, {retained} kept" if replaced is not None else ""
    return _entry(
        kind="compaction",
        key=f"{session_id}:{event.id}",
        timestamp=event.ts.isoformat(),
        title=_session_title(session, session_id),
        summary=summary,
        status="ok",
        label="compacted",
        session_id=session_id,
    )


class ActivityMixin:
    def _activity_routes(self) -> list:
        return [
            Route("/api/activity", self._api_activity, methods=["GET"]),
        ]

    async def _api_activity(self, request: Request) -> JSONResponse:
        """Recent activity, newest first.

        `types` is a comma-separated subset of _ACTIVITY_TYPES and `limit` bounds
        the response (default 50, capped at 200).
        """
        if err := self._check_auth(request):
            return err
        raw_types = request.query_params.get("types", "")
        kinds = [t.strip() for t in raw_types.split(",") if t.strip()] or list(_ACTIVITY_TYPES)
        unknown = sorted(set(kinds) - set(_ACTIVITY_TYPES))
        if unknown:
            return JSONResponse({"error": f"unknown activity types: {', '.join(unknown)}"}, status_code=400)
        raw_limit = request.query_params.get("limit")
        if raw_limit:
            try:
                limit = max(1, min(int(raw_limit), _MAX_LIMIT))
            except ValueError:
                return JSONResponse({"error": f"limit must be an integer: {raw_limit}"}, status_code=400)
        else:
            limit = _DEFAULT_LIMIT

        # Off the loop: latest_event_per_session group-bys the whole events table, ~100ms
        # on a large history.db. The schedule collector stays on the loop on purpose - it
        # reads run_history race-free only because the scheduler mutates it on this loop.
        entries = await asyncio.to_thread(self._activity_history_entries, kinds, limit)
        if "job" in kinds:
            entries.extend(self._activity_job_entries(limit))
        if "schedule" in kinds:
            entries.extend(self._activity_schedule_entries(limit))
        entries.sort(key=_sort_key, reverse=True)
        return JSONResponse({"entries": entries[:limit]})

    def _activity_history_entries(self, kinds: list[str], limit: int) -> list[dict]:
        """Session-run and compaction rows from the shared event log.

        A session contributes one row, its newest end-of-run event; each
        compaction is its own row.
        """
        from tsugite.history import get_history_backend

        backend = get_history_backend()
        rows: list[tuple[str, "Event"]] = []
        previews: dict[str, str] = {}
        if "session" in kinds:
            rows.extend(backend.latest_event_per_session(types=sorted(SESSION_END_EVENT_TYPES), limit=limit))
            # A success session_end carries no text; the turn's answer lives on the
            # final_result event recorded just before it.
            previews = {
                sid: _one_line(str(event.data.get("result") or ""), 200)
                for sid, event in backend.latest_event_per_session(types=["final_result"], limit=limit)
            }
        if "compaction" in kinds:
            rows.extend(backend.recent_events(types=["compaction"], limit=limit))
        if not rows:
            return []

        sessions = self._daemon_sessions({session_id for session_id, _event in rows})
        entries: list[dict] = []
        for session_id, event in rows:
            if event.type == "compaction":
                entries.append(_compaction_entry(session_id, sessions.get(session_id), event))
            else:
                entry = _session_entry(session_id, sessions.get(session_id), event)
                if not entry["summary"]:
                    entry["summary"] = previews.get(session_id, "")
                entries.append(entry)
        return entries

    def _daemon_sessions(self, wanted: set[str]) -> "dict[str, Session]":
        """Daemon session rows for the ids being rendered, by id.

        A CLI session, or one already pruned, has events but no row here.
        """
        if self.session_runner is None or not wanted:
            return {}
        return {s.id: s for s in self.session_runner.store.list_sessions(include_superseded=True) if s.id in wanted}

    def _activity_job_entries(self, limit: int) -> list[dict]:
        if self.job_store is None:
            return []
        store = self.job_store
        terminal = [j for j in store.list_all() if j.state in store.terminal_states]
        terminal.sort(key=lambda j: j.resolved_at or j.updated_at, reverse=True)
        return [
            _entry(
                kind="job",
                key=job.id,
                timestamp=job.resolved_at or job.updated_at,
                title=_one_line(job.prompt, 120) or job.id,
                summary=_one_line(job.error, 200),
                status=_JOB_STATUS.get(job.state),
                label=job.state,
                session_id=job.parent_session_id,
                job_id=job.id,
            )
            for job in terminal[:limit]
        ]

    def _activity_schedule_entries(self, limit: int) -> list[dict]:
        if self.scheduler is None:
            return []
        entries: list[dict] = []
        for schedule in self.scheduler.list():
            for run in schedule.run_history:
                timestamp = run.get("timestamp") or ""
                if _parse_ts(timestamp) is None:
                    # An unparseable timestamp would sort to the bottom and render as "never".
                    continue
                status = run.get("status")
                entries.append(
                    _entry(
                        kind="schedule",
                        key=f"{schedule.id}:{timestamp}",
                        timestamp=timestamp,
                        title=schedule.id,
                        summary=_one_line(run.get("error") or schedule.prompt, 200),
                        status=_RUN_STATUS.get(status),
                        label=status or "ran",
                        session_id=run.get("session_id"),
                        schedule_id=schedule.id,
                    )
                )
        entries.sort(key=_sort_key, reverse=True)
        return entries[:limit]
