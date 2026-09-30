"""Synchronous client for the daemon HTTP API. Stdlib only, so a plain script can import it.

    from tsugite.daemon_client import DaemonClient

    client = DaemonClient()
    job = client.create_job(user_id="me", task="rebase the branch")
    client.cancel_job(job["id"])

Host, port and token arguments apply when given (default http://127.0.0.1:8374). Without
them, the URL comes from TSUGITE_DAEMON_URL and the token from TSUGITE_DAEMON_TOKEN.
"""

import json
import os
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from tsugite.constants import DEFAULT_DAEMON_HOST, DEFAULT_DAEMON_PORT


class DaemonError(Exception):
    """The daemon answered with an error status. `status` is the HTTP code."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class DaemonConnectionError(DaemonError):
    """The daemon could not be reached."""


class DaemonClient:
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        token: Optional[str] = None,
    ):
        env_url = os.environ.get("TSUGITE_DAEMON_URL") if host is None and port is None else None
        base_url = env_url or f"http://{host or DEFAULT_DAEMON_HOST}:{port or DEFAULT_DAEMON_PORT}"
        self.base_url = base_url.rstrip("/")
        self.token = token if token is not None else os.environ.get("TSUGITE_DAEMON_TOKEN")

    def request(
        self,
        method: str,
        path: str,
        body: Optional[dict] = None,
        query: Optional[dict] = None,
        timeout: float = 10,
    ) -> Any:
        """Send one request and return the decoded JSON body (None when empty)."""
        url = self.base_url + path
        if query:
            url += "?" + urlencode(query)
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"

        try:
            with urlopen(
                Request(url, data=data, headers=headers, method=method),
                timeout=timeout,
            ) as resp:
                raw = resp.read()
        except HTTPError as e:
            raise DaemonError(_error_message(e), e.code) from e
        except URLError as e:
            raise DaemonConnectionError(f"Could not connect to daemon at {self.base_url}: {e.reason}") from e
        return json.loads(raw) if raw else None

    def list_chat_sessions(self) -> list[dict]:
        return self.request("GET", "/api/chat/sessions")["sessions"]

    def compact_chat(self, user_id: str) -> dict:
        return self.request("POST", "/api/chat/compact", {"user_id": user_id}, timeout=120)

    def list_sessions(self) -> list[dict]:
        return self.request("GET", "/api/sessions")["sessions"]

    def set_primary_session(self, session_id: str) -> dict:
        return self.request("POST", f"/api/sessions/{quote(session_id, safe='')}/set-primary", {})

    def clear_primary_session(self, user_id: str) -> dict:
        return self.request("POST", "/api/sessions/clear-primary", query={"user_id": user_id})

    def list_schedules(self) -> list[dict]:
        return self.request("GET", "/api/schedules")["schedules"]

    def get_schedule(self, schedule_id: str) -> dict:
        return self.request("GET", _schedule_path(schedule_id))

    def create_schedule(self, **fields) -> dict:
        """Fields as the API takes them: id, prompt, schedule_type, cron_expr, run_at, timezone, ..."""
        return self.request("POST", "/api/schedules", fields)

    def cleanup_schedules(self) -> dict:
        return self.request("POST", "/api/schedules/cleanup")

    def delete_schedule(self, schedule_id: str) -> dict:
        return self.request("DELETE", _schedule_path(schedule_id))

    def enable_schedule(self, schedule_id: str) -> dict:
        return self.request("POST", _schedule_path(schedule_id, "enable"))

    def disable_schedule(self, schedule_id: str) -> dict:
        return self.request("POST", _schedule_path(schedule_id, "disable"))

    def run_schedule(self, schedule_id: str) -> dict:
        return self.request("POST", _schedule_path(schedule_id, "run"))

    def list_jobs(self, state: Optional[str] = None, limit: Optional[int] = None) -> list[dict]:
        query = {k: v for k, v in (("state", state), ("limit", limit)) if v is not None}
        return self.request("GET", "/api/jobs", query=query)["jobs"]

    def create_job(self, user_id: str, task: str, **fields) -> dict:
        """Extra fields (repo, model, agent, acceptance_criteria, ...) pass through to POST /api/jobs."""
        return self.request("POST", "/api/jobs", {"user_id": user_id, "task": task, **fields})

    def cancel_job(self, job_id: str, reason: Optional[str] = None) -> dict:
        return self.request("POST", _job_path(job_id, "cancel"), {"reason": reason} if reason else {})

    def mark_job_done(self, job_id: str, reason: Optional[str] = None) -> dict:
        return self.request("POST", _job_path(job_id, "mark-done"), {"reason": reason} if reason else {})

    def retry_job(self, job_id: str, **fields) -> dict:
        """Fields: hint, model, verifier_model, reset_counter, fresh_workspace."""
        return self.request("POST", _job_path(job_id, "retry"), fields)


def _schedule_path(schedule_id: str, action: str = "") -> str:
    return f"/api/schedules/{quote(schedule_id, safe='')}" + (f"/{action}" if action else "")


def _job_path(job_id: str, action: str) -> str:
    return f"/api/jobs/{quote(job_id, safe='')}/{action}"


def _error_message(e: HTTPError) -> str:
    text = e.read().decode(errors="replace")
    try:
        return json.loads(text)["error"]
    except (ValueError, KeyError, TypeError):
        return text or e.reason
