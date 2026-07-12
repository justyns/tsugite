"""CCDriverAdapter: the adapter-plugin entry point.

Exposes ONE public HTTP route (the per-job hook receiver, token-in-path, no daemon
auth) and ONE job executor ("cc"). The gateway mounts the route under
/api/plugins/cc_driver and registers the executor on the jobs orchestrator, handing
this adapter the orchestrator via set_jobs_orchestrator so the hook route can call
complete_worker/fail_worker.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from tsugite_daemon.adapters.base import BaseAdapter
from tsugite_daemon.job_store import JobState

from tsugite_cc_driver.executor import CCExecutor, DriveStateStore
from tsugite_cc_driver.hooks import decide_stop, decide_stop_failure, notification_attention

logger = logging.getLogger(__name__)


class CCDriverConfig(BaseModel):
    enabled: bool = False
    claude_binary: str = "claude"
    permission_mode: str = "bypassPermissions"
    # Autonomous default is sandboxed: filesystem isolation pairs with the
    # bypass/skip-permissions trust workaround so a driven claude can't write
    # outside the job workspace. Network stays on (it needs the API).
    sandbox: bool = True
    max_consecutive_continues: int = 5
    completion_marker: str = "CCDRIVER_GOAL_COMPLETE"
    base_url: str = "http://127.0.0.1:8374"
    state_dir: Optional[str] = None


def _resolve_state_dir(state_dir: Optional[str]) -> str:
    """Where per-job settings.json files live. Defaults to the XDG data path, same
    as every other tsugite subsystem (see tsugite.config.get_xdg_data_path)."""
    if state_dir:
        return str(Path(state_dir).expanduser())
    from tsugite.config import get_xdg_data_path

    return str(get_xdg_data_path("cc-driver"))


class CCDriverAdapter(BaseAdapter):
    """Not tied to a single agent - it's a job executor + hook receiver, so it
    skips BaseAdapter's agent/workspace init and sets only what the gateway wiring
    touches."""

    def __init__(self, config: CCDriverConfig, session_store=None, identity_map=None):
        self.agent_name = "cc_driver"
        self.session_store = session_store
        self._identity_map = identity_map or {}
        self.event_bus = None  # set by attach_plugin_http
        self.http_check_auth = None  # set by attach_plugin_http
        self.config = config
        self.config.state_dir = _resolve_state_dir(config.state_dir)
        self._drive_state = DriveStateStore()
        self._executor = CCExecutor(self.config, self._drive_state)
        self._orchestrator = None

    # ── plugin wiring ──

    def get_public_http_routes(self) -> list:
        # Unauthenticated: Claude Code can't send a daemon bearer token. Access is
        # gated by the random per-job token in the path (unknown token -> 404).
        return [Route("/hook/{token}", self._hook, methods=["POST"])]

    def get_job_executors(self) -> dict:
        return {"cc": self._executor}

    def set_jobs_orchestrator(self, orchestrator) -> None:
        """Injected by the gateway at executor registration so the hook route and
        the executor can report job outcomes."""
        self._orchestrator = orchestrator
        self._executor.orchestrator = orchestrator

    async def start(self) -> None:  # BaseAdapter contract - nothing to run
        return None

    async def stop(self) -> None:
        return None

    # ── hook receiver ──

    async def _hook(self, request: Request) -> JSONResponse:
        token = request.path_params["token"]
        state = self._drive_state.by_token(token)
        if state is None:
            # Orphaned/stale claude (daemon restarted, job gone) -> 404 so it just stops.
            return JSONResponse({"error": "unknown job token"}, status_code=404)

        job_id = state.job_id
        orch = self._orchestrator
        job = orch.get_job(job_id) if orch is not None else None
        if job is None or job.state != JobState.RUNNING.value:
            # Not RUNNING (verifying/terminal): allow claude to stop; the verifier
            # or a retry drives the next attempt.
            return JSONResponse({})

        try:
            payload = await request.json()
        except Exception:
            payload = {}

        # Every payload carries these; record them so a respawn can --resume.
        if payload.get("session_id"):
            state.cc_session_id = payload["session_id"]
        if payload.get("transcript_path"):
            state.transcript_path = payload["transcript_path"]

        event = payload.get("hook_event_name")
        if event == "Stop":
            decision = decide_stop(
                payload,
                consecutive_continues=state.consecutive_continues,
                max_consecutive_continues=self.config.max_consecutive_continues,
                completion_marker=self.config.completion_marker,
            )
            state.consecutive_continues = decision.new_consecutive_continues
            if decision.complete:
                await orch.complete_worker(job_id, decision.summary or "")
            return JSONResponse(decision.response)

        if event == "StopFailure":
            await orch.fail_worker(job_id, decide_stop_failure(payload))
            return JSONResponse({})

        if event == "Notification":
            message = notification_attention(payload)
            if message and self.event_bus is not None:
                try:
                    self.event_bus.emit("needs_attention", {"job_id": job_id, "message": message})
                except Exception:
                    logger.debug("cc-driver: needs_attention emit failed for job '%s'", job_id)
            return JSONResponse({})

        # Unknown/other events: acknowledge without driving.
        return JSONResponse({})


def create_adapter(*, config, agents_config, session_store, identity_map):
    """Adapter-plugin factory (tsugite.adapters entry point).

    `config` is the daemon.yaml plugins.cc_driver dict; agents_config is unused
    (cc-driver is not agent-scoped).

    Returns None (the gateway loop skips it) unless `enabled` is truthy - cc-driver
    spawns claude with skip-permissions, so it must be explicit opt-in rather than
    silently activated just by being installed.
    """
    cfg = CCDriverConfig(**(config or {}))
    if not cfg.enabled:
        logger.info("cc-driver plugin installed but disabled (set plugins.cc_driver.enabled: true to activate)")
        return None
    return CCDriverAdapter(cfg, session_store=session_store, identity_map=identity_map)
