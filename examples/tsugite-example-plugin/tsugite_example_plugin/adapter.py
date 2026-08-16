"""The example plugin's daemon adapter - HTTP routes + job executors.

Kept in its own module (not __init__.py) so the tool / hook / subscriber half of
the plugin still imports on an install without the `[daemon]` extra; only the
daemon itself loads this module, via the tsugite.adapters entry point:

    [project.entry-points."tsugite.adapters"]
    example = "tsugite_example_plugin.adapter:create_adapter"
"""

from __future__ import annotations

import logging

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from tsugite_daemon.adapters.base import BaseAdapter

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 4. DAEMON ADAPTER - a long-lived object the daemon starts, that can mount HTTP
#    routes and register custom job executors. Only loaded when the daemon runs
#    and the plugin is enabled in daemon.yaml.
# ─────────────────────────────────────────────────────────────────────────────
class EchoExecutor:
    """A custom job executor. A job created with executor="echo" runs through
    this instead of an agent. The contract is duck-typed:

        async start(job, followup)  - do the work; report back via the orchestrator
        async cancel(job)           - tear down (called before the worktree is pruned)

    Report outcomes with orchestrator.complete_worker(job_id, summary) /
    fail_worker(job_id, error); those feed into the SAME verifier / retry / stuck
    machinery every job uses. Read job state with orchestrator.get_job(job_id) and
    surface a live terminal with orchestrator.attach_worker_terminal(job_id, tid).
    """

    def __init__(self):
        self.orchestrator = None  # injected via adapter.set_jobs_orchestrator

    async def start(self, job, followup=None) -> None:
        # A real executor would spawn a process here; we just echo the prompt back
        # and complete immediately, letting the normal verifier grade it.
        await self.orchestrator.complete_worker(job.id, f"echo: {job.prompt}")

    async def cancel(self, job) -> None:
        return None


class ExampleAdapter(BaseAdapter):
    """Not agent-scoped, so it skips BaseAdapter.__init__ and sets only the
    attributes the gateway wiring touches (event_bus)."""

    def __init__(self, config: dict):
        self.config = config
        self.event_bus = None  # set by the gateway (SSE broadcast bus)
        self._executor = EchoExecutor()

    # -- HTTP routes: mounted under /api/plugins/example/ --

    def get_http_routes(self) -> list:
        # Auto-wrapped with the daemon bearer-token check, so assume an
        # authenticated (web-UI-style) caller. GET /api/plugins/example/ping
        return [Route("/ping", self._ping, methods=["GET"])]

    def get_public_http_routes(self) -> list:
        # NO auth wrapper - do your own. Here a token-in-path guards it, like an
        # inbound webhook. POST /api/plugins/example/webhook/{token}
        return [Route("/webhook/{token}", self._webhook, methods=["POST"])]

    async def _ping(self, request: Request) -> JSONResponse:
        return JSONResponse({"pong": True, "greeting": self.config.get("greeting", "hello")})

    async def _webhook(self, request: Request) -> JSONResponse:
        if request.path_params["token"] != self.config.get("webhook_token"):
            return JSONResponse({"error": "bad token"}, status_code=403)
        # Emit an SSE event any web-UI client can subscribe to on /api/events.
        if self.event_bus is not None:
            self.event_bus.emit("example_event", {"body": await request.body()})
        return JSONResponse({"ok": True})

    # -- Job executors --

    def get_job_executors(self) -> dict:
        return {"echo": self._executor}

    def set_jobs_orchestrator(self, orchestrator) -> None:
        # Hand the executor a handle so it can report job outcomes. (A real adapter
        # with HTTP handlers that touch jobs would keep its own ref too.)
        self._executor.orchestrator = orchestrator

    # -- Lifecycle (awaited by the gateway on boot / shutdown) --

    async def start(self) -> None:
        logger.info("example adapter started")

    async def stop(self) -> None:
        logger.info("example adapter stopped")


def create_adapter(*, config, agents_config, session_store, identity_map):
    """Adapter factory (the tsugite.adapters entry point). `config` is the
    daemon.yaml plugins.example dict. Return None to stay disabled - the gateway
    skips a None, so `enabled` is a real opt-in switch."""
    config = config or {}
    if not config.get("enabled"):
        return None
    return ExampleAdapter(config)
