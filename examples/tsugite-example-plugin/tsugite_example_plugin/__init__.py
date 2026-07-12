"""Example tsugite plugin - every author-facing extension point in one file.

Two entry points in pyproject.toml wire it up. The first (module-only) runs the
@tool / @hook / @subscribe decorators at import; the second exposes the daemon
adapter (HTTP routes + job executors):

    [project.entry-points."tsugite.plugins"]
    example = "tsugite_example_plugin"                  # tool + hook + subscriber

    [project.entry-points."tsugite.adapters"]
    example = "tsugite_example_plugin:create_adapter"    # HTTP routes + job executors

Enable the adapter half in daemon.yaml:

    plugins:
      example:
        enabled: true
        greeting: "hi"

Other, more specialized groups (each replaces a whole subsystem, so they get
their own file normally, not shown here): `tsugite.providers` (an LLM backend),
`tsugite.attachments` (a URL/file handler), `tsugite.sandbox` (an exec sandbox),
`tsugite.history` (a session-history backend), `tsugite.secrets` (a secret store),
and `tsugite.executors` (a code-execution backend - unrelated to the *job*
executors shown below via `get_job_executors()`, despite the similar name).

Note: this single file imports the daemon adapter base, so as written it needs the
`[daemon]` extra installed. In a real plugin, put the daemon-only adapter in its
own module so the tool + hook + subscriber half still loads on a non-daemon install.
"""

from __future__ import annotations

import logging

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from tsugite_daemon.adapters.base import BaseAdapter

from tsugite.events.bus import subscribe
from tsugite.hooks import hook
from tsugite.tools import tool

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. AGENT TOOL - a Python function agents can call. The signature (with type
#    hints) and the docstring become the tool schema the model sees. Return any
#    JSON-serializable value. Flags: require_daemon / parent_only / interactive_only.
# ─────────────────────────────────────────────────────────────────────────────
@tool(category="example")
def greet(name: str, excited: bool = False) -> str:
    """Return a greeting for `name`. Set `excited` for an exclamation mark."""
    return f"Hello, {name}{'!' if excited else '.'}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. LIFECYCLE HOOK - a Python callable fired at an agent-loop phase. It receives
#    a context dict. Sync or async. Phases: pre_message, pre_context_build,
#    post_context_build, pre_llm_call, pre_tool_call, pre_response, post_response,
#    pre_compact, post_compact, session_end. (The tool phases pre_tool_call /
#    post_tool need tools=[...] to say which tools match; "*" = all.)
#    On the context-build phases a returned string is injected into the prompt; on
#    other phases the return is ignored (do side effects like logging/notifying).
# ─────────────────────────────────────────────────────────────────────────────
@hook("pre_tool_call", tools=["*"])
def log_tool_use(context: dict) -> None:
    """Fire before any tool call. Side effect only (returns are ignored here)."""
    logger.info("example plugin: about to run tool '%s'", context.get("tool"))


# ─────────────────────────────────────────────────────────────────────────────
# 3. EVENT SUBSCRIBER - react to events on the daemon's event bus. Distinct from
#    a hook: hooks fire at agent-loop phases (and can inject context); subscribers
#    just observe bus events. Filter by name or a predicate.
# ─────────────────────────────────────────────────────────────────────────────
@subscribe(event_name="tool_call")
def on_tool_call(event) -> None:
    """Observe every tool invocation on the bus."""
    logger.info("example plugin: bus tool_call %s", event.tool_name)


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
    attributes the gateway wiring touches (event_bus, http_check_auth)."""

    def __init__(self, config: dict):
        self.config = config
        self.event_bus = None  # set by the gateway (SSE broadcast bus)
        self.http_check_auth = None  # set by the gateway (daemon bearer-token check)
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
