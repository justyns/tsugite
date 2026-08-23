"""HTTPServer: the Starlette ASGI app assembly and core/auth handlers."""

import asyncio
import json
from pathlib import Path
from typing import Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from tsugite_daemon.adapters.http.activity import ActivityMixin
from tsugite_daemon.adapters.http.agents import AgentsMixin
from tsugite_daemon.adapters.http.files import FilesMixin
from tsugite_daemon.adapters.http.helpers import (
    ActiveChat,
    HTTPAgentAdapter,
    logger,
)
from tsugite_daemon.adapters.http.hooks_config import HooksConfigMixin
from tsugite_daemon.adapters.http.introspection import IntrospectionMixin
from tsugite_daemon.adapters.http.jobs import JobsMixin
from tsugite_daemon.adapters.http.push import PushMixin
from tsugite_daemon.adapters.http.schedules import SchedulesMixin
from tsugite_daemon.adapters.http.secrets import SecretsMixin
from tsugite_daemon.adapters.http.sessions import SessionsMixin
from tsugite_daemon.adapters.http.sse import (
    SSEBroadcaster,
    sse_stream,
)
from tsugite_daemon.adapters.http.static import StaticMixin
from tsugite_daemon.adapters.http.terminals import TerminalsMixin
from tsugite_daemon.adapters.http.usage import UsageMixin
from tsugite_daemon.adapters.http.webhooks import WebhooksMixin
from tsugite_daemon.adapters.http.workspace_files import WorkspaceFilesMixin
from tsugite_daemon.config import AgentConfig, HTTPConfig
from tsugite_daemon.webhook_store import WebhookStore


class HTTPServer(
    AgentsMixin,
    WorkspaceFilesMixin,
    HooksConfigMixin,
    SessionsMixin,
    SchedulesMixin,
    JobsMixin,
    TerminalsMixin,
    WebhooksMixin,
    FilesMixin,
    PushMixin,
    SecretsMixin,
    UsageMixin,
    ActivityMixin,
    IntrospectionMixin,
    StaticMixin,
):
    """Runs a Starlette ASGI app with uvicorn for the HTTP API."""

    def __init__(
        self,
        config: HTTPConfig,
        adapters: dict[str, HTTPAgentAdapter],
        webhook_store: WebhookStore,
        agent_configs: dict[str, AgentConfig],
        gateway=None,
        token_store=None,
    ):
        self.config = config
        self.adapters = adapters
        self.webhook_store = webhook_store
        self.agent_configs = agent_configs
        self.gateway = gateway
        self._token_store = token_store
        self._server = None
        self.scheduler = None  # Set by Gateway after SchedulerAdapter is created
        self.session_runner = None  # Set by Gateway after SessionRunner is created
        self.jobs_orchestrator = None  # Set by Gateway after JobsOrchestrator is created
        self.job_store = None  # Set by Gateway alongside jobs_orchestrator
        self.terminal_store = None  # Set by Gateway when terminal viewer is wired
        self.pty_manager = None  # Set by Gateway alongside terminal_store
        self.push_store = None  # Set by Gateway if web-push is configured
        self.vapid_public_key = None  # Set by Gateway if web-push is configured
        self.plugin_ui_surfaces: list[dict] = []  # Filled by Gateway from the plugin UI surface registry
        self._active_chats: dict[tuple[str, str, str], ActiveChat] = {}
        self.event_bus = SSEBroadcaster()
        self.app = self._build_app()

    def mount_plugin_routes(self, plugin_name: str, authed_routes: list, public_routes: list) -> None:
        """Mount an adapter plugin's routes under `/api/plugins/<plugin_name>`.

        `authed_routes` each get their endpoint wrapped with the daemon bearer
        token check; `public_routes` are mounted verbatim (the plugin owns their
        access control). Called after plugin load and before uvicorn starts, so
        appending to the live router is race-free. No-op when neither list has
        anything to mount.

        Auth-wrapping needs a flat Route endpoint. Anything else (e.g. a Mount)
        is skipped with a warning - never mounted unwrapped, which would
        silently drop the bearer check - so a plugin mistake can't abort daemon
        startup or open an unauthenticated hole.
        """
        from starlette.routing import Mount

        routes = []
        for r in authed_routes:
            if not isinstance(r, Route):
                logger.warning(
                    "Plugin '%s': cannot auth-wrap non-Route entry %r from get_http_routes(); skipping it",
                    plugin_name,
                    r,
                )
                continue
            routes.append(self._wrap_route_with_auth(r))
        routes.extend(public_routes)
        if not routes:
            return
        self.app.router.routes.append(Mount(f"/api/plugins/{plugin_name}", routes=routes))

    def _wrap_route_with_auth(self, route: Route) -> Route:
        """Return a copy of `route` whose endpoint short-circuits to 401 unless the
        request carries a valid daemon token."""
        endpoint = route.endpoint

        async def guarded(request: Request):
            if err := self._check_auth(request):
                return err
            return await endpoint(request)

        return Route(route.path, guarded, methods=list(route.methods or []), name=route.name)

    def _check_auth(self, request: Request) -> Optional[JSONResponse]:
        token = request.headers.get("authorization", "").removeprefix("Bearer ")

        path = request.url.path

        if token and self._token_store:
            valid, identity = self._token_store.validate(token)
            if valid:
                logger.debug("auth ok (%s) path=%s", identity, path)
                return None

        if token:
            logger.warning("auth failed (invalid token) path=%s", path)
        else:
            logger.warning("auth failed (no token) path=%s", path)
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    def _get_adapter(
        self, request: Request, *, fallback_session_id: Optional[str] = None
    ) -> tuple[Optional[HTTPAgentAdapter], Optional[JSONResponse]]:
        """Authenticate and resolve the agent adapter from the request.

        Returns (adapter, None) on success, or (None, error_response) on failure.

        When the URL agent isn't a live adapter but ``fallback_session_id`` names a
        session, resolve that session's OWNING agent instead. Session-scoped routes
        (chat / cancel / respond) pass the body's session_id so a reply lands on the
        agent that owns the session even when the client pointed the request at a
        non-adapter agent - e.g. the web UI opening a job's session with the worker
        agent-file (``job_worker``) as the agent.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return None, auth_err
        agent_name = request.path_params["agent"]
        adapter = self.adapters.get(agent_name)
        if adapter is None and fallback_session_id:
            adapter = self._adapter_owning_session(fallback_session_id)
        if not adapter:
            return None, JSONResponse({"error": f"unknown agent: {agent_name}"}, status_code=404)
        return adapter, None

    def _adapter_owning_session(self, session_id: str) -> Optional[HTTPAgentAdapter]:
        """The live HTTP adapter that owns ``session_id``, or None. Adapters share
        one SessionStore, so any adapter's store answers; the session's ``agent``
        field is the authoritative owner."""
        store = next((a.session_store for a in self.adapters.values()), None)
        if store is None:
            return None
        try:
            session = store.get_session(session_id)
        except (ValueError, KeyError):
            return None
        return self.adapters.get(session.agent)

    def _build_app(self) -> Starlette:
        routes = [
            *self._core_routes(),
            *self._agent_routes(),
            *self._workspace_routes(),
            *self._hooks_routes(),
            *self._session_routes(),
            *self._schedule_routes(),
            *self._job_routes(),
            *self._terminal_routes(),
            *self._webhook_routes(),
            *self._file_routes(),
            *self._push_routes(),
            *self._secrets_routes(),
            *self._usage_routes(),
            *self._activity_routes(),
            *self._introspection_routes(),
            *self._static_routes(),
        ]
        return Starlette(routes=routes)

    def _core_routes(self) -> list:
        return [
            Route("/api/health", self._health, methods=["GET"]),
            Route("/api/agents", self._list_agents, methods=["GET"]),
            Route("/api/models", self._list_models, methods=["GET"]),
            Route("/api/events", self._events, methods=["GET"]),
            Route("/api/commands", self._list_commands, methods=["GET"]),
            Route("/api/context-providers", self._list_context_providers, methods=["GET"]),
            Route("/api/context-providers/{key}/choices", self._context_provider_choices, methods=["GET"]),
            Route("/api/context-providers/{key}/search", self._context_provider_search, methods=["GET"]),
            Route("/api/context-providers/{key}/capture", self._context_provider_capture, methods=["POST"]),
            Route("/api/daemon/reload-config", self._reload_config, methods=["POST"]),
        ]

    async def _reload_config(self, request: Request) -> JSONResponse:
        """Re-read daemon.yaml and hot-apply what can be applied (the HTTP agent
        set); everything boot-only is reported back as restart_required."""
        if err := self._check_auth(request):
            return err
        if not self.gateway:
            return JSONResponse({"error": "no gateway attached (test server?)"}, status_code=503)
        try:
            result = await self.gateway.reload_config()
        except Exception as e:  # noqa: BLE001 -- config errors go to the caller, not the log alone
            logger.exception("Config reload failed")
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse(result)

    async def _health(self, request: Request) -> JSONResponse:
        try:
            from importlib.metadata import version

            v = version("tsugite-cli")
        except Exception:  # noqa: BLE001 -- fall back to in-tree constant
            from tsugite import __version__ as v
        return JSONResponse(
            {
                "status": "ok",
                "version": v,
                "agents": list(self.adapters.keys()),
                "images": {"max_edge": self.config.image_max_edge, "quality": self.config.image_quality},
            }
        )

    async def _list_commands(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        from tsugite_daemon.commands import get_commands

        return JSONResponse(
            {
                "commands": [
                    {
                        "name": cmd.name,
                        "description": cmd.description,
                        "params": [
                            {
                                "name": p.name,
                                "type": p.type.__name__,
                                "description": p.description,
                                "required": p.required,
                                **({"choices": p.choices} if p.choices else {}),
                                **({"widget": p.widget} if p.widget else {}),
                            }
                            for p in cmd.params
                        ],
                    }
                    for cmd in get_commands().values()
                ]
            }
        )

    def _context_provider_ctx(self, session_id: Optional[str]) -> dict:
        """The ``ctx`` a context provider receives, resolved from the request's
        session the way the chat/command endpoints resolve a session's owner and
        workspace: the session's authoritative ``agent`` field, that agent's
        workspace (overridden by the session's own ``workspace_override``, as the
        send path does), and the session's ``user_id``. Fields stay None when the
        session is unknown so a ctx-agnostic provider still runs."""
        ctx = {"session_id": session_id, "user_id": None, "agent": None, "workspace_dir": None}
        if not session_id:
            return ctx
        store = next((a.session_store for a in self.adapters.values()), None)
        if store is None:
            return ctx
        try:
            session = store.get_session(session_id)
        except (ValueError, KeyError):
            return ctx
        adapter = self.adapters.get(session.agent)
        ctx["user_id"] = session.user_id
        ctx["agent"] = session.agent
        if session.workspace_override:
            ctx["workspace_dir"] = Path(session.workspace_override)
        elif adapter is not None:
            ctx["workspace_dir"] = adapter.agent_config.workspace_dir
        return ctx

    async def _list_context_providers(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        from tsugite.context import get_context_providers

        # An autocomplete-only source (a prefix + search, menu=False) is listed too
        # so the frontend learns its prefix, but rides with in_menu=False so the
        # add-context menu still excludes it.
        return JSONResponse(
            {
                "providers": [
                    {
                        "key": p.key,
                        "label": p.label,
                        "icon": p.icon,
                        "has_choices": p.choices is not None,
                        "picker": p.picker,
                        "in_menu": p.in_menu,
                        "autocomplete_prefix": p.autocomplete_prefix,
                    }
                    for p in get_context_providers()
                    if p.in_menu or p.is_autocomplete_source
                ]
            }
        )

    async def _context_provider_choices(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        from tsugite.context import get_choices

        key = request.path_params["key"]
        ctx = self._context_provider_ctx(request.query_params.get("session_id"))
        choices = get_choices(key, ctx)
        return JSONResponse({"choices": [{"value": c.value, "label": c.label} for c in choices]})

    async def _context_provider_search(self, request: Request) -> JSONResponse:
        """Query-aware autocomplete for an ``@<prefix> <query>`` source. Called per
        keystroke, so it always returns 200 with a (possibly empty) result list;
        the frontend renders empty rather than surfacing an error."""
        if err := self._check_auth(request):
            return err
        from tsugite.context import run_search

        key = request.path_params["key"]
        query = request.query_params.get("q", "")
        ctx = self._context_provider_ctx(request.query_params.get("session_id"))
        results = run_search(key, query, ctx)
        return JSONResponse({"results": [{"value": c.value, "label": c.label} for c in results]})

    async def _context_provider_capture(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        from tsugite.context import run_capture

        key = request.path_params["key"]
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}
        ctx = self._context_provider_ctx(body.get("session_id"))
        try:
            items = run_capture(key, body.get("arg"), ctx)
        except Exception as e:  # noqa: BLE001 -- a deliberately-picked provider's error goes to the user
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse({"items": [it.to_metadata() for it in items]})

    async def _list_agents(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        running_by_agent: dict[str, set[str]] = {}
        for agent_name, _user_id, session_id in self._active_chats:
            running_by_agent.setdefault(agent_name, set()).add(session_id)
        agents = [
            {
                "name": name,
                "agent_file": adapter.agent_config.agent_file,
                "workspace_dir": str(adapter.agent_config.workspace_dir),
                "running_tasks": len(running_by_agent.get(name, [])),
            }
            for name, adapter in self.adapters.items()
        ]
        return JSONResponse({"agents": agents})

    async def _list_models(self, request: Request) -> JSONResponse:
        from tsugite.providers import get_provider, list_all_providers
        from tsugite.providers.model_registry import list_models

        if not getattr(self, "_models_primed", False):
            for name in list_all_providers():
                try:
                    get_provider(name)
                except Exception:  # noqa: BLE001 -- provider init may fail without env keys; skip those
                    continue
            self._models_primed = True

        models: list[dict] = []
        for key, info in list_models().items():
            provider, _, model_id = key.partition("/")
            full_id = f"{provider}:{model_id}" if provider and model_id else key
            models.append(
                {
                    "id": full_id,
                    "provider": provider or None,
                    "context_window": info.max_input_tokens,
                    "max_output_tokens": info.max_output_tokens,
                    "input_cost_per_million": info.input_cost_per_million,
                    "output_cost_per_million": info.output_cost_per_million,
                    "supports_vision": info.supports_vision,
                    "supports_reasoning": info.supports_reasoning,
                }
            )
        models.sort(key=lambda m: m["id"])
        return JSONResponse({"models": models})

    def _require_auth_and_scheduler(self, request: Request) -> Optional[JSONResponse]:
        """Check auth and scheduler availability. Returns error response or None."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        if not self.scheduler:
            return JSONResponse({"error": "scheduler not available"}, status_code=503)
        return None

    def _require_auth_and_sessions(self, request: Request) -> Optional[JSONResponse]:
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        if not self.session_runner:
            return JSONResponse({"error": "session runner not available"}, status_code=503)
        return None

    def _require_auth_and_jobs(self, request: Request) -> Optional[JSONResponse]:
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        if not self.jobs_orchestrator:
            return JSONResponse({"error": "jobs orchestrator not available"}, status_code=503)
        return None

    def _require_auth_and_terminals(self, request: Request) -> Optional[JSONResponse]:
        """Check auth and terminal subsystem availability."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        if self.terminal_store is None or self.pty_manager is None:
            return JSONResponse({"error": "terminal viewer not available"}, status_code=503)
        return None

    async def _events(self, request: Request) -> Response:
        if err := self._check_auth(request):
            return err

        # Reconnect reconciliation: a client that was connected before sends
        # its last-seen seq + the server epoch it saw. Same epoch and a
        # replayable gap -> replay the missed events; daemon restarted (epoch
        # change) or gap older than the buffer -> tell it to fully resync.
        client_epoch = request.query_params.get("epoch")
        try:
            last_seq = int(request.query_params.get("last_seq", "0"))
        except ValueError:
            last_seq = 0

        # Subscribe BEFORE computing the replay so no event falls in a crack:
        # anything emitted from here on lands in the live queue, the replay
        # covers everything before, and the seq-dedup below drops the overlap.
        queue = self.event_bus.subscribe()
        replay: list[dict] = []
        resync = False
        if client_epoch is not None:
            if client_epoch != self.event_bus.epoch:
                resync = True
            else:
                replayable = self.event_bus.replay_since(last_seq)
                if replayable is None:
                    resync = True
                else:
                    replay = replayable
        hello = {"type": "hello", "data": {"epoch": self.event_bus.epoch, "seq": self.event_bus.seq, "resync": resync}}

        async def generator():
            try:
                yield f"data: {json.dumps(hello)}\n\n"
                seen = last_seq
                for msg in replay:
                    yield f"data: {json.dumps(msg)}\n\n"
                    seen = msg["seq"]
                async for chunk in sse_stream(queue):
                    # A replayed event can also sit in the live queue (emitted
                    # between the replay snapshot and subscribe); drop dups.
                    if chunk.startswith("data: ") and '"seq"' in chunk:
                        try:
                            if json.loads(chunk[6:]).get("seq", 0) <= seen:
                                continue
                        except json.JSONDecodeError:
                            pass
                    yield chunk
            finally:
                self.event_bus.unsubscribe(queue)

        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    async def start(self):
        import uvicorn

        config = uvicorn.Config(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info",
            log_config=None,
        )
        self._server = uvicorn.Server(config)
        self._server.install_signal_handlers = lambda: None
        logger.info("HTTP API listening on http://%s:%d", self.config.host, self.config.port)
        await self._server.serve()

    async def stop(self):
        if self._server:
            # Signal all SSE subscribers to disconnect
            for q in list(self.event_bus._subscribers):
                try:
                    q.put_nowait(None)
                except asyncio.QueueFull:
                    pass
            self._server.should_exit = True
