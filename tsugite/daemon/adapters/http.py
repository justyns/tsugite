"""HTTP API adapter with SSE streaming and webhook inbox."""

import asyncio
import json
import logging
import mimetypes
import re
import shutil
import threading
from dataclasses import asdict
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from tsugite.agent_inheritance import get_builtin_agents_path, get_global_agents_paths
from tsugite.attachments.base import AttachmentContentType
from tsugite.attachments.file import FileHandler
from tsugite.core.content_blocks import extract_content_blocks
from tsugite.daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite.daemon.config import AgentConfig, HTTPConfig
from tsugite.daemon.scheduler import ScheduleEntry
from tsugite.daemon.webhook_store import WebhookStore
from tsugite.events.base import BaseEvent
from tsugite.history.models import CompactionSummary, HookExecution, Turn
from tsugite.history.storage import SessionStorage, get_history_dir
from tsugite.skill_discovery import get_builtin_skills_path
from tsugite.ui.jsonl import JSONLUIHandler
from tsugite.utils import parse_yaml_frontmatter

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

MAX_TEXT_ATTACH_SIZE = 50 * 1024  # 50KB — ~12K tokens
MAX_BINARY_ATTACH_SIZE = 10 * 1024 * 1024  # 10MB
MAX_UPLOAD_TOTAL = 100 * 1024 * 1024  # 100MB per request
MAX_UPLOAD_FILES = 20
MAX_WORKSPACE_LIST_FILES = 5000

# MIME types treated as text for workspace browsing (beyond text/*)
_TEXT_MIMES = {
    "application/json",
    "application/xml",
    "application/javascript",
    "application/x-sh",
    "application/x-shellscript",
    "application/toml",
    "application/yaml",
    "application/x-yaml",
    "application/sql",
    "application/graphql",
    "application/xhtml+xml",
    "image/svg+xml",
}
# Extensions that mimetypes doesn't know about or misidentifies
_TEXT_EXTENSIONS_EXTRA = {
    ".yaml",
    ".yml",
    ".toml",
    ".jsonl",
    ".ndjson",
    ".tsx",
    ".jsx",
    ".mjs",
    ".cjs",
    ".vue",
    ".svelte",
    ".go",
    ".rs",
    ".kt",
    ".scala",
    ".swift",
    ".ex",
    ".exs",
    ".erl",
    ".jl",
    ".r",
    ".env",
    ".cfg",
    ".ini",
    ".conf",
    ".editorconfig",
    ".gitignore",
    ".gitattributes",
    ".dockerignore",
    ".graphql",
    ".gql",
    ".proto",
    ".lock",
    ".sum",
    ".mod",
}


def _is_text_mime(path: Path) -> bool:
    """Check if a file is likely text based on its MIME type or extension."""
    if path.suffix.lower() in _TEXT_EXTENSIONS_EXTRA:
        return True
    mime, _ = mimetypes.guess_type(path.name)
    if mime is None:
        return False
    return mime.startswith("text/") or mime in _TEXT_MIMES


logger = logging.getLogger(__name__)

_file_handler = FileHandler()


def _sanitize_filename(name: str) -> str:
    """Strip path separators and dangerous characters from a filename."""
    name = Path(name).name  # strip directory components
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = name.lstrip(".")
    return name[:200] or "upload"


def _deduplicate_dest(uploads_dir: Path, name: str, max_copies: int = 1000) -> tuple[Path, Optional[str]]:
    """Find a non-colliding destination path in uploads_dir. Returns (path, error_or_None)."""
    dest = uploads_dir / name
    if not dest.exists():
        return dest, None
    stem, suffix = dest.stem, dest.suffix
    for counter in range(1, max_copies + 1):
        dest = uploads_dir / f"{stem}_{counter}{suffix}"
        if not dest.exists():
            return dest, None
    return dest, "too many copies of this file"


def _should_context_attach(path: Path, size: int) -> bool:
    """Determine if a file should be attached as LLM context."""
    _, content_type = _file_handler._detect_content_type(path)
    if content_type == AttachmentContentType.TEXT:
        return size <= MAX_TEXT_ATTACH_SIZE
    if path.suffix.lower() in FileHandler.BINARY_EXTENSIONS:
        return size <= MAX_BINARY_ATTACH_SIZE
    return False


class SSEBroadcaster:
    """Pub/sub for pushing real-time events to SSE subscribers."""

    def __init__(self):
        self._subscribers: list[asyncio.Queue] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread_id: Optional[int] = None

    def subscribe(self) -> asyncio.Queue:
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
            self._loop_thread_id = threading.current_thread().ident
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    def emit(self, event_type: str, data: dict | None = None):
        if not self._subscribers:
            return
        msg = {"type": event_type, "data": data or {}}
        on_loop = threading.current_thread().ident == self._loop_thread_id
        for q in list(self._subscribers):
            try:
                if on_loop:
                    q.put_nowait(msg)
                elif self._loop:
                    self._loop.call_soon_threadsafe(q.put_nowait, msg)
            except (asyncio.QueueFull, RuntimeError):
                pass


async def sse_stream(queue: asyncio.Queue, keepalive_interval: float = 15.0):
    """Shared async generator for SSE streams with keepalive."""
    while True:
        try:
            data = await asyncio.wait_for(queue.get(), timeout=keepalive_interval)
        except asyncio.TimeoutError:
            yield ": keepalive\n\n"
            continue
        if data is None:
            break
        yield f"data: {json.dumps(data)}\n\n"


class SSEProgressHandler(JSONLUIHandler):
    """Converts agent events to SSE messages via an async queue."""

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.done = False
        self.has_final = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._persist_event: Optional[Callable] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def set_event_persister(self, fn: Callable):
        """Set a callback to persist select events to the session event log."""
        self._persist_event = fn

    latest_prompt_messages: Optional[list] = None

    def handle_event(self, event: BaseEvent) -> None:
        """Handle event from agent thread — schedule onto the event loop."""
        from tsugite.events import PromptSnapshotEvent

        if isinstance(event, PromptSnapshotEvent):
            if event.messages:
                self.latest_prompt_messages = event.messages
            if not event.token_breakdown:
                return  # Messages-only update, don't emit SSE or persist
        super().handle_event(event)

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        if event_type == "final_result":
            self.has_final = True
        payload = {"type": event_type, **data}
        if self._loop and self._loop.is_running():
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None
            if running_loop is self._loop:
                self.queue.put_nowait(payload)
            else:
                self._loop.call_soon_threadsafe(self.queue.put_nowait, payload)
        else:
            self.queue.put_nowait(payload)

        if event_type in ("prompt_snapshot", "reaction") and self._persist_event:
            self._persist_event(payload)

    def signal_done(self):
        """Set done and wake up the generator."""
        self.done = True
        self.queue.put_nowait(None)

    async def event_generator(self):
        while True:
            try:
                data = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if self.done:
                    break
                yield ": keepalive\n\n"
                continue
            if data is None:
                break
            yield f"data: {json.dumps(data)}\n\n"
        # Drain remaining
        while not self.queue.empty():
            data = self.queue.get_nowait()
            if data is not None:
                yield f"data: {json.dumps(data)}\n\n"
        yield 'data: {"type": "done"}\n\n'


class HTTPInteractionBackend:
    """Interaction backend for HTTP — emits SSE events, blocks until response."""

    TIMEOUT = 300  # 5 minutes

    def __init__(self, progress: SSEProgressHandler):
        self._progress = progress
        self._event = threading.Event()
        self._response: Optional[str] = None
        self.pending_message: Optional[str] = None

    def ask_user(self, question: str, question_type: str = "text", options: Optional[list[str]] = None) -> str:
        self._event.clear()
        self._response = None
        payload = {"question": question, "question_type": question_type}
        if options:
            payload["options"] = options
        self._progress._emit("ask_user", payload)

        if not self._event.wait(timeout=self.TIMEOUT):
            raise RuntimeError("Timed out waiting for user response (HTTP)")
        return self._response or ""

    def submit_response(self, response: str) -> None:
        self._response = response
        self._event.set()


class HTTPAgentAdapter(BaseAdapter):
    """Per-agent adapter for HTTP. Lifecycle managed by HTTPServer."""

    def resolve_http_user(self, user_id: str) -> str:
        """Resolve an HTTP user_id to canonical identity (no channel context needed for HTTP DMs)."""
        ctx = ChannelContext(source="http", channel_id=None, user_id=user_id, reply_to=f"http:{user_id}")
        return self.resolve_user(user_id, ctx)

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class HTTPServer:
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
        self.push_store = None  # Set by Gateway if web-push is configured
        self.vapid_public_key = None  # Set by Gateway if web-push is configured
        self._active_backends: dict[tuple[str, str], HTTPInteractionBackend] = {}
        self._active_chat_tasks: dict[tuple[str, str], asyncio.Task] = {}
        self._active_progress: dict[tuple[str, str], SSEProgressHandler] = {}
        self.event_bus = SSEBroadcaster()
        self.app = self._build_app()

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

    def _get_adapter(self, request: Request) -> tuple[Optional[HTTPAgentAdapter], Optional[JSONResponse]]:
        """Authenticate and resolve the agent adapter from the request.

        Returns (adapter, None) on success, or (None, error_response) on failure.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return None, auth_err
        agent_name = request.path_params["agent"]
        adapter = self.adapters.get(agent_name)
        if not adapter:
            return None, JSONResponse({"error": f"unknown agent: {agent_name}"}, status_code=404)
        return adapter, None

    def _build_app(self) -> Starlette:
        routes = [
            Route("/api/health", self._health, methods=["GET"]),
            Route("/api/agents", self._list_agents, methods=["GET"]),
            Route("/api/agents/{agent}/sessions", self._list_sessions, methods=["GET"]),
            Route("/api/agents/{agent}/sessions/new", self._new_interactive_session, methods=["POST"]),
            Route("/api/agents/{agent}/chat", self._chat, methods=["POST"]),
            Route("/api/agents/{agent}/chat/cancel", self._cancel_chat, methods=["POST"]),
            Route("/api/agents/{agent}/upload", self._upload, methods=["POST"]),
            Route("/api/agents/{agent}/status", self._status, methods=["GET"]),
            Route("/api/agents/{agent}/attachments", self._attachments, methods=["GET"]),
            Route("/api/agents/{agent}/history", self._history, methods=["GET"]),
            Route("/api/agents/{agent}/prompt-snapshot", self._prompt_snapshot, methods=["GET"]),
            Route("/api/agents/{agent}/config", self._update_agent_config, methods=["PATCH"]),
            Route("/api/agents/{agent}/compact", self._compact, methods=["POST"]),
            Route("/api/agents/{agent}/respond", self._respond, methods=["POST"]),
            Route("/api/schedules", self._list_schedules, methods=["GET"]),
            Route("/api/schedules", self._create_schedule, methods=["POST"]),
            Route("/api/schedules/cleanup", self._cleanup_schedules, methods=["POST"]),
            Route("/api/schedules/{schedule_id}", self._get_schedule, methods=["GET"]),
            Route("/api/schedules/{schedule_id}", self._update_schedule, methods=["PATCH"]),
            Route("/api/schedules/{schedule_id}", self._delete_schedule, methods=["DELETE"]),
            Route("/api/schedules/{schedule_id}/enable", self._enable_schedule, methods=["POST"]),
            Route("/api/schedules/{schedule_id}/disable", self._disable_schedule, methods=["POST"]),
            Route("/api/schedules/{schedule_id}/run", self._run_schedule, methods=["POST"]),
            Route("/api/schedules/{schedule_id}/sessions", self._schedule_sessions, methods=["GET"]),
            Route("/api/sessions", self._api_list_sessions, methods=["GET"]),
            Route("/api/sessions", self._api_start_session, methods=["POST"]),
            Route("/api/sessions/{session_id}/metadata", self._api_get_metadata, methods=["GET"]),
            Route("/api/sessions/{session_id}/metadata", self._api_update_metadata, methods=["PATCH"]),
            Route("/api/sessions/{session_id}/metadata/{key}", self._api_delete_metadata, methods=["DELETE"]),
            Route("/api/sessions/{session_id}/scratchpad", self._api_get_scratchpad, methods=["GET"]),
            Route("/api/sessions/{session_id}/scratchpad", self._api_update_scratchpad, methods=["PUT"]),
            Route("/api/sessions/{session_id}", self._api_get_session, methods=["GET"]),
            Route("/api/sessions/{session_id}", self._api_update_session, methods=["PATCH"]),
            Route("/api/sessions/{session_id}/cancel", self._api_cancel_session, methods=["POST"]),
            Route("/api/sessions/{session_id}/restart", self._api_restart_session, methods=["POST"]),
            Route("/api/sessions/{session_id}/events", self._api_session_events, methods=["GET"]),
            Route("/api/webhooks", self._list_webhooks, methods=["GET"]),
            Route("/api/webhooks", self._create_webhook, methods=["POST"]),
            Route("/api/webhooks/{token}", self._delete_webhook, methods=["DELETE"]),
            Route("/webhook/{token}", self._webhook, methods=["POST"]),
            Route("/api/agent-files", self._list_agent_files, methods=["GET"]),
            Route("/api/agent-files/content", self._read_agent_file, methods=["GET"]),
            Route("/api/agent-files/content", self._save_agent_file, methods=["PUT"]),
            Route("/api/skill-files", self._list_skill_files, methods=["GET"]),
            Route("/api/skill-files/content", self._read_skill_file, methods=["GET"]),
            Route("/api/skill-files/content", self._save_skill_file, methods=["PUT"]),
            Route("/api/agents/{agent}/workspace", self._list_workspace_files, methods=["GET"]),
            Route("/api/agents/{agent}/workspace/content", self._read_workspace_file, methods=["GET"]),
            Route("/api/agents/{agent}/workspace/content", self._save_workspace_file, methods=["PUT"]),
            Route("/api/agents/{agent}/workspace/attach", self._attach_workspace_file, methods=["POST"]),
            Route("/api/events", self._events, methods=["GET"]),
            Route("/api/push/vapid-key", self._push_vapid_key, methods=["GET"]),
            Route("/api/push/subscribe", self._push_subscribe, methods=["POST"]),
            Route("/api/push/unsubscribe", self._push_unsubscribe, methods=["POST"]),
            Route("/api/secrets", self._secrets_list, methods=["GET"]),
            Route("/api/secrets/{name:path}", self._secrets_set, methods=["POST"]),
            Route("/api/secrets/{name:path}", self._secrets_delete, methods=["DELETE"]),
            Route("/api/kv/namespaces", self._kv_namespaces, methods=["GET"]),
            Route("/api/kv/{namespace}/keys", self._kv_keys, methods=["GET"]),
            Route("/api/kv/{namespace}/keys/{key:path}", self._kv_get, methods=["GET"]),
            Route("/api/commands", self._list_commands, methods=["GET"]),
            Route("/api/agents/{agent}/commands/{command_name}", self._run_command, methods=["POST"]),
            Route("/api/usage/summary", self._usage_summary, methods=["GET"]),
            Route("/api/usage/agents", self._usage_agents, methods=["GET"]),
            Route("/api/usage/models", self._usage_models, methods=["GET"]),
            Route("/api/usage/total", self._usage_total, methods=["GET"]),
            Mount("/static", app=StaticFiles(directory=str(WEB_DIR)), name="static"),
            Route("/sw.js", self._serve_sw, methods=["GET"]),
            Route("/", self._serve_ui, methods=["GET"]),
        ]
        return Starlette(routes=routes)

    async def _health(self, request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "agents": list(self.adapters.keys())})

    async def _list_commands(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        from tsugite.daemon.commands import get_commands

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
                            }
                            for p in cmd.params
                        ],
                    }
                    for cmd in get_commands().values()
                ]
            }
        )

    async def _run_command(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err
        from tsugite.daemon.commands import get_commands

        command_name = request.path_params["command_name"]
        commands = get_commands()
        if command_name not in commands:
            return JSONResponse({"error": f"Unknown command: {command_name}"}, status_code=404)

        cmd = commands[command_name]
        try:
            body = await request.json()
        except Exception:
            body = {}

        allowed_keys = {p.name for p in cmd.params}
        filtered = {k: v for k, v in body.items() if k in allowed_keys}

        missing = [p.name for p in cmd.params if p.required and p.name not in filtered]
        if missing:
            return JSONResponse({"error": f"Missing required params: {', '.join(missing)}"}, status_code=400)

        try:
            result = await cmd.handler(adapter, **filtered)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        return JSONResponse({"result": result})

    def _get_usage_store(self):
        from tsugite.usage import get_usage_store

        return get_usage_store()

    def _parse_limit(self, request: Request, default: int = 10, cap: int = 100) -> int:
        try:
            return max(1, min(int(request.query_params.get("limit", str(default))), cap))
        except ValueError:
            return default

    async def _usage_summary(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        store = self._get_usage_store()
        period = request.query_params.get("period", "day")
        since = request.query_params.get("since")
        agent = request.query_params.get("agent")
        return JSONResponse(store.summary(agent=agent, period=period, since=since))

    async def _usage_agents(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        store = self._get_usage_store()
        since = request.query_params.get("since")
        limit = self._parse_limit(request)
        return JSONResponse(store.top_agents(since=since, limit=limit))

    async def _usage_models(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        store = self._get_usage_store()
        since = request.query_params.get("since")
        limit = self._parse_limit(request)
        return JSONResponse(store.top_models(since=since, limit=limit))

    async def _usage_total(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        store = self._get_usage_store()
        since = request.query_params.get("since")
        return JSONResponse(store.total(since=since))

    async def _list_agents(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        running_by_agent: dict[str, list[str]] = {}
        for agent_name, user_id in self._active_backends:
            running_by_agent.setdefault(agent_name, []).append(user_id)
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

    async def _list_sessions(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        source = request.query_params.get("source")
        status = request.query_params.get("status")
        parent_id = request.query_params.get("parent_id")
        try:
            limit = max(1, min(int(request.query_params.get("limit", "100")), 1000))
        except (ValueError, TypeError):
            limit = 100

        all_sessions = adapter.session_store.list_sessions(
            agent=adapter.agent_name, source=source, status=status, parent_id=parent_id, limit=limit
        )

        default_ids = adapter.session_store.default_interactive_ids(adapter.agent_name)
        sessions = []
        for s in all_sessions:
            user_id = s.user_id or ""
            if user_id.isdigit():
                label = f"Discord: {user_id}"
            elif user_id.startswith("web-"):
                label = f"Web: {user_id}"
            else:
                label = user_id or s.source
            sessions.append(
                {
                    "id": s.id,
                    "user_id": user_id,
                    "label": label,
                    "conversation_id": s.id,
                    "source": s.source,
                    "status": s.status,
                    "state": s.status,
                    "created_at": s.created_at,
                    "last_active": s.last_active,
                    "parent_id": s.parent_id,
                    "prompt": s.prompt or "",
                    "model": s.model,
                    "error": s.error,
                    "result": s.result,
                    "title": s.title,
                    "is_default": default_ids.get(user_id) == s.id,
                    "metadata": s.metadata or {},
                }
            )

        return JSONResponse({"sessions": sessions})

    async def _new_interactive_session(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        user_id = adapter.resolve_http_user(body.get("user_id", "web-anonymous"))

        from tsugite.daemon.session_store import Session, SessionSource
        from tsugite.history.storage import generate_session_id

        session_id = generate_session_id(adapter.agent_name)
        session = Session(
            id=session_id,
            agent=adapter.agent_name,
            source=SessionSource.INTERACTIVE.value,
            user_id=user_id,
        )
        adapter.session_store.create_session(session)
        if self.event_bus:
            self.event_bus.emit("session_update", {"action": "created", "id": session_id})
        return JSONResponse({"id": session_id}, status_code=201)

    async def _status(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        user_id = adapter.resolve_http_user(request.query_params.get("user_id", "web-anonymous"))
        session_id = request.query_params.get("session_id")
        if session_id:
            try:
                session = adapter.session_store.get_session(session_id)
            except ValueError:
                session = adapter.session_store.get_or_create_interactive(user_id, adapter.agent_name)
        else:
            session = adapter.session_store.get_or_create_interactive(user_id, adapter.agent_name)

        backend_key = (adapter.agent_name, user_id)
        backend = self._active_backends.get(backend_key)

        return JSONResponse(
            {
                "model": adapter.resolve_model(),
                "tokens": session.cumulative_tokens,
                "context_limit": adapter.session_store.get_context_limit(adapter.agent_name),
                "threshold": adapter.session_store.get_compaction_threshold(adapter.agent_name),
                "message_count": session.message_count,
                "compacting": adapter.session_store.is_compacting(user_id, adapter.agent_name),
                "metadata": session.metadata or {},
                "busy": backend is not None,
                "pending_message": backend.pending_message if backend else None,
                "attachments": [
                    {"name": a.name, "content_type": a.content_type.value, "mime_type": a.mime_type}
                    for a in adapter._get_all_attachments()
                ],
            }
        )

    async def _update_agent_config(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        body = await request.json()
        agent_name = request.path_params["agent"]
        agent_config = adapter.agent_config

        if "model" in body:
            new_model = body["model"].strip() if body["model"] else None
            agent_config.model = new_model

            from tsugite.daemon.memory import DEFAULT_CONTEXT_LIMIT, get_context_limit

            if new_model:
                context_limit = get_context_limit(new_model, fallback=DEFAULT_CONTEXT_LIMIT)
                agent_config.context_limit = context_limit
            else:
                context_limit = DEFAULT_CONTEXT_LIMIT
                agent_config.context_limit = None
            adapter.session_store.update_context_limit(adapter.agent_name, context_limit)

            if self.gateway:
                from tsugite.daemon.config import save_daemon_config

                save_daemon_config(self.gateway.config, self.gateway.config_path)

        self.event_bus.emit("agent_status", {"agent": agent_name})
        return JSONResponse(
            {"status": "ok", "model": adapter.resolve_model(), "context_limit": agent_config.context_limit}
        )

    async def _attachments(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err
        attachments = []
        for a in adapter._get_all_attachments():
            entry = {"name": a.name, "content_type": a.content_type.value, "mime_type": a.mime_type}
            if a.content_type.value == "text":
                entry["content"] = a.content
            else:
                entry["size_bytes"] = len(a.content) if a.content else 0
            attachments.append(entry)
        return JSONResponse({"attachments": attachments})

    def _collect_turns(self, session_id: str, limit: int = 0) -> tuple[list, str | None, str | None, str | None]:
        """Collect turns from a session and its compaction chain.

        Args:
            session_id: The current (newest) session ID.
            limit: Max number of turns to return (0 = unlimited). Walks
                   newest-to-oldest and stops early once enough turns are
                   collected.

        Returns:
            (turns, compaction_summary, compacted_from, compaction_reason)
            where the latter three come from the current session's metadata/records.
        """
        history_dir = get_history_dir()
        visited: set[str] = set()

        # Walk the compaction chain newest-to-oldest, loading lazily.
        # Each entry: (storage, records_or_None, compacted_from)
        chain: list[tuple[SessionStorage, list | None, str | None]] = []
        current_id = session_id
        turns_loaded = 0
        while current_id and current_id not in visited:
            visited.add(current_id)
            path = history_dir / f"{current_id}.jsonl"
            if not path.exists():
                break
            try:
                storage = SessionStorage.load(path)
                compacted_from_id = storage._meta.compacted_from if storage._meta else None
                if limit > 0 and turns_loaded >= limit and len(chain) > 0:
                    # We have enough turns from newer links; only store metadata
                    chain.append((storage, None, compacted_from_id))
                    break
                records = storage.load_records()
                turn_count = sum(1 for r in records if isinstance(r, Turn))
                turns_loaded += turn_count
                chain.append((storage, records, compacted_from_id))
                current_id = compacted_from_id
            except Exception:
                break

        # Extract compaction summary/compacted_from/reason from the newest session
        compaction_summary = None
        compacted_from = None
        compaction_reason = None
        if chain:
            newest_storage, newest_records, _ = chain[0]
            if newest_storage._meta:
                compacted_from = newest_storage._meta.compacted_from
            if newest_records:
                cs = next((r for r in newest_records if isinstance(r, CompactionSummary)), None)
                if cs:
                    compaction_summary = cs.summary
                    compaction_reason = cs.compaction_reason

        # Iterate oldest-first (reversed chain) and append for chronological order
        collected: list = []
        for idx in reversed(range(len(chain))):
            _, records, _ = chain[idx]
            if records is None:
                continue
            turns_and_hooks = [r for r in records if isinstance(r, (Turn, HookExecution))]

            # Trim retained turns that overlap with the next (newer) session
            if idx > 0:
                _, next_records, _ = chain[idx - 1]
                comp_summary = None
                if next_records:
                    comp_summary = next((r for r in next_records if isinstance(r, CompactionSummary)), None)
                if comp_summary and comp_summary.retained_turns:
                    turn_count_total = sum(1 for r in turns_and_hooks if isinstance(r, Turn))
                    keep_from = turn_count_total - comp_summary.retained_turns
                    trimmed: list = []
                    seen_turns = 0
                    for r in turns_and_hooks:
                        if isinstance(r, Turn):
                            if seen_turns < keep_from:
                                seen_turns += 1
                                continue
                            seen_turns += 1
                        elif isinstance(r, HookExecution):
                            if seen_turns < keep_from:
                                continue
                        trimmed.append(r)
                    turns_and_hooks = trimmed
            else:
                comp_summary = None

            collected.extend(turns_and_hooks)

            if comp_summary:
                marker = {"marker": "compaction", "summary": comp_summary.summary}
                if comp_summary.compaction_reason:
                    marker["reason"] = comp_summary.compaction_reason
                collected.append(marker)

        if limit > 0:
            turn_count = 0
            for i in range(len(collected) - 1, -1, -1):
                if isinstance(collected[i], Turn):
                    turn_count += 1
                    if turn_count > limit:
                        collected = collected[i + 1 :]
                        break

        return collected, compaction_summary, compacted_from, compaction_reason

    async def _history(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        user_id = adapter.resolve_http_user(request.query_params.get("user_id", "web-anonymous"))
        detail = request.query_params.get("detail", "false").lower() == "true"
        try:
            limit = max(1, min(int(request.query_params.get("limit", "100")), 1000))
        except (ValueError, TypeError):
            limit = 100
        session_id = request.query_params.get("session_id")
        if session_id:
            conversation_id = session_id
        else:
            session = adapter.session_store.get_or_create_interactive(user_id, adapter.agent_name)
            conversation_id = session.id

        turns, compaction_summary, compacted_from, compaction_reason = self._collect_turns(conversation_id, limit=limit)

        # Read events from session event log for reactions and prompt snapshots
        all_events = adapter.session_store.read_events(conversation_id)
        reaction_events = [e for e in all_events if e.get("type") == "reaction" and e.get("emoji")]
        snapshot_events = [e for e in all_events if e.get("type") == "prompt_snapshot"] if detail else []
        # Build list of (turn_timestamp, turn_index) for matching
        turn_timestamps = []
        turn_index = 0
        for item in turns:
            if isinstance(item, dict) or isinstance(item, HookExecution):
                continue
            ts = item.timestamp.isoformat() if item.timestamp else ""
            turn_timestamps.append((ts, turn_index))
            turn_index += 1

        # Assign events to the most recent turn before them
        def _assign_to_turns(events):
            by_turn: dict[int, list] = {}
            for ev in events:
                ev_ts = ev.get("timestamp", "")
                assigned = -1
                for ts, idx in turn_timestamps:
                    if ts and ts <= ev_ts:
                        assigned = idx
                if assigned >= 0:
                    by_turn.setdefault(assigned, []).append(ev)
            return by_turn

        reactions_by_turn = {k: [e["emoji"] for e in v] for k, v in _assign_to_turns(reaction_events).items()}

        # Snapshots are emitted BEFORE the LLM call, so their timestamp is earlier
        # than the turn's. Assign each snapshot to the next turn after it.
        snapshots_by_turn: dict[int, dict] = {}
        for ev in snapshot_events:
            ev_ts = ev.get("timestamp", "")
            for ts, idx in turn_timestamps:
                if ts and ts >= ev_ts:
                    snapshots_by_turn[idx] = ev
                    break

        result_turns = []
        real_turn_index = 0
        for item in turns:
            if isinstance(item, dict) and item.get("marker") == "compaction":
                entry = {"type": "compaction"}
                if item.get("summary"):
                    entry["summary"] = item["summary"]
                if item.get("reason"):
                    entry["reason"] = item["reason"]
                result_turns.append(entry)
                continue
            if isinstance(item, HookExecution):
                result_turns.append(item.model_dump(mode="json", exclude_none=True))
                continue
            user_msg = ""
            for msg in item.messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    user_msg = content if isinstance(content, str) else str(content)
                    break
            content_blocks = {}
            for msg in item.messages:
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str) and "<content" in content:
                        _, blocks = extract_content_blocks(content)
                        content_blocks.update(blocks)

            turn_data = {
                "user": user_msg,
                "assistant": item.final_answer or "",
                "timestamp": item.timestamp.isoformat() if item.timestamp else None,
                "tools_used": item.functions_called or [],
            }
            if content_blocks:
                turn_data["content_blocks"] = content_blocks
            if detail:
                turn_data["messages"] = item.messages
            turn_reactions = reactions_by_turn.get(real_turn_index)
            if turn_reactions:
                turn_data["reactions"] = turn_reactions
            snapshot = snapshots_by_turn.get(real_turn_index)
            if snapshot:
                turn_data["prompt_snapshot"] = {
                    "token_breakdown": snapshot.get("token_breakdown", {}),
                }
            real_turn_index += 1
            result_turns.append(turn_data)

        return JSONResponse(
            {
                "conversation_id": conversation_id,
                "turns": result_turns,
                "compaction_summary": compaction_summary,
                "compacted_from": compacted_from,
                "compaction_reason": compaction_reason,
            }
        )

    async def _prompt_snapshot(self, request: Request) -> JSONResponse:
        """Return the latest prompt snapshot for the current session.

        Includes full messages from the live progress handler if available,
        otherwise just the token breakdown from the persisted event log.
        """
        adapter, err = self._get_adapter(request)
        if err:
            return err

        user_id = adapter.resolve_http_user(request.query_params.get("user_id", "web-anonymous"))
        agent_name = request.path_params["agent"]

        # Check live progress handler for full messages
        backend_key = (agent_name, user_id)
        live_progress = self._active_progress.get(backend_key)
        if live_progress and live_progress.latest_prompt_messages:
            # Reconstruct breakdown from persisted event log
            session_id = request.query_params.get("session_id")
            if not session_id:
                session = adapter.session_store.get_or_create_interactive(user_id, adapter.agent_name)
                session_id = session.id
            events = adapter.session_store.read_events(session_id)
            snapshots = [e for e in events if e.get("type") == "prompt_snapshot"]
            breakdown = snapshots[-1].get("token_breakdown", {}) if snapshots else {}
            return JSONResponse(
                {
                    "prompt_snapshot": {
                        "messages": live_progress.latest_prompt_messages,
                        "token_breakdown": breakdown,
                    }
                }
            )

        # Fallback: breakdown only from event log
        session_id = request.query_params.get("session_id")
        if not session_id:
            session = adapter.session_store.get_or_create_interactive(user_id, adapter.agent_name)
            session_id = session.id

        events = adapter.session_store.read_events(session_id)
        snapshots = [e for e in events if e.get("type") == "prompt_snapshot"]
        if not snapshots:
            return JSONResponse({"prompt_snapshot": None})

        latest = snapshots[-1]
        return JSONResponse({"prompt_snapshot": {"token_breakdown": latest.get("token_breakdown", {})}})

    async def _compact(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        user_id = adapter.resolve_http_user(body.get("user_id", "web-anonymous"))

        session = adapter.session_store.get_or_create_interactive(user_id, adapter.agent_name)

        if session.message_count == 0:
            return JSONResponse({"error": "no session to compact"}, status_code=404)

        agent_name = request.path_params["agent"]
        old_conv_id = session.id

        if not adapter.session_store.begin_compaction(user_id, adapter.agent_name):
            return JSONResponse({"error": "compaction already in progress"}, status_code=409)

        adapter._broadcast_compaction(agent_name, started=True)
        try:
            instructions = body.get("instructions")
            await adapter._compact_session(session.id, instructions=instructions, reason="manual")
        except Exception as e:
            msg = str(e) or repr(e)
            logger.exception("Compaction failed for agent %s", adapter.agent_name)
            return JSONResponse({"error": f"compaction failed: {msg}"}, status_code=500)
        finally:
            adapter.session_store.end_compaction(user_id, adapter.agent_name)
            adapter._broadcast_compaction(agent_name, started=False)

        new_session = adapter.session_store.get_or_create_interactive(user_id, adapter.agent_name)
        self.event_bus.emit("agent_status", {"agent": agent_name})
        if new_session:
            self.event_bus.emit("session_update", {"action": "compacted", "id": new_session.id})
        return JSONResponse(
            {
                "status": "compacted",
                "old_conversation_id": old_conv_id,
                "new_conversation_id": new_session.id if new_session else None,
            }
        )

    async def _respond(self, request: Request) -> JSONResponse:
        """Submit a response to an active ask_user prompt."""
        adapter, err = self._get_adapter(request)
        if err:
            return err

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        response = body.get("response", "")
        if not isinstance(response, str):
            return JSONResponse({"error": "response must be a string"}, status_code=400)
        if len(response) > 10_000:
            return JSONResponse({"error": "response too long (max 10000 chars)"}, status_code=400)

        user_id = adapter.resolve_http_user(body.get("user_id", "web-anonymous"))
        agent_name = request.path_params["agent"]
        logger.info("[%s] respond from user_id=%s", agent_name, user_id)

        key = (agent_name, user_id)
        backend = self._active_backends.get(key)
        if not backend:
            return JSONResponse({"error": "no pending question for this agent/user"}, status_code=404)

        backend.submit_response(response)
        return JSONResponse({"status": "ok"})

    async def _upload(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        uploads_dir = adapter.agent_config.workspace_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        form = await request.form()
        files = form.getlist("files")
        if not files:
            return JSONResponse({"error": "no files provided"}, status_code=400)
        if len(files) > MAX_UPLOAD_FILES:
            return JSONResponse({"error": f"too many files (max {MAX_UPLOAD_FILES})"}, status_code=400)

        total_size = 0
        results = []
        written_paths = []
        for upload in files:
            content = await upload.read()
            total_size += len(content)
            if total_size > MAX_UPLOAD_TOTAL:
                for p in written_paths:
                    p.unlink(missing_ok=True)
                return JSONResponse({"error": "total upload size exceeds 100MB"}, status_code=413)

            name = _sanitize_filename(upload.filename or "upload")
            dest = uploads_dir / name
            if not dest.resolve().is_relative_to(uploads_dir.resolve()):
                continue
            dest, dedup_err = _deduplicate_dest(uploads_dir, name)
            if dedup_err:
                continue

            dest.write_bytes(content)
            written_paths.append(dest)
            file_size = len(content)
            mime_type, content_type = _file_handler._detect_content_type(dest)
            context_attach = _should_context_attach(dest, file_size)

            results.append(
                {
                    "name": dest.name,
                    "content_type": content_type.value,
                    "mime_type": mime_type,
                    "size": file_size,
                    "context_attach": context_attach,
                }
            )

        await form.close()
        return JSONResponse({"files": results})

    async def _chat(self, request: Request) -> Response:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        message = body.get("message", "").strip()
        uploaded_files = body.get("uploaded_files", [])
        if not isinstance(uploaded_files, list):
            uploaded_files = []

        if not message and not uploaded_files:
            return JSONResponse({"error": "message or uploaded_files is required"}, status_code=400)

        raw_user_id = body.get("user_id", "web-anonymous")
        agent_name = request.path_params["agent"]
        user_id = adapter.resolve_http_user(raw_user_id)
        logger.info("[%s] <- %s (http): %s", agent_name, user_id, message[:100])

        # Process uploaded files — only accept filenames, resolve against uploads dir
        uploaded_attachments = []
        workspace_only_files = []
        uploads_dir = adapter.agent_config.workspace_dir / "uploads"

        for file_info in uploaded_files:
            if not isinstance(file_info, dict):
                continue
            filename = _sanitize_filename(file_info.get("name", ""))
            file_path = (uploads_dir / filename).resolve()
            if not file_path.is_relative_to(uploads_dir.resolve()) or not file_path.exists():
                continue

            if _should_context_attach(file_path, file_path.stat().st_size):
                try:
                    attachment = _file_handler.fetch(str(file_path))
                    uploaded_attachments.append(attachment)
                except Exception as e:
                    logger.warning("Failed to create attachment for %s: %s", file_path, e)
                    workspace_only_files.append(filename)
            else:
                workspace_only_files.append(filename)

        if workspace_only_files:
            names = ", ".join(workspace_only_files)
            message += f"\n\n[Uploaded files available in workspace: {names}]"

        if uploaded_attachments:
            names = ", ".join(a.name for a in uploaded_attachments)
            message += f"\n\n[Attached files (content included below, saved to uploads/): {names}]"

        metadata = {"client_ip": request.client.host if request.client else "unknown"}
        if uploaded_attachments:
            metadata["uploaded_attachments"] = uploaded_attachments

        session_id = body.get("session_id")
        if session_id:
            metadata["conv_id_override"] = session_id

        channel_context = ChannelContext(
            source="http",
            channel_id=None,
            user_id=raw_user_id,
            reply_to=f"http:{raw_user_id}",
            metadata=metadata,
        )

        progress = SSEProgressHandler()
        progress.set_loop(asyncio.get_running_loop())
        session = adapter.session_store.get_or_create_interactive(user_id, adapter.agent_name)
        progress.set_event_persister(
            lambda payload: adapter.session_store.append_event(
                session.id,
                {
                    **payload,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        )
        custom_logger = SimpleNamespace(ui_handler=progress)

        interaction_backend = HTTPInteractionBackend(progress)
        backend_key = (agent_name, user_id)
        interaction_backend.pending_message = message
        self._active_backends[backend_key] = interaction_backend
        self._active_progress[backend_key] = progress

        async def run_agent():
            from tsugite.interaction import set_interaction_backend

            set_interaction_backend(interaction_backend)
            try:
                response = await adapter.handle_message(
                    user_id=user_id,
                    message=message,
                    channel_context=channel_context,
                    custom_logger=custom_logger,
                )
                # Only emit final_result if the EventBus didn't already
                # (FinalAnswerEvent fires during handle_message for normal completions,
                # but not for max_turns/error cases)
                logger.info("[%s] -> %s (http): %s", adapter.agent_name, user_id, (response or "")[:100])
                if not progress.has_final:
                    progress._emit("final_result", {"result": response})

                self.event_bus.emit("agent_status", {"agent": agent_name})
                self.event_bus.emit("history_update", {"agent": agent_name})

                # Emit session info for the web UI status bar
                session = adapter.session_store.get_or_create_interactive(user_id, adapter.agent_name)
                if session:
                    progress._emit(
                        "session_info",
                        {
                            "tokens": session.cumulative_tokens,
                            "context_limit": adapter.session_store.get_context_limit(adapter.agent_name),
                            "threshold": adapter.session_store.get_compaction_threshold(adapter.agent_name),
                            "message_count": session.message_count,
                            "model": adapter.resolve_model(),
                            "attachments": [a.name for a in adapter._get_all_attachments()],
                        },
                    )
            except asyncio.CancelledError:
                logger.info("[%s] Chat cancelled by user for %s", adapter.agent_name, user_id)
                progress._emit("cancelled", {"reason": "User cancelled"})
            except Exception as e:
                logger.exception("[%s] Chat error", adapter.agent_name)
                progress._emit("error", {"error": str(e)})
            finally:
                self._active_backends.pop(backend_key, None)
                self._active_chat_tasks.pop(backend_key, None)
                progress.signal_done()

        task = asyncio.create_task(run_agent())
        self._active_chat_tasks[backend_key] = task
        task.add_done_callback(lambda _: self._active_chat_tasks.pop(backend_key, None))

        return StreamingResponse(
            progress.event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def _cancel_chat(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        raw_user_id = body.get("user_id", "web-anonymous")
        user_id = adapter.resolve_http_user(raw_user_id)
        backend_key = (request.path_params["agent"], user_id)
        task = self._active_chat_tasks.get(backend_key)
        if task and not task.done():
            task.cancel()
            return JSONResponse({"status": "cancelled"})
        return JSONResponse({"error": "no active chat"}, status_code=404)

    def _require_auth_and_scheduler(self, request: Request) -> Optional[JSONResponse]:
        """Check auth and scheduler availability. Returns error response or None."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        if not self.scheduler:
            return JSONResponse({"error": "scheduler not available"}, status_code=503)
        return None

    def _schedule_action(self, schedule_id: str, action: str, status_label: str) -> JSONResponse:
        """Run a scheduler action (enable/disable/remove) with standard error handling."""
        try:
            getattr(self.scheduler, action)(schedule_id)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        self.event_bus.emit("schedule_update", {"action": status_label, "id": schedule_id})
        return JSONResponse({"status": status_label})

    async def _list_schedules(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_scheduler(request):
            return err
        return JSONResponse({"schedules": [asdict(e) for e in self.scheduler.list()]})

    async def _create_schedule(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_scheduler(request):
            return err

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        required = {"id", "agent", "prompt", "schedule_type"}
        missing = required - set(body.keys())
        if missing:
            return JSONResponse({"error": f"missing fields: {', '.join(missing)}"}, status_code=400)

        try:
            valid_fields = {f.name for f in dataclass_fields(ScheduleEntry)}
            entry = ScheduleEntry(**{k: v for k, v in body.items() if k in valid_fields})
            entry = self.scheduler.add(entry)
        except (ValueError, TypeError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)

        self.event_bus.emit("schedule_update", {"action": "created", "id": entry.id})
        return JSONResponse(asdict(entry), status_code=201)

    async def _get_schedule(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_scheduler(request):
            return err
        try:
            entry = self.scheduler.get(request.path_params["schedule_id"])
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        return JSONResponse(asdict(entry))

    async def _update_schedule(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_scheduler(request):
            return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        schedule_id = request.path_params["schedule_id"]
        allowed = {
            "prompt",
            "cron_expr",
            "run_at",
            "timezone",
            "agent",
            "schedule_type",
            "model",
            "agent_file",
            "max_turns",
        }
        fields = {k: v for k, v in body.items() if k in allowed}
        if not fields:
            return JSONResponse({"error": "no updatable fields provided"}, status_code=400)

        try:
            entry = self.scheduler.update(schedule_id, **fields)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        self.event_bus.emit("schedule_update", {"action": "updated", "id": schedule_id})
        return JSONResponse(asdict(entry))

    async def _delete_schedule(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_scheduler(request):
            return err
        return self._schedule_action(request.path_params["schedule_id"], "remove", "removed")

    async def _enable_schedule(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_scheduler(request):
            return err
        return self._schedule_action(request.path_params["schedule_id"], "enable", "enabled")

    async def _disable_schedule(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_scheduler(request):
            return err
        return self._schedule_action(request.path_params["schedule_id"], "disable", "disabled")

    async def _run_schedule(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_scheduler(request):
            return err
        schedule_id = request.path_params["schedule_id"]
        try:
            self.scheduler.get(schedule_id)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        self.scheduler.fire_now(schedule_id)
        self.event_bus.emit("schedule_update", {"action": "triggered", "id": schedule_id})
        return JSONResponse({"status": "triggered", "schedule_id": schedule_id})

    async def _cleanup_schedules(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_scheduler(request):
            return err
        removed = self.scheduler.cleanup()
        return JSONResponse({"removed": removed, "count": len(removed)})

    async def _schedule_sessions(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_scheduler(request):
            return err
        schedule_id = request.path_params["schedule_id"]
        # Find any adapter to access the session store
        adapter = next(iter(self.adapters.values()), None)
        if not adapter:
            return JSONResponse({"error": "no adapters"}, status_code=500)
        sessions = adapter.session_store.list_sessions(parent_id=schedule_id, source="schedule")
        return JSONResponse(
            {
                "schedule_id": schedule_id,
                "sessions": [
                    {
                        "id": s.id,
                        "status": s.status,
                        "created_at": s.created_at,
                        "last_active": s.last_active,
                        "result": (s.result or "")[:500],
                        "error": s.error,
                    }
                    for s in sessions
                ],
            }
        )

    # ── Session / Review API ──

    def _require_auth_and_sessions(self, request: Request) -> Optional[JSONResponse]:
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        if not self.session_runner:
            return JSONResponse({"error": "session runner not available"}, status_code=503)
        return None

    def _session_detail(self, session_id: str) -> dict:
        return self.session_runner.store.session_detail(session_id)

    async def _api_list_sessions(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        state = request.query_params.get("state")
        sessions = self.session_runner.store.list_sessions(status=state)
        return JSONResponse(
            {
                "sessions": [
                    {
                        "id": s.id,
                        "agent": s.agent,
                        "source": s.source,
                        "state": s.status,
                        "prompt": s.prompt or "",
                        "created_at": s.created_at,
                        "updated_at": s.last_active,
                        "error": s.error,
                        "title": s.title,
                        "metadata": s.metadata or {},
                    }
                    for s in sessions
                ]
            }
        )

    async def _api_start_session(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        prompt = body.get("prompt", "").strip()
        agent = body.get("agent", "")
        if not prompt or not agent:
            return JSONResponse({"error": "prompt and agent are required"}, status_code=400)
        if agent not in self.adapters:
            return JSONResponse({"error": f"unknown agent: {agent}"}, status_code=400)

        from tsugite.daemon.session_store import Session, SessionSource

        session = Session(
            id=body.get("session_id", ""),
            agent=agent,
            source=SessionSource.BACKGROUND.value,
            prompt=prompt,
            model=body.get("model"),
            agent_file=body.get("agent_file"),
            notify=body.get("notify", []),
        )

        try:
            result = self.session_runner.start_session(session)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

        self.event_bus.emit("session_update", {"action": "started", "id": result.id})
        return JSONResponse(
            {"id": result.id, "status": result.status, "created_at": result.created_at},
            status_code=201,
        )

    async def _api_get_session(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        try:
            return JSONResponse(self._session_detail(session_id))
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)

    async def _api_update_session(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        title = body.get("title")
        status = body.get("status")
        if title is None and status is None:
            return JSONResponse({"error": "No updatable fields provided"}, status_code=400)
        try:
            result = {}
            if title is not None:
                self.session_runner.rename_session(session_id, title)
                result["title"] = title
            if status is not None:
                if status != "completed":
                    return JSONResponse({"error": "Only 'completed' status is allowed"}, status_code=400)
                self.session_runner.store.update_session(session_id, status=status)
                self.event_bus.emit("session_update", {"action": "completed", "id": session_id})
                result["status"] = status
            return JSONResponse({"ok": True, **result})
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)

    async def _api_cancel_session(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        try:
            self.session_runner.cancel_session(session_id)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        self.event_bus.emit("session_update", {"action": "cancelled", "id": session_id})
        return JSONResponse({"status": "cancelled"})

    async def _api_restart_session(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        try:
            old = self.session_runner.store.get_session(session_id)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)

        from tsugite.daemon.session_store import Session, SessionSource, SessionStatus

        restartable = {SessionStatus.FAILED.value, SessionStatus.CANCELLED.value}
        if old.status not in restartable:
            return JSONResponse({"error": f"cannot restart session in '{old.status}' state"}, status_code=400)

        new_session = Session(
            id="",
            agent=old.agent,
            source=old.source or SessionSource.BACKGROUND.value,
            prompt=old.prompt,
            model=old.model,
            agent_file=old.agent_file,
            notify=old.notify,
        )
        try:
            result = self.session_runner.start_session(new_session)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        self.event_bus.emit("session_update", {"action": "restarted", "id": result.id})
        return JSONResponse(
            {"id": result.id, "status": result.status, "restarted_from": session_id},
            status_code=201,
        )

    async def _api_session_events(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        try:
            self.session_runner.store.get_session(session_id)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        events = self.session_runner.store.read_events(session_id)
        return JSONResponse({"events": events})

    async def _api_get_metadata(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        try:
            session = self.session_runner.store.get_session(session_id)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        return JSONResponse({"metadata": session.metadata or {}})

    async def _api_update_metadata(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "Body must be a JSON object"}, status_code=400)
        try:
            session = self.session_runner.update_session_metadata(session_id, body)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse({"ok": True, "metadata": session.metadata or {}})

    async def _api_delete_metadata(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        key = request.path_params["key"]
        try:
            session = self.session_runner.delete_session_metadata(session_id, key)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse({"ok": True, "metadata": session.metadata or {}})

    async def _api_get_scratchpad(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        try:
            session = self.session_runner.store.get_session(session_id)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        return JSONResponse({"scratchpad": session.scratchpad})

    async def _api_update_scratchpad(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        content = body.get("content", "")
        if not isinstance(content, str):
            return JSONResponse({"error": "content must be a string"}, status_code=400)
        from tsugite.tools.scratchpad import SCRATCHPAD_HARD_LIMIT

        if len(content) > SCRATCHPAD_HARD_LIMIT:
            return JSONResponse(
                {"error": f"Content exceeds hard limit of {SCRATCHPAD_HARD_LIMIT} chars"}, status_code=400
            )
        try:
            self.session_runner.store.update_session(session_id, scratchpad=content)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        self.event_bus.emit("session_update", {"action": "scratchpad_updated", "id": session_id})
        return JSONResponse({"ok": True, "scratchpad": content})

    async def _list_webhooks(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        webhooks = [
            {"token": w.token, "agent": w.agent, "source": w.source, "created_at": w.created_at}
            for w in self.webhook_store.list()
        ]
        return JSONResponse({"webhooks": webhooks})

    async def _create_webhook(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        agent = body.get("agent", "")
        source = body.get("source", "")
        if not agent or not source:
            return JSONResponse({"error": "agent and source are required"}, status_code=400)
        if agent not in self.agent_configs:
            return JSONResponse({"error": f"unknown agent: {agent}"}, status_code=400)

        try:
            entry = self.webhook_store.add(agent=agent, source=source, token=body.get("token"))
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

        return JSONResponse(
            {
                "token": entry.token,
                "agent": entry.agent,
                "source": entry.source,
                "created_at": entry.created_at,
            },
            status_code=201,
        )

    async def _delete_webhook(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        token = request.path_params["token"]
        try:
            self.webhook_store.remove(token)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        return JSONResponse({"status": "removed"})

    async def _webhook(self, request: Request) -> JSONResponse:
        token = request.path_params["token"]
        client_ip = request.client.host if request.client else "unknown"
        webhook = self.webhook_store.get(token)
        if not webhook:
            logger.warning("Webhook rejected: invalid token [%s] from %s", token[:8], client_ip)
            return JSONResponse({"error": "invalid webhook token"}, status_code=404)

        logger.info("Received webhook [%s] from %s", token[:8], client_ip)

        if webhook.agent not in self.agent_configs:
            return JSONResponse({"error": "webhook agent not configured"}, status_code=500)

        try:
            raw = await request.body()
        except Exception:
            return JSONResponse({"error": "failed to read body"}, status_code=400)

        try:
            payload_data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload_data = raw.decode("utf-8", errors="replace")

        event_type = ""
        if isinstance(payload_data, dict):
            event_type = payload_data.get("event") or payload_data.get("type") or payload_data.get("action") or ""
        logger.info(
            "Webhook [%s] source: %s | event: %s | agent: %s",
            token[:8],
            webhook.source,
            event_type or "unknown",
            webhook.agent,
        )

        agent_config = self.agent_configs[webhook.agent]
        inbox_dir = agent_config.workspace_dir / "inbox" / "webhooks"
        inbox_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        filename = f"{now.strftime('%Y%m%dT%H%M%S')}-{webhook.source}.json"
        envelope = {
            "source": webhook.source,
            "agent": webhook.agent,
            "received_at": now.isoformat(),
            "headers": {k: v for k, v in request.headers.items() if k.lower() != "authorization"},
            "payload": payload_data,
        }
        (inbox_dir / filename).write_text(json.dumps(envelope, indent=2, default=str))
        logger.info("Webhook [%s] saved to inbox: %s", token[:8], filename)

        return JSONResponse({"status": "accepted", "file": filename}, status_code=202)

    def _get_allowed_agent_dirs(self) -> list[tuple[Path, str, bool]]:
        """Return (directory, source_label, is_readonly) for all agent directories."""
        dirs: list[tuple[Path, str, bool]] = [(get_builtin_agents_path(), "builtin", True)]
        seen: set[Path] = set()
        for cfg in self.agent_configs.values():
            for subdir in [cfg.workspace_dir / ".tsugite", cfg.workspace_dir / "agents"]:
                resolved = subdir.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    dirs.append((subdir, "project", False))
        for gpath in get_global_agents_paths():
            dirs.append((gpath, "global", False))
        return dirs

    def _validate_md_path(
        self, path_str: str, allowed_dirs: list[tuple[Path, str, bool]]
    ) -> tuple[Path, bool, Optional[JSONResponse]]:
        """Validate a markdown file path is within allowed directories.

        Returns (resolved_path, is_readonly, error_response_or_none).
        """
        try:
            resolved = Path(path_str).resolve()
        except (ValueError, OSError):
            return Path(), False, JSONResponse({"error": "invalid path"}, status_code=400)
        if resolved.suffix != ".md":
            return Path(), False, JSONResponse({"error": "only .md files allowed"}, status_code=400)
        for dir_path, _, readonly in allowed_dirs:
            try:
                resolved.relative_to(dir_path.resolve())
                return resolved, readonly, None
            except ValueError:
                continue
        return Path(), False, JSONResponse({"error": "path not in allowed directories"}, status_code=403)

    def _collect_md_files(self, allowed_dirs: list[tuple[Path, str, bool]], glob_pattern: str = "*.md") -> list[dict]:
        """Collect markdown files from allowed directories with frontmatter metadata."""
        files = []
        seen_paths: set[Path] = set()
        for dir_path, source, readonly in allowed_dirs:
            if not dir_path.is_dir():
                continue
            for md_file in sorted(dir_path.glob(glob_pattern)):
                resolved = md_file.resolve()
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                name, description = md_file.stem, ""
                try:
                    content = md_file.read_text(encoding="utf-8")
                    fm, _ = parse_yaml_frontmatter(content, str(md_file))
                    name = fm.get("name", md_file.stem)
                    description = fm.get("description", "")
                except Exception as e:
                    logger.warning("Failed to parse frontmatter %s: %s", md_file, e)
                files.append(
                    {
                        "path": str(resolved),
                        "name": name,
                        "source": source,
                        "readonly": readonly,
                        "description": description,
                    }
                )
        return files

    async def _read_md_file(self, request: Request, allowed_dirs: list[tuple[Path, str, bool]]) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        path_str = request.query_params.get("path", "")
        if not path_str:
            return JSONResponse({"error": "path parameter required"}, status_code=400)
        resolved, readonly, err = self._validate_md_path(path_str, allowed_dirs)
        if err:
            return err
        if not resolved.exists():
            return JSONResponse({"error": "file not found"}, status_code=404)
        try:
            content = resolved.read_text(encoding="utf-8")
        except OSError as e:
            return JSONResponse({"error": f"read failed: {e}"}, status_code=500)
        return JSONResponse({"path": str(resolved), "content": content, "readonly": readonly})

    async def _save_md_file(self, request: Request, allowed_dirs: list[tuple[Path, str, bool]]) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        path_str = body.get("path", "")
        content = body.get("content")
        if not path_str or content is None:
            return JSONResponse({"error": "path and content required"}, status_code=400)
        resolved, readonly, err = self._validate_md_path(path_str, allowed_dirs)
        if err:
            return err
        if readonly:
            return JSONResponse({"error": "file is read-only (builtin)"}, status_code=403)
        if not resolved.exists():
            return JSONResponse({"error": "file not found"}, status_code=404)
        try:
            resolved.write_text(content, encoding="utf-8")
        except OSError as e:
            return JSONResponse({"error": f"write failed: {e}"}, status_code=500)
        return JSONResponse({"status": "saved"})

    async def _list_agent_files(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        return JSONResponse({"files": self._collect_md_files(self._get_allowed_agent_dirs(), "*.md")})

    async def _read_agent_file(self, request: Request) -> JSONResponse:
        return await self._read_md_file(request, self._get_allowed_agent_dirs())

    async def _save_agent_file(self, request: Request) -> JSONResponse:
        return await self._save_md_file(request, self._get_allowed_agent_dirs())

    def _get_allowed_skill_dirs(self) -> list[tuple[Path, str, bool]]:
        """Return (directory, source_label, is_readonly) for all skill directories."""
        dirs: list[tuple[Path, str, bool]] = [(get_builtin_skills_path(), "builtin", True)]
        seen: set[Path] = set()
        for cfg in self.agent_configs.values():
            for subdir in [cfg.workspace_dir / ".tsugite" / "skills", cfg.workspace_dir / "skills"]:
                resolved = subdir.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    dirs.append((subdir, "project", False))
        dirs.append((Path.home() / ".config" / "tsugite" / "skills", "global", False))
        return dirs

    async def _list_skill_files(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        return JSONResponse({"files": self._collect_md_files(self._get_allowed_skill_dirs(), "**/*.md")})

    async def _read_skill_file(self, request: Request) -> JSONResponse:
        return await self._read_md_file(request, self._get_allowed_skill_dirs())

    async def _save_skill_file(self, request: Request) -> JSONResponse:
        return await self._save_md_file(request, self._get_allowed_skill_dirs())

    # -- Workspace browser endpoints --

    def _validate_workspace_path(
        self, adapter: "HTTPAgentAdapter", path_str: str
    ) -> tuple[Path, Optional[JSONResponse]]:
        """Validate a workspace file path stays within the workspace directory."""
        workspace_dir = adapter.agent_config.workspace_dir
        try:
            resolved = (workspace_dir / path_str).resolve()
        except (ValueError, OSError):
            return Path(), JSONResponse({"error": "invalid path"}, status_code=400)
        if not resolved.is_relative_to(workspace_dir.resolve()):
            return Path(), JSONResponse({"error": "path outside workspace"}, status_code=403)
        return resolved, None

    async def _list_workspace_files(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        workspace_dir = adapter.agent_config.workspace_dir
        if not workspace_dir.is_dir():
            return JSONResponse({"entries": [], "subdir": "", "workspace_dir": str(workspace_dir)})

        subdir = request.query_params.get("subdir", "")
        if subdir:
            target, path_err = self._validate_workspace_path(adapter, subdir)
            if path_err:
                return path_err
            if not target.is_dir():
                return JSONResponse({"error": "not a directory"}, status_code=400)
        else:
            target = workspace_dir

        from tsugite.tools.fs import _build_gitignore_matcher

        gitignore_spec = _build_gitignore_matcher(workspace_dir)
        entries = []
        try:
            import stat as stat_mod

            for item in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                try:
                    st = item.lstat()
                except OSError:
                    continue
                if stat_mod.S_ISLNK(st.st_mode):
                    continue
                is_dir = stat_mod.S_ISDIR(st.st_mode)
                rel = str(item.relative_to(workspace_dir))
                if gitignore_spec and gitignore_spec.match_file(rel + ("/" if is_dir else "")):
                    continue
                if is_dir:
                    entries.append({"path": rel, "name": item.name, "is_dir": True})
                elif stat_mod.S_ISREG(st.st_mode) and _is_text_mime(item):
                    entries.append(
                        {
                            "path": rel,
                            "name": item.name,
                            "is_dir": False,
                            "size": st.st_size,
                            "modified": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                        }
                    )
        except OSError as e:
            return JSONResponse({"error": f"listing failed: {e}"}, status_code=500)

        return JSONResponse({"entries": entries, "subdir": subdir, "workspace_dir": str(workspace_dir)})

    async def _read_workspace_file(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        path_str = request.query_params.get("path", "")
        if not path_str:
            return JSONResponse({"error": "path parameter required"}, status_code=400)

        resolved, path_err = self._validate_workspace_path(adapter, path_str)
        if path_err:
            return path_err
        if not resolved.exists():
            return JSONResponse({"error": "file not found"}, status_code=404)
        if resolved.is_dir():
            return JSONResponse({"error": "path is a directory"}, status_code=400)

        st = resolved.stat()

        if not _is_text_mime(resolved):
            return JSONResponse({"path": path_str, "content": None, "is_text": False, "size": st.st_size})

        max_size = self.config.max_workspace_file_size
        if st.st_size > max_size:
            return JSONResponse(
                {"error": f"file too large (max {max_size // 1024}KB for text viewing)"}, status_code=413
            )

        try:
            content = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return JSONResponse({"path": path_str, "content": None, "is_text": False, "size": st.st_size})
        except OSError as e:
            return JSONResponse({"error": f"read failed: {e}"}, status_code=500)

        return JSONResponse({"path": path_str, "content": content, "is_text": True})

    async def _save_workspace_file(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        path_str = body.get("path", "")
        content = body.get("content")
        if not path_str or content is None:
            return JSONResponse({"error": "path and content required"}, status_code=400)

        max_size = self.config.max_workspace_file_size
        if len(content) > max_size:
            return JSONResponse({"error": f"content too large (max {max_size // 1024}KB)"}, status_code=413)

        resolved, path_err = self._validate_workspace_path(adapter, path_str)
        if path_err:
            return path_err
        if not resolved.exists():
            return JSONResponse({"error": "file not found"}, status_code=404)
        if not _is_text_mime(resolved):
            return JSONResponse({"error": "file type not editable"}, status_code=400)

        try:
            resolved.write_text(content, encoding="utf-8")
        except OSError as e:
            return JSONResponse({"error": f"write failed: {e}"}, status_code=500)

        return JSONResponse({"status": "saved"})

    async def _attach_workspace_file(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        path_str = request.query_params.get("path", "")
        if not path_str:
            return JSONResponse({"error": "path parameter required"}, status_code=400)

        resolved, path_err = self._validate_workspace_path(adapter, path_str)
        if path_err:
            return path_err
        if not resolved.exists():
            return JSONResponse({"error": "file not found"}, status_code=404)
        if not resolved.is_file():
            return JSONResponse({"error": "not a file"}, status_code=400)

        uploads_dir = adapter.agent_config.workspace_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        dest, dedup_err = _deduplicate_dest(uploads_dir, resolved.name)
        if dedup_err:
            return JSONResponse({"error": dedup_err}, status_code=409)
        if not dest.resolve().is_relative_to(uploads_dir.resolve()):
            return JSONResponse({"error": "invalid filename"}, status_code=400)

        try:
            shutil.copy2(resolved, dest)
        except OSError as e:
            return JSONResponse({"error": f"copy failed: {e}"}, status_code=500)

        file_size = dest.stat().st_size
        mime_type, content_type = _file_handler._detect_content_type(dest)
        context_attach = _should_context_attach(dest, file_size)

        return JSONResponse(
            {
                "files": [
                    {
                        "name": dest.name,
                        "content_type": content_type.value,
                        "mime_type": mime_type,
                        "size": file_size,
                        "context_attach": context_attach,
                    }
                ]
            }
        )

    async def _push_vapid_key(self, request: Request) -> JSONResponse:
        if not self.vapid_public_key:
            return JSONResponse({"error": "web push not configured"}, status_code=404)
        return JSONResponse({"public_key": self.vapid_public_key})

    async def _push_subscribe(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        if not self.push_store:
            return JSONResponse({"error": "web push not configured"}, status_code=404)
        body = await request.json()
        if not body.get("endpoint"):
            return JSONResponse({"error": "missing endpoint"}, status_code=400)
        self.push_store.subscribe(body)
        return JSONResponse({"status": "subscribed"})

    async def _push_unsubscribe(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        if not self.push_store:
            return JSONResponse({"error": "web push not configured"}, status_code=404)
        body = await request.json()
        endpoint = body.get("endpoint")
        if not endpoint:
            return JSONResponse({"error": "missing endpoint"}, status_code=400)
        self.push_store.unsubscribe(endpoint)
        return JSONResponse({"status": "unsubscribed"})

    async def _secrets_list(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        from tsugite.secrets import get_backend

        return JSONResponse({"secrets": get_backend().list_names()})

    async def _secrets_set(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        from tsugite.secrets import get_backend

        name = request.path_params["name"]
        body = await request.json()
        value = body.get("value")
        if not value:
            return JSONResponse({"error": "value is required"}, status_code=400)
        try:
            get_backend().set(name, value)
        except NotImplementedError:
            return JSONResponse({"error": "backend does not support writing"}, status_code=400)
        return JSONResponse({"status": "ok", "name": name})

    async def _secrets_delete(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        from tsugite.secrets import get_backend

        name = request.path_params["name"]
        try:
            deleted = get_backend().delete(name)
        except NotImplementedError:
            return JSONResponse({"error": "backend does not support deletion"}, status_code=400)
        if not deleted:
            return JSONResponse({"error": "secret not found"}, status_code=404)
        return JSONResponse({"status": "ok", "name": name})

    async def _kv_namespaces(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        from tsugite.kvstore import get_backend

        return JSONResponse({"namespaces": get_backend().list_namespaces()})

    async def _kv_keys(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        from tsugite.kvstore import get_backend

        namespace = request.path_params["namespace"]
        prefix = request.query_params.get("prefix", "")
        keys = get_backend().list_keys(namespace, prefix)
        return JSONResponse({"namespace": namespace, "keys": keys})

    async def _kv_get(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        from tsugite.kvstore import get_backend

        namespace = request.path_params["namespace"]
        key = request.path_params["key"]
        result = get_backend().get_with_metadata(namespace, key)
        if result is None:
            return JSONResponse({"error": "key not found"}, status_code=404)
        return JSONResponse({"namespace": namespace, "key": key, **result})

    async def _events(self, request: Request) -> Response:
        if err := self._check_auth(request):
            return err
        queue = self.event_bus.subscribe()

        async def generator():
            try:
                async for chunk in sse_stream(queue):
                    yield chunk
            finally:
                self.event_bus.unsubscribe(queue)

        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    async def _serve_sw(self, request: Request) -> Response:
        sw_path = WEB_DIR / "sw.js"
        if not sw_path.exists():
            return JSONResponse({"error": "service worker not found"}, status_code=404)
        return Response(sw_path.read_bytes(), media_type="application/javascript")

    async def _serve_ui(self, request: Request) -> Response:
        ui_path = WEB_DIR / "index.html"
        if not ui_path.exists():
            return JSONResponse({"error": "web UI not found"}, status_code=404)
        return HTMLResponse(ui_path.read_text(), headers={"Cache-Control": "no-cache"})

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
            if hasattr(self, "event_bus"):
                for q in list(self.event_bus._subscribers):
                    try:
                        q.put_nowait(None)
                    except asyncio.QueueFull:
                        pass
            self._server.should_exit = True
