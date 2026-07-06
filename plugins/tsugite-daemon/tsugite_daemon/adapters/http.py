"""HTTP API adapter with SSE streaming and webhook inbox."""

import asyncio
import json
import logging
import mimetypes
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Optional
from uuid import uuid4

if TYPE_CHECKING:
    from tsugite_daemon.session_store import SessionStore

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from tsugite.agent_inheritance import iter_agent_search_paths
from tsugite.attachments.base import AttachmentContentType
from tsugite.attachments.file import FileHandler
from tsugite.events.base import BaseEvent
from tsugite.skill_discovery import get_builtin_skills_path
from tsugite.ui.jsonl import JSONLUIHandler
from tsugite.utils import parse_yaml_frontmatter
from tsugite_daemon.adapters.base import _PERSIST_EVENT_TYPES, BaseAdapter, ChannelContext
from tsugite_daemon.config import AgentConfig, HTTPConfig
from tsugite_daemon.scheduler import ScheduleEntry, entry_to_dict
from tsugite_daemon.webhook_store import WebhookStore

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def build_session_event_persister(session_store: "SessionStore", session_id: str) -> Callable:
    """Persist selected agent events to the per-session JSONL, and sync
    `Session.cumulative_tokens` to `prompt_snapshot.token_breakdown.total` so
    the UI badge matches the prompt inspector. See `tests/test_displayed_token_count.py`
    for the failure modes this addresses.
    """

    def _persist(payload: dict[str, Any]) -> None:
        session_store.append_event(
            session_id,
            {**payload, "timestamp": datetime.now(timezone.utc).isoformat()},
        )
        if payload.get("type") != "prompt_snapshot":
            return
        total = (payload.get("token_breakdown") or {}).get("total")
        if isinstance(total, int) and total > 0:
            session_store.set_cumulative_tokens(session_id, total)

    return _persist


def _resolve_full_model_id(model: str) -> str:
    """Return 'provider:full-model-id' so UI can show e.g. claude_code:opus -> claude-opus-4-7.

    Returns the input unchanged on any error or when no alias is involved.
    """
    if not model or ":" not in model:
        return model
    try:
        from tsugite.models import get_model_id, parse_model_string

        provider, _, _ = parse_model_string(model)
        return f"{provider}:{get_model_id(model)}"
    except Exception:
        return model


_WEB_ASSETS_VERSION_CACHE: Optional[tuple[float, str]] = None  # (computed_at_monotonic, version)
_WEB_ASSETS_VERSION_TTL = 5.0


def _web_assets_version() -> str:
    """Latest mtime across web assets, as an int string.

    Cached with a short TTL: PWA /sw.js fetches shouldn't rglob the whole web
    tree on the event loop per request, but editing assets under a running
    daemon (dev, in-place upgrades) must still bust client caches without a
    daemon restart.
    """
    global _WEB_ASSETS_VERSION_CACHE
    now = time.monotonic()
    if _WEB_ASSETS_VERSION_CACHE is not None and now - _WEB_ASSETS_VERSION_CACHE[0] < _WEB_ASSETS_VERSION_TTL:
        return _WEB_ASSETS_VERSION_CACHE[1]
    latest = 0.0
    for p in WEB_DIR.rglob("*"):
        if p.is_file():
            latest = max(latest, p.stat().st_mtime)
    version = str(int(latest))
    _WEB_ASSETS_VERSION_CACHE = (now, version)
    return version


class _NoCacheStaticFiles(StaticFiles):
    """StaticFiles that asks browsers to revalidate JS/CSS/JSON on each load.

    Relies on Starlette's built-in ETag/Last-Modified handling for 304s.
    """

    _REVALIDATE_SUFFIXES = (".js", ".mjs", ".css", ".json", ".map")

    def file_response(self, full_path, stat_result, scope, status_code=200):
        response = super().file_response(full_path, stat_result, scope, status_code)
        if str(full_path).endswith(self._REVALIDATE_SUFFIXES):
            response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response


MAX_TEXT_ATTACH_SIZE = 50 * 1024  # 50KB — ~12K tokens
MAX_BINARY_ATTACH_SIZE = 10 * 1024 * 1024  # 10MB
MAX_UPLOAD_TOTAL = 100 * 1024 * 1024  # 100MB per request
MAX_WEBHOOK_BODY = 5 * 1024 * 1024  # 5MB per delivery (GitHub caps payloads at ~25MB; ours are envelopes)
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


def _format_upload_message_suffix(workspace_only_files: list[str], attachment_names: list[str]) -> str:
    """Hint appended to the user's message describing where uploaded files were saved.

    Non-inlined files land in <workspace>/uploads/ and the agent has to open them itself,
    so the hint must give a tool-ready relative path, not a bare filename.
    """
    suffix = ""
    if workspace_only_files:
        paths = ", ".join(f"uploads/{n}" for n in workspace_only_files)
        suffix += f"\n\n[Uploaded files saved to the workspace, readable at: {paths}]"
    if attachment_names:
        names = ", ".join(attachment_names)
        suffix += f"\n\n[Attached files (content included below, saved to uploads/): {names}]"
    return suffix


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


# Events the per-chat SSE already delivers to the active client. Skipping them
# on the cross-session broadcaster prevents the active tab from rendering the
# same progress twice, and keeps the broadcast to what other surfaces (sidebar
# progress cache, non-active session detail view) actually read.
#
# Turn-end events (final_result, error, cancelled) are deliberately included
# here even though other tabs won't see them live: the same _emit() flow also
# fires history_update on the global event bus after the turn settles, which
# triggers loadHistory() in those tabs and rebuilds the message list from JSONL.
# Broadcasting turn-end events too would race the active tab's per-chat reader
# (the session's `sending` flag in sessionsState is cleared in streaming.js's
# finally block, leaving a window where late-arriving session_event(final_result)
# bypasses the dedup guard and pushes a duplicate bubble until the next reload).
_BROADCAST_SKIP_EVENTS = frozenset(
    {
        "stream_chunk",
        "stream_complete",
        "prompt_snapshot",
        "final_result",
        "error",
        "cancelled",
    }
)


class SSEProgressHandler(JSONLUIHandler):
    """Converts agent events to SSE messages via an async queue."""

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.done = False
        self.has_final = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._persist_event: Optional[Callable] = None
        self._broadcaster: Optional["SSEBroadcaster"] = None
        self._session_id: Optional[str] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def set_event_persister(self, fn: Callable):
        """Set a callback to persist select events to the session event log."""
        self._persist_event = fn

    def set_broadcaster(self, broadcaster: "SSEBroadcaster") -> None:
        self._broadcaster = broadcaster

    def set_session_id(self, session_id: str) -> None:
        self._session_id = session_id

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

        if event_type in _PERSIST_EVENT_TYPES and self._persist_event:
            self._persist_event(payload)

        if self._broadcaster and self._session_id and event_type not in _BROADCAST_SKIP_EVENTS:
            self._broadcaster.emit(
                "session_event",
                {"session_id": self._session_id, "event_type": event_type, **data},
            )

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
        # Yield once so any pending call_soon_threadsafe(put_nowait) callbacks
        # scheduled from worker threads get a chance to land in the queue before
        # we drain. Without this, events emitted just before signal_done can
        # race past the drain and be silently dropped.
        await asyncio.sleep(0)
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


@dataclass
class ActiveChat:
    """In-flight chat interaction. One per (agent, user, session_id) triple.

    Consolidates what used to be three parallel maps keyed by the same triple
    so adding a fourth piece of per-chat state doesn't mean adding a fourth map
    and remembering to keep their lifecycles in sync.
    """

    backend: HTTPInteractionBackend
    progress: SSEProgressHandler
    task: Optional[asyncio.Task] = None
    # Cooperative cancel signal: cancelling the task tears down the SSE stream but
    # cannot stop the agent loop running in a to_thread worker. The worker checks
    # this Event at safe checkpoints and exits cleanly. See tsugite/cancellation.py.
    cancel_event: threading.Event = field(default_factory=threading.Event)


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
        self.jobs_orchestrator = None  # Set by Gateway after JobsOrchestrator is created
        self.job_store = None  # Set by Gateway alongside jobs_orchestrator
        self.terminal_store = None  # Set by Gateway when terminal viewer is wired
        self.pty_manager = None  # Set by Gateway alongside terminal_store
        self.push_store = None  # Set by Gateway if web-push is configured
        self.vapid_public_key = None  # Set by Gateway if web-push is configured
        self._active_chats: dict[tuple[str, str, str], ActiveChat] = {}
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
            Route("/api/agents/{agent}/sessions/{session_id}/branch", self._branch, methods=["POST"]),
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
            Route("/api/agents/{agent}/unload-skill", self._unload_skill, methods=["POST"]),
            Route("/api/agents/{agent}/effort-levels", self._effort_levels, methods=["GET"]),
            Route("/api/models", self._list_models, methods=["GET"]),
            Route("/api/sessions/{session_id}/settings", self._session_settings_get, methods=["GET"]),
            Route("/api/sessions/{session_id}/settings", self._session_settings_patch, methods=["PATCH"]),
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
            Route("/api/jobs", self._api_list_jobs, methods=["GET"]),
            Route("/api/jobs/{job_id}/cancel", self._api_cancel_job, methods=["POST"]),
            Route("/api/jobs/{job_id}/mark-done", self._api_mark_job_done, methods=["POST"]),
            Route("/api/jobs/{job_id}/retry", self._api_retry_job, methods=["POST"]),
            Route("/api/terminals", self._api_list_terminals, methods=["GET"]),
            Route("/api/terminals", self._api_create_terminal, methods=["POST"]),
            Route("/api/terminals/{terminal_id}", self._api_get_terminal, methods=["GET"]),
            Route("/api/terminals/{terminal_id}/kill", self._api_kill_terminal, methods=["POST"]),
            Route("/api/terminals/{terminal_id}/stdin", self._api_terminal_stdin, methods=["POST"]),
            Route("/api/terminals/{terminal_id}/restart", self._api_restart_terminal, methods=["POST"]),
            Route("/api/terminals/{terminal_id}/stream", self._api_terminal_stream, methods=["GET"]),
            Route("/api/sessions/{session_id}/restart", self._api_restart_session, methods=["POST"]),
            Route("/api/sessions/{session_id}/events", self._api_session_events, methods=["GET"]),
            Route("/api/sessions/{session_id}/pin", self._api_pin_session, methods=["POST"]),
            Route("/api/sessions/{session_id}/unpin", self._api_unpin_session, methods=["POST"]),
            Route("/api/sessions/pinned/reorder", self._api_reorder_pins, methods=["POST"]),
            # NB: clear-primary literal must precede {session_id}/set-primary -- Starlette matches in order.
            Route("/api/sessions/clear-primary", self._api_clear_primary, methods=["POST"]),
            Route("/api/sessions/{session_id}/set-primary", self._api_set_primary, methods=["POST"]),
            Route("/api/sessions/{session_id}/mark-viewed", self._api_mark_viewed, methods=["POST"]),
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
            Route("/api/skills/issues", self._list_skill_issues, methods=["GET"]),
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
            Route("/api/commands", self._list_commands, methods=["GET"]),
            Route("/api/agents/{agent}/commands/{command_name}", self._run_command, methods=["POST"]),
            Route("/api/usage/summary", self._usage_summary, methods=["GET"]),
            Route("/api/usage/agents", self._usage_agents, methods=["GET"]),
            Route("/api/usage/models", self._usage_models, methods=["GET"]),
            Route("/api/usage/total", self._usage_total, methods=["GET"]),
            Mount("/static", app=_NoCacheStaticFiles(directory=str(WEB_DIR)), name="static"),
            Route("/sw.js", self._serve_sw, methods=["GET"]),
            Route("/", self._serve_ui, methods=["GET"]),
        ]
        return Starlette(routes=routes)

    async def _health(self, request: Request) -> JSONResponse:
        try:
            from importlib.metadata import version

            v = version("tsugite-cli")
        except Exception:  # noqa: BLE001 — fall back to in-tree constant
            from tsugite import __version__ as v
        return JSONResponse({"status": "ok", "version": v, "agents": list(self.adapters.keys())})

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
        from tsugite_daemon.commands import get_commands

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

        from tsugite_daemon.commands import CommandError

        try:
            result = await cmd.handler(adapter, **filtered)
        except CommandError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
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

    def _session_busy(self, agent_name: str, session) -> bool:
        """The one definition of busy, shared by the sessions payload, /status,
        and the /chat 409 guard - the server must never 409 a send while
        reporting the session idle. True when the durable turn_in_flight marker
        is set OR a live HTTP chat task exists (covers the brief window between
        task creation and begin_turn)."""
        if session.turn_in_flight:
            return True
        return any(
            a == agent_name and sid == session.id and chat.task is not None and not chat.task.done()
            for (a, _user, sid), chat in self._active_chats.items()
        )

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

    async def _list_sessions(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        source = request.query_params.get("source")
        status = request.query_params.get("status")
        parent_id = request.query_params.get("parent_id")
        include_superseded = request.query_params.get("include_superseded", "").lower() in ("1", "true", "yes")
        try:
            limit = max(1, min(int(request.query_params.get("limit", "100")), 1000))
        except (ValueError, TypeError):
            limit = 100

        all_sessions = adapter.session_store.list_sessions(
            agent=adapter.agent_name,
            source=source,
            status=status,
            parent_id=parent_id,
            limit=limit,
            include_superseded=include_superseded,
        )

        from tsugite_daemon.session_store import SessionStatus

        default_ids = adapter.session_store.default_primary_ids(adapter.agent_name)
        live_statuses = {SessionStatus.RUNNING.value, SessionStatus.ACTIVE.value}

        def _user_label(user_id: str, source: str) -> str:
            if user_id.isdigit():
                return f"Discord: {user_id}"
            if user_id.startswith("web-"):
                return f"Web: {user_id}"
            return user_id or source

        sessions = []
        for s in all_sessions:
            user_id = s.user_id or ""
            label = _user_label(user_id, s.source)
            unread = bool(s.last_active and (not s.last_viewed_at or s.last_active > s.last_viewed_at))
            row = {
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
                "pinned": s.pinned,
                "pin_position": s.pin_position,
                "last_viewed_at": s.last_viewed_at,
                "superseded_by": s.superseded_by,
                "unread": unread,
                "is_primary": s.is_primary,
                # Authoritative busy flag. The UI must render busy state from
                # this, never infer it from cached progress labels.
                "busy": self._session_busy(adapter.agent_name, s),
            }
            if s.status in live_statuses:
                row["progress"] = adapter.session_store.session_progress_summary(s.id)
            sessions.append(row)

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
        title = body.get("title")
        if title is not None and not isinstance(title, str):
            return JSONResponse({"error": "title must be a string"}, status_code=400)

        from tsugite_daemon.session_store import create_interactive_session

        session_id = create_interactive_session(
            adapter.session_store, adapter.agent_name, user_id, title=title, event_bus=self.event_bus
        )
        return JSONResponse({"id": session_id}, status_code=201)

    async def _status(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        user_id = adapter.resolve_http_user(request.query_params.get("user_id", "web-anonymous"))
        session_id = request.query_params.get("session_id")
        session = None
        if session_id:
            try:
                session = adapter.session_store.get_session(session_id)
            except ValueError:
                session = None
        if session is None:
            session = adapter.session_store.find_default_session(user_id, adapter.agent_name)

        model = adapter.resolve_model()
        resolved_model = _resolve_full_model_id(model)
        attachments = [
            {"name": a.name, "content_type": a.content_type.value, "mime_type": a.mime_type}
            for a in adapter._get_all_attachments()
        ]
        if session is not None:
            tokens = session.cumulative_tokens
            message_count = session.message_count
            session_metadata = session.metadata or {}
            chat = self._active_chats.get((adapter.agent_name, user_id, session.id))
            backend = chat.backend if chat else None
        else:
            tokens, message_count, session_metadata, backend = 0, 0, {}, None

        return JSONResponse(
            {
                "model": model,
                "resolved_model": resolved_model if resolved_model != model else None,
                "tokens": tokens,
                "context_limit": (
                    adapter.session_store.get_session_context_limit(session.id)
                    if session
                    else adapter.session_store.get_context_limit(adapter.agent_name)
                ),
                "threshold": (
                    adapter.session_store.get_session_compaction_threshold(session.id)
                    if session
                    else adapter.session_store.get_compaction_threshold(adapter.agent_name)
                ),
                "message_count": message_count,
                "compacting": adapter.session_store.is_compacting(
                    user_id, adapter.agent_name, session_id=session.id if session else None
                ),
                "metadata": session_metadata,
                "busy": bool(session and self._session_busy(adapter.agent_name, session)),
                "pending_message": backend.pending_message if backend else None,
                "attachments": attachments,
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
            # Pin existing sessions to the current model before mutating the
            # agent default so they don't silently switch on their next turn.
            # The default change should only affect sessions created after it.
            adapter.session_store.freeze_session_models_to_current(adapter.agent_name, agent_config.model)
            agent_config.model = new_model

            from tsugite_daemon.memory import DEFAULT_CONTEXT_LIMIT, get_context_limit

            if new_model:
                context_limit = get_context_limit(new_model, fallback=DEFAULT_CONTEXT_LIMIT)
                agent_config.context_limit = context_limit
            else:
                context_limit = DEFAULT_CONTEXT_LIMIT
                agent_config.context_limit = None
            adapter.session_store.update_context_limit(adapter.agent_name, context_limit)

            if self.gateway:
                from tsugite_daemon.config import save_daemon_config

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

    @staticmethod
    def _collect_events(session_id: str, limit: int = 0) -> list[dict]:
        """Read one session's JSONL and return its events as raw dicts in file
        order. ``limit`` trims to the last N ``user_input`` bubbles plus
        whatever follows them.

        Predecessor files are not walked: the new file's leading ``compaction``
        event already carries the canonical pre-compaction summary, so reading
        ancestor files would duplicate context the agent has already received.
        Offline chain traversal is supported via the ``compacted_into`` /
        ``source_session_id`` pointers written into each file at compaction
        time.
        """
        from tsugite.history import get_history_backend

        backend = get_history_backend()
        if not backend.exists(session_id):
            return []
        events: list[dict] = []
        user_input_offsets: list[int] = []
        for event in backend.load(session_id).iter_events():
            if event.type == "user_input":
                user_input_offsets.append(len(events))
            # Same {type, ts, data} shape the raw JSONL lines had (id never exposed).
            events.append(event.model_dump(mode="json", exclude={"id"}, exclude_none=True))

        if limit > 0 and len(user_input_offsets) > limit:
            events = events[user_input_offsets[-limit] :]

        return events

    @staticmethod
    def _resolve_session_id(adapter, user_id: str, request: Request) -> Optional[str]:
        """Use ?session_id= when given, otherwise the user's primary session, otherwise None."""
        session_id = request.query_params.get("session_id")
        if session_id:
            return session_id
        primary = adapter.session_store.find_default_session(user_id, adapter.agent_name)
        return primary.id if primary else None

    async def _history(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        user_id = adapter.resolve_http_user(request.query_params.get("user_id", "web-anonymous"))
        try:
            limit = max(1, min(int(request.query_params.get("limit", "100")), 1000))
        except (ValueError, TypeError):
            limit = 100
        conversation_id = self._resolve_session_id(adapter, user_id, request)
        if conversation_id is None:
            return JSONResponse({"conversation_id": None, "events": []})

        events = self._collect_events(conversation_id, limit=limit)

        # UI events (reactions, prompt_snapshots) are now part of the same
        # session JSONL as conversation events, so they're already included.

        return JSONResponse(
            {
                "conversation_id": conversation_id,
                "events": events,
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

        session_id = self._resolve_session_id(adapter, user_id, request)
        if session_id is None:
            return JSONResponse({"prompt_snapshot": None})
        events = adapter.session_store.read_events(session_id)
        snapshots = [e for e in events if e.get("type") == "prompt_snapshot"]
        breakdown = snapshots[-1].get("token_breakdown", {}) if snapshots else {}

        backend_key = (agent_name, user_id, session_id)
        chat = self._active_chats.get(backend_key)
        live_progress = chat.progress if chat else None
        if live_progress and live_progress.latest_prompt_messages:
            return JSONResponse(
                {
                    "prompt_snapshot": {
                        "messages": live_progress.latest_prompt_messages,
                        "token_breakdown": breakdown,
                    }
                }
            )

        if not snapshots:
            return JSONResponse({"prompt_snapshot": None})
        return JSONResponse({"prompt_snapshot": {"token_breakdown": breakdown}})

    async def _branch(self, request: Request) -> JSONResponse:
        """Fork a session at an event into an independent branch (#400)."""
        adapter, err = self._get_adapter(request)
        if err:
            return err

        session_id = request.path_params["session_id"]
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        at_event_id = body.get("at_event_id")
        if at_event_id is None:
            return JSONResponse({"error": "at_event_id is required"}, status_code=400)

        try:
            branch = adapter.session_store.branch_session(session_id, int(at_event_id), label=body.get("label"))
        except (ValueError, TypeError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)

        self.event_bus.emit("agent_status", {"agent": request.path_params["agent"]})
        return JSONResponse({"session_id": branch.id})

    async def _compact(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        user_id = adapter.resolve_http_user(body.get("user_id", "web-anonymous"))

        session_id = body.get("session_id")
        session = None
        if session_id:
            try:
                session = adapter.session_store.get_session(session_id)
            except ValueError:
                session = None
        if session is None:
            session = adapter.session_store.find_default_session(user_id, adapter.agent_name)

        if session is None or session.message_count == 0:
            return JSONResponse({"error": "no session to compact"}, status_code=404)

        agent_name = request.path_params["agent"]
        old_conv_id = session.id

        if not adapter.session_store.begin_compaction(user_id, adapter.agent_name, session_id=old_conv_id):
            return JSONResponse({"error": "compaction already in progress"}, status_code=409)

        adapter._broadcast_compaction("compaction_started", agent_name, old_conv_id)
        new_session = None
        try:
            instructions = body.get("instructions")
            new_session = await adapter._compact_session(session.id, instructions=instructions, reason="manual")
        except Exception as e:
            msg = str(e) or repr(e)
            logger.exception("Compaction failed for agent %s", adapter.agent_name)
            return JSONResponse({"error": f"compaction failed: {msg}"}, status_code=500)
        finally:
            adapter.session_store.end_compaction(user_id, adapter.agent_name, session_id=old_conv_id)
            adapter._broadcast_compaction("compaction_finished", agent_name, old_conv_id)

        self.event_bus.emit("agent_status", {"agent": agent_name})
        if new_session:
            self.event_bus.emit(
                "session_update",
                {"action": "compacted", "id": old_conv_id, "successor_id": new_session.id},
            )
        return JSONResponse(
            {
                "status": "compacted",
                "old_conversation_id": old_conv_id,
                "new_conversation_id": new_session.id if new_session else old_conv_id,
            }
        )

    async def _unload_skill(self, request: Request) -> JSONResponse:
        """Suppress a skill for the rest of this session's lifetime.

        AgentPreparer will skip the skill on subsequent turns so it does not
        reload from auto_load_skills or a trigger match. In-memory only; a
        daemon restart resets suppression by design.
        """
        adapter, err = self._get_adapter(request)
        if err:
            return err

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        skill_name = body.get("name")
        if not isinstance(skill_name, str) or not skill_name:
            return JSONResponse({"error": "name is required"}, status_code=400)

        user_id = adapter.resolve_http_user(body.get("user_id", "web-anonymous"))
        session = adapter.session_store.find_default_session(user_id, adapter.agent_name)
        if session is None:
            return JSONResponse({"error": "no default session"}, status_code=404)
        adapter.session_store.suppress_skill(session.id, skill_name)

        # Drop it from the currently-loaded manager too so any in-flight code path
        # that still reads the global manager's state sees the removal. Best-effort:
        # if the manager hasn't been initialised yet (no prior `prepare()` call)
        # there's nothing to clear, and the suppression above still takes effect
        # on the next turn.
        from tsugite.tools.skills import get_skill_manager

        try:
            get_skill_manager().unload_skill(skill_name)
        except (AttributeError, RuntimeError):
            logger.debug("unload_skill on global manager skipped for %s", skill_name, exc_info=True)

        return JSONResponse({"status": "ok", "session_id": session.id, "name": skill_name})

    @staticmethod
    def _resolve_session_model(adapter: "HTTPAgentAdapter", session_id: Optional[str]) -> str:
        """Resolve the effective model for a session, honoring a per-session override.

        Falls back to the agent/daemon default (``adapter.resolve_model()``) when
        no session is given or the session has no model override.
        """
        if session_id:
            override = adapter.session_store.get_model_override(session_id)
            if override:
                return override
        return adapter.resolve_model()

    async def _effort_levels(self, request: Request) -> JSONResponse:
        """Return the effort levels supported by the session's resolved model."""
        adapter, err = self._get_adapter(request)
        if err:
            return err

        from tsugite.models import get_model_id, parse_model_string, resolve_model_alias
        from tsugite.providers import get_provider

        model_string = self._resolve_session_model(adapter, request.query_params.get("session_id"))
        levels: list[str] | None = None
        try:
            resolved = resolve_model_alias(model_string)
            provider_name, _, _ = parse_model_string(resolved)
            provider = get_provider(provider_name)
            info = provider.get_model_info(get_model_id(resolved))
            if info and info.supported_effort_levels:
                levels = list(info.supported_effort_levels)
        except (ValueError, Exception):  # noqa: BLE001 — treat any resolution failure as "unknown"
            pass

        return JSONResponse({"model": model_string, "supported_effort_levels": levels})

    def _resolve_session_for_settings(
        self, request: Request
    ) -> tuple[Optional[HTTPAgentAdapter], Optional[str], Optional[JSONResponse]]:
        """Look up (adapter, session_id) for a /api/sessions/{session_id}/settings call."""
        session_id = request.path_params["session_id"]
        for adapter in self.adapters.values():
            try:
                adapter.session_store.get_session(session_id)
                return adapter, session_id, None
            except ValueError:
                continue
        return None, None, JSONResponse({"error": f"unknown session: {session_id}"}, status_code=404)

    def _resolve_effort_or_400(
        self, adapter: "HTTPAgentAdapter", value: Any, session_id: Optional[str] = None
    ) -> tuple[Optional[str], Optional[JSONResponse]]:
        """Validate a reasoning_effort value against the session's resolved model."""
        if value is None:
            return None, None
        from tsugite.models import UnsupportedEffortError, resolve_reasoning_effort

        try:
            return resolve_reasoning_effort(self._resolve_session_model(adapter, session_id), value), None
        except UnsupportedEffortError as err:
            return None, JSONResponse({"error": str(err), "supported": err.supported}, status_code=400)

    def _session_settings_payload(self, adapter: "HTTPAgentAdapter", session_id: str) -> dict:
        session = adapter.session_store.get_session(session_id)
        return {
            "reasoning_effort": adapter.session_store.get_reasoning_effort(session_id),
            "model": adapter.session_store.get_model_override(session_id),
            "agent": session.agent if session else None,
        }

    async def _session_settings_get(self, request: Request) -> JSONResponse:
        adapter, session_id, err = self._resolve_session_for_settings(request)
        if err:
            return err
        return JSONResponse(self._session_settings_payload(adapter, session_id))

    async def _session_settings_patch(self, request: Request) -> JSONResponse:
        adapter, session_id, err = self._resolve_session_for_settings(request)
        if err:
            return err

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        if "reasoning_effort" in body:
            value, err_resp = self._resolve_effort_or_400(adapter, body["reasoning_effort"], session_id)
            if err_resp:
                return err_resp
            adapter.session_store.set_reasoning_effort(session_id, value)

        if "model" in body:
            raw = body["model"]
            if raw is None or raw == "":
                adapter.session_store.set_model_override(session_id, None)
            elif not isinstance(raw, str):
                return JSONResponse({"error": "model must be a string"}, status_code=400)
            else:
                from tsugite.models import get_provider_and_model

                try:
                    get_provider_and_model(raw)
                except Exception as exc:  # noqa: BLE001
                    return JSONResponse({"error": f"unknown model: {raw} ({exc})"}, status_code=400)
                adapter.session_store.set_model_override(session_id, raw)

        if "agent" in body:
            name = body["agent"]
            if not isinstance(name, str) or not name:
                return JSONResponse({"error": "agent must be a non-empty string"}, status_code=400)
            if name not in self.adapters:
                return JSONResponse({"error": f"unknown agent: {name}"}, status_code=400)
            adapter.session_store.set_agent_override(session_id, name)

        return JSONResponse(self._session_settings_payload(adapter, session_id))

    async def _list_models(self, request: Request) -> JSONResponse:
        from tsugite.providers import get_provider, list_all_providers
        from tsugite.providers.model_registry import list_models

        if not getattr(self, "_models_primed", False):
            for name in list_all_providers():
                try:
                    get_provider(name)
                except Exception:  # noqa: BLE001 — provider init may fail without env keys; skip those
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
                    "supports_vision": info.supports_vision,
                    "supports_reasoning": info.supports_reasoning,
                }
            )
        models.sort(key=lambda m: m["id"])
        return JSONResponse({"models": models})

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
        session_id = body.get("session_id")
        if not session_id:
            return JSONResponse({"error": "session_id is required"}, status_code=400)
        logger.info("[%s] respond from user_id=%s session_id=%s", agent_name, user_id, session_id)

        key = (agent_name, user_id, session_id)
        chat = self._active_chats.get(key)
        if not chat:
            return JSONResponse({"error": "no pending question for this session"}, status_code=404)

        chat.backend.submit_response(response)
        return JSONResponse({"status": "ok"})

    async def _upload(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        uploads_dir = adapter.agent_config.workspace_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        # Reject from the header BEFORE request.form() spools the whole
        # multipart body to disk/memory - the per-file check below only fires
        # after the bytes are already materialized.
        try:
            declared = int(request.headers.get("content-length", "0"))
        except ValueError:
            declared = 0
        if declared > MAX_UPLOAD_TOTAL:
            return JSONResponse({"error": "total upload size exceeds 100MB"}, status_code=413)

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

        reasoning_effort, err_resp = self._resolve_effort_or_400(adapter, body.get("reasoning_effort"))
        if err_resp:
            return err_resp

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

        message += _format_upload_message_suffix(workspace_only_files, [a.name for a in uploaded_attachments])

        metadata = {"client_ip": request.client.host if request.client else "unknown"}
        if uploaded_attachments:
            metadata["uploaded_attachments"] = uploaded_attachments
        if reasoning_effort:
            metadata["reasoning_effort_override"] = reasoning_effort

        from tsugite_daemon.session_store import FINISHED_STATUSES

        session_id = body.get("session_id")
        target_session = None
        if session_id:
            try:
                target_session = adapter.session_store.get_session(session_id)
            except ValueError:
                target_session = None
            if target_session is not None and target_session.status in FINISHED_STATUSES:
                successor = adapter.session_store.resolve_compacted_successor(session_id)
                if successor is not None and successor.status not in FINISHED_STATUSES:
                    target_session = successor
                else:
                    return JSONResponse(
                        {
                            "error": f"Session is {target_session.status}. Start a new session to continue.",
                            "code": "session_finished",
                        },
                        status_code=409,
                    )
            if target_session is not None:
                metadata["conv_id_override"] = target_session.id

        if target_session is None:
            target_session = adapter.session_store.get_or_create_interactive(user_id, adapter.agent_name)
        target_session_id = target_session.id

        backend_key = (agent_name, user_id, target_session_id)
        # Same predicate the sessions payload and /status report - the server
        # must not 409 a send while telling the sidebar the session is idle.
        if self._session_busy(agent_name, target_session):
            return JSONResponse(
                {"error": "a turn is already running for this session", "code": "turn_in_flight"},
                status_code=409,
            )

        channel_context = ChannelContext(
            source="http",
            channel_id=None,
            user_id=raw_user_id,
            reply_to=f"http:{raw_user_id}",
            metadata=metadata,
        )

        progress = SSEProgressHandler()
        progress.set_loop(asyncio.get_running_loop())
        progress.set_session_id(target_session_id)
        progress.set_broadcaster(self.event_bus)
        progress.set_event_persister(build_session_event_persister(adapter.session_store, target_session_id))
        custom_logger = SimpleNamespace(ui_handler=progress)

        interaction_backend = HTTPInteractionBackend(progress)
        interaction_backend.pending_message = message
        chat_state = ActiveChat(backend=interaction_backend, progress=progress)
        self._active_chats[backend_key] = chat_state

        async def run_agent():
            from tsugite.cancellation import set_cancel_event
            from tsugite.interaction import set_interaction_backend

            set_interaction_backend(interaction_backend)
            # Bind the cooperative cancel Event into the run context so the agent
            # loop (copy_context + to_thread) observes a user Stop and exits cleanly.
            set_cancel_event(chat_state.cancel_event)
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
                self.event_bus.emit("history_update", {"agent": agent_name, "session_id": target_session_id})

                try:
                    refreshed = adapter.session_store.get_session(target_session_id)
                except ValueError:
                    refreshed = None
                if refreshed is not None:
                    progress._emit(
                        "session_info",
                        {
                            "session_id": target_session_id,
                            "tokens": refreshed.cumulative_tokens,
                            "context_limit": adapter.session_store.get_session_context_limit(target_session_id),
                            "threshold": adapter.session_store.get_session_compaction_threshold(target_session_id),
                            "message_count": refreshed.message_count,
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
                self._active_chats.pop(backend_key, None)
                progress.signal_done()

        task = asyncio.create_task(run_agent())
        chat_state.task = task

        return StreamingResponse(
            progress.event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Session-Id": target_session_id,
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
        session_id = body.get("session_id")
        if not session_id:
            return JSONResponse({"error": "session_id is required"}, status_code=400)
        backend_key = (request.path_params["agent"], user_id, session_id)
        chat = self._active_chats.get(backend_key)
        if chat and chat.task and not chat.task.done():
            # Signal the worker thread to stop at its next safe checkpoint (the real
            # stop - task.cancel alone only tears down the awaiting coroutine/SSE
            # stream, leaving the to_thread worker running to completion).
            chat.cancel_event.set()
            chat.task.cancel()
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
        return JSONResponse({"schedules": [entry_to_dict(e) for e in self.scheduler.list()]})

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
        return JSONResponse(entry_to_dict(entry), status_code=201)

    async def _get_schedule(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_scheduler(request):
            return err
        try:
            entry = self.scheduler.get(request.path_params["schedule_id"])
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        return JSONResponse(entry_to_dict(entry))

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
            "execution_type",
            "command",
            "script_timeout",
            "target_session",
        }
        fields = {k: v for k, v in body.items() if k in allowed}
        if not fields:
            return JSONResponse({"error": "no updatable fields provided"}, status_code=400)

        try:
            entry = self.scheduler.update(schedule_id, **fields)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        self.event_bus.emit("schedule_update", {"action": "updated", "id": schedule_id})
        return JSONResponse(entry_to_dict(entry))

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
                        "is_primary": s.is_primary,
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

        from tsugite_daemon.session_store import Session, SessionSource

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
        pinned = body.get("pinned")
        has_pin_position = "pin_position" in body
        pin_position = body.get("pin_position")
        last_viewed_at = body.get("last_viewed_at")
        if title is None and status is None and pinned is None and not has_pin_position and last_viewed_at is None:
            return JSONResponse({"error": "No updatable fields provided"}, status_code=400)
        runner = self.session_runner
        try:
            result = {}
            if title is not None:
                runner.rename_session(session_id, title)
                result["title"] = title
            if pinned is not None or has_pin_position:
                # When only pin_position is sent, keep the existing pinned state but route
                # through set_pin so siblings rebalance instead of leaving gaps.
                target_pinned = bool(pinned) if pinned is not None else runner.store.get_session(session_id).pinned
                session = runner.set_pin(session_id, target_pinned, position=pin_position if has_pin_position else None)
                result["pinned"] = session.pinned
                result["pin_position"] = session.pin_position
            if last_viewed_at is not None:
                session = runner.mark_viewed(session_id, ts=last_viewed_at or None)
                result["last_viewed_at"] = session.last_viewed_at
            if status is not None:
                if status != "completed":
                    return JSONResponse({"error": "Only 'completed' status is allowed"}, status_code=400)
                runner.store.update_session(session_id, status=status)
                self.event_bus.emit("session_update", {"action": "completed", "id": session_id})
                result["status"] = status
            return JSONResponse({"ok": True, **result})
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)

    @staticmethod
    async def _optional_json_body(request: Request) -> dict:
        """Best-effort JSON body parse; missing/invalid bodies become {}."""
        try:
            body = await request.json()
        except Exception:
            return {}
        return body if isinstance(body, dict) else {}

    async def _api_pin_session(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        body = await self._optional_json_body(request)
        try:
            session = self.session_runner.set_pin(session_id, True, position=body.get("position"))
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        return JSONResponse({"ok": True, "pinned": session.pinned, "pin_position": session.pin_position})

    async def _api_unpin_session(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        try:
            self.session_runner.set_pin(session_id, False)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        return JSONResponse({"ok": True})

    async def _api_reorder_pins(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        ids = body.get("ids") if isinstance(body, dict) else None
        if not isinstance(ids, list) or not all(isinstance(s, str) for s in ids):
            return JSONResponse({"error": "ids must be a list of strings"}, status_code=400)
        ordered = self.session_runner.reorder_pins(ids)
        return JSONResponse({"ok": True, "ordered": [s.id for s in ordered]})

    async def _api_set_primary(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        try:
            self.session_runner.set_primary_session(session_id)
        except ValueError as e:
            msg = str(e)
            status = 404 if "not found" in msg else 400
            return JSONResponse({"error": msg}, status_code=status)
        return JSONResponse({"ok": True, "id": session_id, "is_primary": True})

    async def _api_clear_primary(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        agent = request.query_params.get("agent")
        user_id = request.query_params.get("user_id")
        if not agent or not user_id:
            return JSONResponse({"error": "agent and user_id query params required"}, status_code=400)
        self.session_runner.clear_primary_session(user_id, agent)
        return JSONResponse({"ok": True})

    async def _api_mark_viewed(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        body = await self._optional_json_body(request)
        try:
            session = self.session_runner.mark_viewed(session_id, ts=body.get("ts"))
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        return JSONResponse({"ok": True, "last_viewed_at": session.last_viewed_at})

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

    # ── Jobs tile actions ──

    def _require_auth_and_jobs(self, request: Request) -> Optional[JSONResponse]:
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        if not self.jobs_orchestrator:
            return JSONResponse({"error": "jobs orchestrator not available"}, status_code=503)
        return None

    async def _api_list_jobs(self, request: Request) -> JSONResponse:
        """Return Jobs for the Jobs tab, newest first. Optional ?state=<state>
        filter accepts a real Job state, the alias 'stuck' (= stuck + errored),
        or 'active' (= running + verifying). ?limit=N caps the response
        (default 100; 0 = unlimited)."""
        if err := self._check_auth(request):
            return err
        if not self.job_store:
            return JSONResponse({"error": "jobs orchestrator not available"}, status_code=503)
        jobs = self.job_store.list_all()
        state_filter = request.query_params.get("state")
        if state_filter:
            alias = {
                "stuck": frozenset({"stuck", "errored"}),
                "active": frozenset({"running", "verifying"}),
                "resolved": frozenset({"done", "cancelled"}),
            }
            allowed = alias.get(state_filter, frozenset({state_filter}))
            jobs = [j for j in jobs if j.state in allowed]
        try:
            limit = int(request.query_params.get("limit", "100"))
        except ValueError:
            limit = 100
        if limit > 0:
            jobs = jobs[:limit]
        return JSONResponse({"jobs": [j.to_payload() for j in jobs]})

    async def _api_cancel_job(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_jobs(request):
            return err
        job_id = request.path_params["job_id"]
        body = await self._optional_json_body(request)
        reason = body.get("reason") or "cancelled by user"
        try:
            await self.jobs_orchestrator.cancel_job(job_id, reason=reason)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        return JSONResponse({"status": "cancelled"})

    async def _api_mark_job_done(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_jobs(request):
            return err
        job_id = request.path_params["job_id"]
        body = await self._optional_json_body(request)
        reason = body.get("reason") or "marked done by user"
        try:
            await self.jobs_orchestrator.mark_done_manual(job_id, reason=reason)
        except ValueError as e:
            # 404 if unknown, 409 if not in STUCK state.
            status = 404 if "Unknown job" in str(e) else 409
            return JSONResponse({"error": str(e)}, status_code=status)
        return JSONResponse({"status": "done"})

    async def _api_retry_job(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_jobs(request):
            return err
        job_id = request.path_params["job_id"]
        body = await self._optional_json_body(request)
        hint = (body.get("hint") or "").strip()
        if not hint:
            return JSONResponse({"error": "hint is required"}, status_code=400)
        reset_counter = bool(body.get("reset_counter", False))
        fresh_workspace = bool(body.get("fresh_workspace", False))
        try:
            await self.jobs_orchestrator.retry_with_hint(
                job_id,
                hint=hint,
                reset_counter=reset_counter,
                fresh_workspace=fresh_workspace,
            )
        except ValueError as e:
            status = 404 if "Unknown job" in str(e) else 409
            return JSONResponse({"error": str(e)}, status_code=status)
        return JSONResponse({"status": "running"})

    # ── Terminal viewer API ──

    def _require_auth_and_terminals(self, request: Request) -> Optional[JSONResponse]:
        """Check auth and terminal subsystem availability."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        if self.terminal_store is None or self.pty_manager is None:
            return JSONResponse({"error": "terminal viewer not available"}, status_code=503)
        return None

    def _terminal_to_dict(self, terminal) -> dict:
        from dataclasses import asdict

        proc = self.pty_manager.get(terminal.id) if self.pty_manager else None
        data = asdict(terminal)
        # Surface live runtime info even before the on_exit hook persists the
        # final counts. Stale-on-disk wins for terminated terminals (proc dropped).
        if proc is not None:
            data["bytes_out"] = max(data.get("bytes_out", 0), proc.bytes_out)
            data["lines_out"] = max(data.get("lines_out", 0), proc.lines_out)
            data["truncated"] = proc.truncated
            if proc.last_line:
                data["last_line"] = proc.last_line
        else:
            data.setdefault("truncated", False)
        return data

    def _emit_terminal_state(self, terminal_id: str, new_state: str) -> None:
        """on_state_change callback for spawn_terminal: broadcast a state change."""
        self.event_bus.emit("terminal_state", {"terminal_id": terminal_id, "state": new_state})

    async def _api_list_terminals(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_terminals(request):
            return err
        parent = request.query_params.get("parent_session_id")
        if parent:
            terminals = self.terminal_store.list_for_parent(parent)
        else:
            terminals = self.terminal_store.list_all()
        return JSONResponse({"terminals": [self._terminal_to_dict(t) for t in terminals]})

    async def _api_get_terminal(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_terminals(request):
            return err
        terminal_id = request.path_params["terminal_id"]
        terminal = self.terminal_store.get(terminal_id)
        if terminal is None:
            return JSONResponse({"error": f"unknown terminal: {terminal_id}"}, status_code=404)
        return JSONResponse(self._terminal_to_dict(terminal))

    async def _api_create_terminal(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_terminals(request):
            return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        cmd = (body.get("cmd") or "").strip()
        if not cmd:
            return JSONResponse({"error": "cmd is required"}, status_code=400)
        cwd = body.get("cwd")
        parent_session_id = body.get("parent_session_id")
        env = body.get("env") if isinstance(body.get("env"), dict) else None

        from tsugite_pty.terminal_runtime import spawn_terminal

        try:
            terminal = spawn_terminal(
                store=self.terminal_store,
                manager=self.pty_manager,
                cmd=cmd,
                cwd=cwd,
                env=env,
                parent_session_id=parent_session_id,
                on_state_change=self._emit_terminal_state,
            )
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception as e:
            logger.exception("Failed to spawn terminal")
            return JSONResponse({"error": str(e)}, status_code=500)
        return JSONResponse(self._terminal_to_dict(terminal), status_code=201)

    async def _api_kill_terminal(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_terminals(request):
            return err
        terminal_id = request.path_params["terminal_id"]
        terminal = self.terminal_store.get(terminal_id)
        if terminal is None:
            return JSONResponse({"error": f"unknown terminal: {terminal_id}"}, status_code=404)
        self.pty_manager.kill(terminal_id)
        return JSONResponse({"status": "killed", "terminal_id": terminal_id})

    async def _api_terminal_stdin(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_terminals(request):
            return err
        terminal_id = request.path_params["terminal_id"]
        terminal = self.terminal_store.get(terminal_id)
        if terminal is None:
            return JSONResponse({"error": f"unknown terminal: {terminal_id}"}, status_code=404)
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        data = body.get("data", "")
        if not isinstance(data, str):
            return JSONResponse({"error": "data must be a string"}, status_code=400)
        written = self.pty_manager.write_stdin(terminal_id, data.encode("utf-8", errors="replace"))
        return JSONResponse({"status": "ok", "bytes_written": written})

    async def _api_restart_terminal(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_terminals(request):
            return err
        terminal_id = request.path_params["terminal_id"]
        old = self.terminal_store.get(terminal_id)
        if old is None:
            return JSONResponse({"error": f"unknown terminal: {terminal_id}"}, status_code=404)

        from tsugite_pty.terminal_runtime import spawn_terminal
        from tsugite_pty.terminal_store import TerminalState

        # Refuse to restart a still-live PTY - caller should kill first to avoid
        # leaking the original process.
        if old.state not in (
            TerminalState.SUCCEEDED.value,
            TerminalState.FAILED.value,
            TerminalState.CANCELLED.value,
            TerminalState.STREAM_LOST.value,
        ):
            return JSONResponse(
                {"error": f"cannot restart terminal in '{old.state}' state; kill it first"},
                status_code=409,
            )

        try:
            new_terminal = spawn_terminal(
                store=self.terminal_store,
                manager=self.pty_manager,
                cmd=old.cmd,
                cwd=old.cwd,
                parent_session_id=old.parent_session_id,
                on_state_change=self._emit_terminal_state,
            )
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception as e:
            logger.exception("Failed to restart terminal")
            return JSONResponse({"error": str(e)}, status_code=500)

        return JSONResponse(
            {**self._terminal_to_dict(new_terminal), "restarted_from": terminal_id},
            status_code=201,
        )

    async def _api_terminal_stream(self, request: Request) -> Response:
        if err := self._require_auth_and_terminals(request):
            return err
        terminal_id = request.path_params["terminal_id"]
        terminal = self.terminal_store.get(terminal_id)
        if terminal is None:
            return JSONResponse({"error": f"unknown terminal: {terminal_id}"}, status_code=404)

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=512)
        closing = asyncio.Event()

        def _safe_put(payload) -> None:
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                # Slow client; drop the chunk rather than backing up the daemon.
                pass

        def _push(payload: dict) -> None:
            if threading.current_thread() is threading.main_thread() and loop.is_running():
                try:
                    running = asyncio.get_running_loop()
                except RuntimeError:
                    running = None
                if running is loop:
                    _safe_put(payload)
                    return
            try:
                loop.call_soon_threadsafe(_safe_put, payload)
            except RuntimeError:
                pass

        def _close() -> None:
            # Guarantee stream teardown: enqueue the sentinel (best-effort under
            # backpressure) AND set `closing` so the generator breaks even if the
            # sentinel was dropped from a full queue.
            try:
                loop.call_soon_threadsafe(_safe_put, None)
                loop.call_soon_threadsafe(closing.set)
            except RuntimeError:
                pass

        # Emit the current state up front so a late-connecting client doesn't
        # need a separate fetch to know whether the terminal is still running.
        # Pushed before any output/exit events so consumers can size their
        # rendering up front.
        _push({"type": "state", "state": terminal.state})

        proc = self.pty_manager.get(terminal_id)
        unsub = None
        exit_unsub = None
        if proc is not None:

            def _on_chunk(chunk: bytes) -> None:
                _push({"type": "output", "chunk": chunk.decode("utf-8", errors="replace")})

            def _on_exit(p) -> None:
                _push({"type": "exit", "exit_code": p.exit_code})
                _close()

            # Snapshot + subscribe atomically so a chunk produced between the two
            # is never lost. The buffer is the ring-capped window (1 MB default);
            # anything older is gone, hence the `truncated` flag the frontend uses.
            existing, unsub = proc.snapshot_and_subscribe(_on_chunk)
            if existing:
                _push({"type": "output", "chunk": existing.decode("utf-8", errors="replace"), "replay": True})
            # on_exit fires synchronously if the process has already exited, which
            # is fine - it just queues the exit event after what we already pushed.
            exit_unsub = proc.on_exit(_on_exit)
        else:
            # No live PTY (already evicted post-exit, or pre-spawn failure).
            # Try the on-disk log first so re-opening an old terminal still
            # shows what it printed. terminal_runtime writes this when the
            # process exits with a non-empty buffer.
            current = self.terminal_store.get(terminal_id)
            log_path = self.terminal_store.log_path(terminal_id)
            try:
                if log_path.is_file():
                    contents = log_path.read_bytes()
                    if contents:
                        _push(
                            {
                                "type": "output",
                                "chunk": contents.decode("utf-8", errors="replace"),
                                "replay": True,
                            }
                        )
            except OSError:
                logger.exception("Failed to read terminal log for '%s'", terminal_id)
            _push({"type": "exit", "exit_code": current.exit_code})
            _close()

        async def generator():
            try:
                while True:
                    try:
                        payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        # If close fired but its sentinel was dropped under
                        # backpressure, `closing` still tears the stream down.
                        if closing.is_set() and queue.empty():
                            break
                        yield ": keepalive\n\n"
                        continue
                    if payload is None:
                        break
                    event_type = payload.pop("type", "message")
                    yield f"event: {event_type}\n"
                    yield f"data: {json.dumps(payload)}\n\n"
            finally:
                for fn in (unsub, exit_unsub):
                    if fn is not None:
                        try:
                            fn()
                        except Exception:
                            pass

        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    async def _api_restart_session(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        try:
            old = self.session_runner.store.get_session(session_id)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)

        from tsugite_daemon.session_store import Session, SessionSource, SessionStatus

        restartable = {SessionStatus.FAILED.value, SessionStatus.CANCELLED.value}
        if old.status not in restartable:
            return JSONResponse({"error": f"cannot restart session in '{old.status}' state"}, status_code=400)

        # Carry forward per-session overrides + metadata so a restarted worker
        # session still runs in its provisioned worktree AND the JobsOrchestrator
        # still recognises it via metadata.job_id.
        new_session = Session(
            id="",
            agent=old.agent,
            source=old.source or SessionSource.BACKGROUND.value,
            prompt=old.prompt,
            model=old.model,
            agent_file=old.agent_file,
            notify=old.notify,
            workspace_override=old.workspace_override,
            metadata=dict(old.metadata or {}),
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
        # The source lands in inbox filenames - restrict it to a safe slug so a
        # slash/.. can't escape (or break) the inbox directory on delivery.
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", source):
            return JSONResponse(
                {"error": "source must be 1-64 chars of letters, digits, dot, underscore, or dash"},
                status_code=400,
            )
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
        if len(raw) > MAX_WEBHOOK_BODY:
            return JSONResponse({"error": "payload too large"}, status_code=413)

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
        # Sanitize defensively (pre-validation legacy records may carry anything)
        # and suffix a nonce: bursty sources deliver multiple events in the same
        # second, and write_text would silently clobber the earlier file.
        safe_source = re.sub(r"[^A-Za-z0-9._-]", "_", webhook.source)[:64] or "webhook"
        filename = f"{now.strftime('%Y%m%dT%H%M%S')}-{safe_source}-{uuid4().hex[:8]}.json"
        envelope = {
            "source": webhook.source,
            "agent": webhook.agent,
            "received_at": now.isoformat(),
            "headers": {k: v for k, v in request.headers.items() if k.lower() != "authorization"},
            "payload": payload_data,
        }
        # Off the event loop: a slow disk write here would stall every SSE
        # stream and request in the daemon.
        await asyncio.to_thread((inbox_dir / filename).write_text, json.dumps(envelope, indent=2, default=str))
        logger.info("Webhook [%s] saved to inbox: %s", token[:8], filename)

        return JSONResponse({"status": "accepted", "file": filename}, status_code=202)

    def _get_allowed_agent_dirs(self) -> list[tuple[Path, str, bool]]:
        """Return (directory, source_label, is_readonly) for all agent directories.

        Routes through `iter_agent_search_paths` so the search order + dedup
        logic match every other site (find_agent_file, repl_completer, etc.).
        Workspace agent dirs come from configured agents and feed in as
        extra_project_dirs.
        """
        extra_project_dirs: list[Path] = []
        for cfg in self.agent_configs.values():
            extra_project_dirs.extend([cfg.workspace_dir / ".tsugite", cfg.workspace_dir / "agents"])
        return [
            (entry.path, entry.source.value, entry.readonly)
            for entry in iter_agent_search_paths(extra_project_dirs=extra_project_dirs)
        ]

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

    async def _list_skill_issues(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        from tsugite.tools.skills import get_failed_skills_list

        return JSONResponse({"issues": get_failed_skills_list()})

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
        # Prepend a version constant derived from the web dir's latest mtime so the
        # browser treats the SW as updated whenever any web asset changes.
        version = _web_assets_version()
        body = f'const SW_VERSION = "{version}";\n'.encode() + sw_path.read_bytes()
        return Response(body, media_type="application/javascript", headers={"Cache-Control": "no-cache"})

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
