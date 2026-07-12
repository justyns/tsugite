"""Module-level helpers, constants, and small classes shared by the HTTP handlers."""

import asyncio
import logging
import mimetypes
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from tsugite.attachments.base import AttachmentContentType
from tsugite.attachments.file import FileHandler
from tsugite_daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite_daemon.adapters.http.sse import HTTPInteractionBackend, SSEProgressHandler

if TYPE_CHECKING:
    from tsugite_daemon.session_store import SessionStore

# adapters/http/helpers.py -> up three parents reaches the daemon package root.
WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"


def mounted_api_routes(prefix: str, name: str, routes: list) -> list:
    """A clean-prefix API collection as a Mount PLUS a bare-prefix Route for the
    collection root, so both `/api/jobs` and `/api/jobs/` resolve directly.

    A `Mount("/api/jobs", ...)` only matches sub-paths starting with "/", so the
    bare `/api/jobs` (no slash) never reaches it: Starlette's top-level router
    307-redirects it to the trailing-slash form and builds the Location as an
    absolute URL from the daemon's own bind address (http://127.0.0.1:8374/...).
    Behind a reverse proxy / non-localhost origin the browser can't follow that
    (unreachable host, cross-origin, https->http), so the fetch fails. Adding a
    top-level Route at the bare prefix for each root ("/") handler removes the
    redirect for every collection at once.
    """
    extra: list = []
    for r in routes:
        if isinstance(r, Route) and r.path == "/":
            methods = sorted((r.methods or set()) - {"HEAD", "OPTIONS"})
            extra.append(Route(prefix, r.endpoint, methods=methods, name=r.name))
    return [Mount(prefix, name=name, routes=routes), *extra]


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


MAX_TEXT_ATTACH_SIZE = 50 * 1024  # 50KB -- ~12K tokens
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
