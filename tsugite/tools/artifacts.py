"""Open a document in a pane beside the chat.

Daemon-gated, so it runs in the daemon's parent process (the subprocess executor
proxies `require_daemon` tools out of the sandbox). The path validation here is
the server-side gate. The daemon serves nothing the browser could not already
read through `GET /api/workspace/content` for the same session.
"""

import re
import uuid
from pathlib import Path
from typing import Optional

from ..cli.helpers import get_workspace_dir
from ..exceptions import ToolUnavailableError
from . import tool
from .sessions import get_current_session_id

# Any RFC 3986 scheme prefix (`https:`, `javascript:`, `data:`, ...), not just
# `://`, so a URL that omits the double slash gets the same error.
_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")

# `event_type` of the SSE frame. The frontend reducer keys off it.
ARTIFACT_EVENT = "artifact_open"

# The one reusable pane slot. `replace_existing=True` (the default) opens into it.
AGENT_ARTIFACT_ID = "agent"

# Ephemeral content travels inline in the SSE frame, so it is capped well below
# anything that would stall the event stream. Bigger output belongs in a file.
MAX_INLINE_CONTENT = 128_000

# An upper bound on what the tool will point the browser at. The authoritative
# limit is `max_workspace_file_size` on the endpoint the pane reads through,
# which is configurable and lower by default.
MAX_ARTIFACT_FILE_BYTES = 4 * 1024 * 1024

CONTENT_TYPES = ("markdown", "html", "text")
MODES = ("rendered", "source")
PLACEMENTS = ("right", "below")

_adapter = None
_event_bus = None


def set_artifact_bridge(adapter, event_bus) -> None:
    """Called by the daemon gateway to set/clear the workspace + broadcaster."""
    global _adapter, _event_bus
    _adapter = adapter
    _event_bus = event_bus


def _infer_content_type(name: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix in (".md", ".markdown"):
        return "markdown"
    if suffix in (".html", ".htm"):
        return "html"
    return "text"


def _resolve_in_workspace(path: str) -> str:
    """Resolve `path` inside the session workspace and return it workspace-relative, or raise."""
    if path.startswith("//") or _URI_SCHEME.match(path):
        raise PermissionError(f"Invalid path '{path}': open_artifact does not fetch an external URL")

    # get_workspace_dir() is the running session's directory, which a job worker
    # overrides to its provisioned worktree. Every adapter shares the one runtime
    # workspace.
    workspace_dir = Path(get_workspace_dir() or _adapter.runtime.workspace_dir).resolve()
    try:
        resolved = (workspace_dir / path).resolve()
    except (ValueError, OSError) as e:
        raise ValueError(f"Invalid path '{path}': {e}") from e
    if not resolved.is_relative_to(workspace_dir):
        raise PermissionError(f"Invalid path '{path}': outside the workspace ({workspace_dir})")
    if not resolved.exists():
        raise ValueError(f"Invalid path '{path}': does not exist in the workspace")
    if not resolved.is_file():
        raise ValueError(f"Invalid path '{path}': not a file")
    size = resolved.stat().st_size
    if size > MAX_ARTIFACT_FILE_BYTES:
        raise ValueError(f"Invalid path '{path}': too large to open ({size} bytes)")
    return str(resolved.relative_to(workspace_dir))


@tool(require_daemon=True, category="artifacts")
def open_artifact(
    path: Optional[str] = None,
    content: Optional[str] = None,
    content_type: Optional[str] = None,
    mode: Optional[str] = None,
    title: Optional[str] = None,
    placement: str = "right",
    replace_existing: bool = True,
) -> str:
    """Open a workspace file (or generated text) in a pane beside the user's chat.

    Use this when the user should READ something while the conversation keeps
    going: a report you just generated, a file you are about to change, a diff, a
    coverage page. The chat stays visible and usable, and the pane sits next to it
    so the user can resize or close it. Put the long artifact in the pane so your
    reply can stay short.

    By default every call updates the SAME pane, so opening five documents over a
    turn leaves one pane showing the latest, not five stacked tabs. Pass
    `replace_existing=False` only when the user needs two open at once.

    Only files inside the session's workspace can be opened; the path is validated
    daemon-side and `..`, absolute paths, symlinks out of the tree, and URLs are
    all refused. HTML renders inside a sandboxed frame that cannot run scripts,
    reach the network, or see the app's session - so an HTML artifact is safe to
    show but will not behave like a live web page.

    Args:
        path: Workspace-relative file to open (e.g. "reports/coverage.html").
            Mutually exclusive with `content`.
        content: Generated text to show instead of a file, for output that has no
            home on disk. Capped at 128KB - write a file and pass `path` for more.
        content_type: "markdown", "html", or "text". Inferred from the file
            extension when `path` is given; defaults to "text" for `content`.
        mode: "rendered" (default for markdown/HTML) or "source" to show the raw
            text. Plain text is always source - there is nothing to render.
        title: Pane title. Defaults to the file name, or "Artifact".
        placement: "right" (default) to split beside the chat, or "below".
            Ignored when an artifact pane is already open - that one is reused.
        replace_existing: Reuse the one agent artifact pane (default). False opens
            an additional pane, which the user then has to close by hand.

    Returns:
        A short confirmation of what was opened.

    Raises:
        ValueError: Bad arguments.
        PermissionError: A path outside the workspace, or a URL.
        ToolUnavailableError: No daemon is running this agent (no UI to open a pane in).

    Example:
        open_artifact(path="reports/coverage.html", title="Coverage")
        open_artifact(content=summary_md, content_type="markdown", title="Summary")
    """
    if _adapter is None or _event_bus is None:
        raise ToolUnavailableError("open_artifact is not available: no daemon is running this agent.")

    if path and content is not None:
        raise ValueError("Invalid arguments: pass path or content, not both")
    if not path and content is None:
        raise ValueError("Invalid arguments: pass either path (a workspace file) or content (generated text)")
    if content_type is not None and content_type not in CONTENT_TYPES:
        raise ValueError(f"Invalid content_type '{content_type}': must be one of {', '.join(CONTENT_TYPES)}")
    if mode is not None and mode not in MODES:
        raise ValueError(f"Invalid mode '{mode}': must be one of {', '.join(MODES)}")
    if placement not in PLACEMENTS:
        raise ValueError(f"Invalid placement '{placement}': must be one of {', '.join(PLACEMENTS)}")

    rel_path = None
    if path:
        rel_path = _resolve_in_workspace(path)
        resolved_type = content_type or _infer_content_type(rel_path)
        default_title = Path(rel_path).name
    else:
        if len(content) > MAX_INLINE_CONTENT:
            raise ValueError(
                f"Invalid content: too large ({len(content)} chars, max {MAX_INLINE_CONTENT}). "
                "Write it to a workspace file and pass path= instead."
            )
        resolved_type = content_type or "text"
        default_title = "Artifact"

    resolved_mode = "source" if resolved_type == "text" else (mode or "rendered")

    payload = {
        "session_id": get_current_session_id(),
        "event_type": ARTIFACT_EVENT,
        "artifact_id": AGENT_ARTIFACT_ID if replace_existing else uuid.uuid4().hex[:12],
        "path": rel_path,
        "content": None if rel_path else content,
        "content_type": resolved_type,
        "mode": resolved_mode,
        "title": title or default_title,
        "placement": placement,
        # Drives the pane's "opened by the agent" badge.
        "opened_by": "agent",
    }
    _event_bus.emit("session_event", payload)

    where = "beside the chat" if placement == "right" else "below the chat"
    return f"Opened {payload['title']} {where}."
