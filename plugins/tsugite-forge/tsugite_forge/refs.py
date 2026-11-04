"""The repos and pull requests a session works with.

Pinned entries and hide markers are stored in the session's `refs` metadata.
Touched and mentioned entries are derived from history on every read.
"""

import os
import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

from tsugite.history import get_history_backend

SOURCES = ("pinned", "touched", "mentioned")
ACTIONS = ("pin", "unpin", "hide", "unhide")
GITHUB_HOST = "github.com"
# Bounds the history read per request, and stops a compaction chain that loops.
MAX_CHAIN = 10
GIT_TIMEOUT = 5
PATH_TOOLS = {"read_file": "path", "write_file": "path", "edit_file": "path", "list_files": "path", "spawn_job": "repo"}
PR_TOOLS = {"forge_pr": "number", "forge_pin": "pr"}
HISTORY_TYPES = ["session_start", "code_execution", "user_input", "model_response"]

URL = re.compile(r"https?://[^\s<>()\"'`]+")
SCP_REMOTE = re.compile(r"^[\w.-]+@([\w.-]+):(.+)$")
BARE_NUMBER = re.compile(r"(?<![\w/#&])#(\d+)\b")
LINK_KINDS = {"forgejo": ("pulls", "issues"), "github": ("pull", "issues")}


def forge_host(forge) -> str:
    return GITHUB_HOST if forge.kind == "github" else urlsplit(forge.url).hostname


def _forge_for(host: str | None, forges):
    return next((forge for forge in forges if host and host.lower() == forge_host(forge)), None)


def remote_ref(url: str, forges) -> tuple[str, str] | None:
    """(forge, owner/repo) for a git remote URL on a configured forge, or None."""
    scp = SCP_REMOTE.match(url)
    if scp:
        host, path = scp.groups()
    else:
        parts = urlsplit(url)
        if parts.scheme not in ("ssh", "http", "https", "git"):
            return None
        host, path = parts.hostname, parts.path
    forge = _forge_for(host, forges)
    segments = [s for s in path.removesuffix("/").removesuffix(".git").split("/") if s]
    if forge is None or len(segments) < 2:
        return None
    # Forgejo can be served under a path prefix, which taking the last two segments skips.
    return forge.name, "/".join(segments[-2:])


def link_ref(url: str, forges) -> tuple[str, str, str, int] | None:
    """(forge, owner/repo, kind, number) for a pull request or issue URL on a configured forge, or None.
    `kind` is the URL path segment."""
    parts = urlsplit(url)
    forge = _forge_for(parts.hostname, forges)
    segments = parts.path.strip("/").split("/")
    if forge is None or len(segments) < 4 or not segments[-1].isdigit():
        return None
    owner, repo, kind, number = segments[-4:]
    if kind not in LINK_KINDS[forge.kind]:
        return None
    return forge.name, f"{owner}/{repo}", kind, int(number)


def mention_refs(text: str, forges, bare_repo: tuple[str, str] | None) -> list[tuple[str, str, int]]:
    """(forge, owner/repo, number) for each pull request or issue URL on a configured forge.

    A bare `#n` counts only against `bare_repo`, the repo every non-hidden pin is in.
    """
    found = []
    for match in URL.finditer(text):
        ref = link_ref(match.group().rstrip(".,;:!?"), forges)
        if ref:
            forge, repo, _kind, number = ref
            found.append((forge, repo, number))
    if bare_repo:
        found += [(*bare_repo, int(number)) for number in BARE_NUMBER.findall(text)]
    return found


def _valid(entry) -> bool:
    return (
        isinstance(entry, dict)
        and isinstance(entry.get("forge"), str)
        and isinstance(entry.get("repo"), str)
        and (entry.get("pr") is None or (isinstance(entry["pr"], int) and not isinstance(entry["pr"], bool)))
        and (entry.get("source") == "pinned" or entry.get("hidden") is True)
    )


def _key(entry: dict) -> tuple:
    return entry["forge"], entry["repo"], entry.get("pr")


def stored_refs(value) -> list[dict]:
    """The well-formed pinned entries and hide markers in a `refs` metadata value."""
    if not isinstance(value, list):
        return []
    refs = []
    for entry in filter(_valid, value):
        mark = {"hidden": True} if entry.get("hidden") is True else {"source": "pinned"}
        refs.append({"forge": entry["forge"], "repo": entry["repo"], "pr": entry.get("pr")} | mark)
    return refs


def apply_action(refs: list[dict], action: str, forge: str, repo: str, pr: int | None) -> list[dict]:
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}, choose from {', '.join(ACTIONS)}")
    hidden = action in ("hide", "unhide")
    key = (forge, repo, pr)
    kept = [entry for entry in refs if not (_key(entry) == key and bool(entry.get("hidden")) == hidden)]
    if action == "pin":
        kept.append({"forge": forge, "repo": repo, "pr": pr, "source": "pinned"})
    elif action == "hide":
        kept.append({"forge": forge, "repo": repo, "pr": pr, "hidden": True})
    return kept


def session_chain(session_id: str) -> list[str]:
    """`session_id` and the sessions it was compacted from, oldest first."""
    backend = get_history_backend()
    chain, sid = [], session_id
    while sid and len(chain) < MAX_CHAIN and backend.exists(sid):
        chain.append(sid)
        compaction = next(backend.load(sid).iter_events(types=["compaction"]), None)
        sid = compaction.data.get("source_session_id") if compaction else None
    return chain[::-1]


def history_key(chain: list[str]) -> tuple:
    """Changes whenever an event is recorded in any session of `chain`."""
    backend = get_history_backend()
    return tuple((sid, backend.count_events(sid)) for sid in chain)


def load_histories(chain: list[str]) -> list[list]:
    backend = get_history_backend()
    return [list(backend.load(sid).iter_events(types=HISTORY_TYPES)) for sid in chain]


def git_origin(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "remote", "get-url", "origin"], capture_output=True, text=True, timeout=GIT_TIMEOUT
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _path(raw, workspace: str | None) -> Path | None:
    # Recorded string arguments over 500 characters are cut and end in "…".
    if not isinstance(raw, str) or not raw or raw.endswith("…"):
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        if not workspace:
            return None
        path = Path(workspace) / path
    # Without this, `ws/../x` would list `ws` among its parents.
    return Path(os.path.normpath(path))


def _number(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return int(value) if isinstance(value, str) and value.isdigit() else None


def _call_refs(call: dict, workspace: str | None, forges, resolve, origin):
    args = call.get("arguments")
    if not isinstance(args, dict):
        return
    tool = call.get("tool")
    if tool in PR_TOOLS:
        ref = resolve(args["repo"]) if isinstance(args.get("repo"), str) else None
        if ref:
            yield (*ref, _number(args.get(PR_TOOLS[tool])))
        return
    if tool not in PATH_TOOLS:
        return
    path = _path(args.get(PATH_TOOLS[tool]), workspace)
    root = next((p for p in [path, *path.parents] if (p / ".git").exists()), None) if path else None
    url = origin(root) if root else None
    ref = remote_ref(url, forges) if url else None
    if ref:
        yield (*ref, None)


def derive(chain: list[list], forges, resolve, bare_repo, origin) -> list[tuple[str, tuple]]:
    """(source, (forge, repo, pr)) for every touched and mentioned ref in `chain`, in order.

    `resolve` maps a forge tool's `repo` argument to (forge, repo), or None. `origin` maps a
    checkout root to its origin URL.
    """
    found = []
    for events in chain:
        workspace = None
        for event in events:
            data = event.data
            if event.type == "session_start" and data.get("workspace"):
                workspace = data["workspace"]
            elif event.type == "code_execution":
                for call in data.get("tool_calls") or []:
                    found += [("touched", ref) for ref in _call_refs(call, workspace, forges, resolve, origin)]
            elif event.type in ("user_input", "model_response"):
                text = data.get("text") if event.type == "user_input" else data.get("raw_content")
                if isinstance(text, str):
                    found += [("mentioned", ref) for ref in mention_refs(text, forges, bare_repo)]
    return found


def merge(stored: list[dict], derived: list[tuple[str, tuple]], pulls: dict[tuple[str, str, int], dict]) -> list[dict]:
    """One row per (forge, repo, pr), with its most authoritative source, pinned rows first."""
    sources: dict[tuple, str] = {}
    for source, key in [("pinned", _key(e)) for e in stored if not e.get("hidden")] + derived:
        if key not in sources or SOURCES.index(source) < SOURCES.index(sources[key]):
            sources[key] = source
    hidden = {_key(e) for e in stored if e.get("hidden")}
    rows = sorted(sources.items(), key=lambda item: SOURCES.index(item[1]))
    return [
        {
            "forge": forge,
            "repo": repo,
            "pr": pr,
            "source": source,
            "hidden": (forge, repo, pr) in hidden,
            "pull": pulls.get((forge, repo, pr)),
        }
        for (forge, repo, pr), source in rows
    ]
