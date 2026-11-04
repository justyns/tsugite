"""Agent-facing forge tools (the `tsugite.plugins` entry point)."""

from datetime import datetime
from pathlib import Path

from tsugite.exceptions import ToolUnavailableError
from tsugite.secrets.registry import get_registry
from tsugite.tools import call_on_loop, tool
from tsugite.ui_surfaces import register_ui_surface

MAX_ROWS = 50
MAX_DIFF_CHARS = 20000
INCLUDE_PARTS = ("checks", "threads", "diff")

PRS_SURFACE = {
    "kind": "prs",
    "label": "Pull requests",
    "icon": "git",
    "entry": "ui/prs.html",
    "assets": Path(__file__).parent / "ui",
    "nav": True,
}

SESSION_SURFACE = {
    "kind": "session-prs",
    "label": "Session PRs",
    "icon": "git",
    "entry": "ui/session.html",
    "assets": Path(__file__).parent / "ui",
    "nav": False,
    "params": ["session"],
    "mode": "workspace",
}

register_ui_surface(**PRS_SURFACE)
register_ui_surface(**SESSION_SURFACE)

_adapter = None


def set_forge_adapter(adapter) -> None:
    global _adapter
    _adapter = adapter


def _running():
    if _adapter is None:
        raise ToolUnavailableError("the forge tools need the tsugite daemon with plugins.forge configured")
    return _adapter


def _age(timestamp: str, now: float) -> str:
    return _duration(now - datetime.fromisoformat(timestamp).timestamp())


def _duration(seconds: float) -> str:
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{int(seconds // size)}{unit}"
    return "0m"


def _line(pr, now: float) -> str:
    parts = [
        f"{pr.forge}:{pr.repo}#{pr.number} {pr.title}",
        pr.author,
        _age(pr.created_at, now),
        f"CI {pr.ci_state}",
        "threads ?" if pr.unresolved_threads is None else f"{pr.unresolved_threads} unresolved",
    ]
    if pr.labels:
        parts.append(", ".join(pr.labels))
    if pr.draft:
        parts.append("draft")
    return " · ".join(parts)


@tool(require_daemon=True, category="forge")
def forge_prs(
    state: str = "open",
    repo: str | None = None,
    author: str | None = None,
    label: str | None = None,
    needs_review: bool | None = None,
) -> str:
    """List cached open pull requests one per line. repo is owner/repo or forge:owner/repo, needs_review=True means awaiting your review."""
    adapter = _running()
    now = adapter.clock()
    prs = adapter.prs(state=state, repo=repo, author=author, label=label, needs_review=needs_review)
    lines = [_line(pr, now) for pr in prs[:MAX_ROWS]] or ["No open pull requests match."]
    if len(prs) > MAX_ROWS:
        lines.append(f"Showing {MAX_ROWS} of {len(prs)}. Narrow with repo, author, label or needs_review.")
    for cache in adapter.forges.values():
        if cache.error:
            cached = (
                "nothing cached yet"
                if cache.last_refresh is None
                else f"showing a {_duration(now - cache.last_refresh)} old cache"
            )
            lines.append(f"{cache.config.name}: refresh failed, {cached}: {cache.error}")
    return get_registry().mask("\n".join(lines))


@tool(require_daemon=True, category="forge")
def forge_pr(repo: str, number: int, include: list[str] | None = None) -> str:
    """Show one pull request (repo is owner/repo or forge:owner/repo) with live checks, threads and diff, as picked by include."""
    adapter = _running()
    cache, name = adapter.resolve(repo)
    parts = INCLUDE_PARTS if include is None else include
    detail = adapter.pr_detail(cache.config.name, name, number, parts)
    pr = detail["pr"]
    status = f"{pr['state']}, draft" if pr["draft"] else pr["state"]
    if "threads" in detail:
        unresolved, total = detail["threads"]["unresolved"], detail["threads"]["total"]
    else:
        unresolved, total = pr["unresolved_threads"], None
    threads = f"{'?' if unresolved is None else unresolved} unresolved" + ("" if total is None else f" of {total}")
    lines = [
        f"{pr['forge']}:{pr['repo']}#{pr['number']} {pr['title']}",
        f"{status} · by {pr['author']} · opened {_age(pr['created_at'], adapter.clock())} ago"
        f" · {pr['head_ref']} -> {pr['base_ref']}",
        f"labels: {', '.join(pr['labels']) or 'none'} · review requested: {', '.join(pr['requested_reviewers']) or 'none'}"
        f" · review: {pr['review_state']}",
        f"CI: {detail.get('checks', pr['ci_state'])} · threads: {threads}",
        pr["url"],
    ]
    if "diff" in detail:
        diff = detail["diff"]
        lines += ["diff:", diff[:MAX_DIFF_CHARS].rstrip("\n")]
        if len(diff) > MAX_DIFF_CHARS:
            lines.append(f"[diff truncated at {MAX_DIFF_CHARS} of {len(diff)} characters]")
    return get_registry().mask("\n".join(lines))


@tool(require_daemon=True, category="forge")
def forge_repos() -> str:
    """List each configured forge with the repos it reads and when it last refreshed."""
    adapter = _running()
    now = adapter.clock()
    lines = []
    for cache in adapter.forges.values():
        refreshed = (
            "never refreshed" if cache.last_refresh is None else f"refreshed {_duration(now - cache.last_refresh)} ago"
        )
        lines.append(f"{cache.config.name} ({cache.config.kind}), {refreshed}: {', '.join(cache.repos) or 'no repos'}")
        if cache.error:
            lines.append(f"  last refresh failed: {cache.error}")
    return get_registry().mask("\n".join(lines))


def _pin_action(action: str, repo: str, pr: int | None) -> str:
    from tsugite.tools.sessions import get_current_session_id

    adapter = _running()
    session_id = get_current_session_id()
    if not session_id:
        raise ToolUnavailableError("no current session")
    if action == "unpin":
        forge, name = adapter.pinned(session_id, repo, pr)
    else:
        cache, name = adapter.resolve(repo)
        forge = cache.config.name
    call_on_loop(adapter.loop, adapter.update_refs, session_id, action, forge, name, pr)
    return f"{forge}:{name}" + ("" if pr is None else f"#{pr}")


@tool(require_daemon=True, category="forge")
def forge_pin(repo: str, pr: int | None = None) -> str:
    """Pin a repo (owner/repo or forge:owner/repo), or one of its pull requests, to the current session."""
    return f"Pinned {_pin_action('pin', repo, pr)} to this session."


@tool(require_daemon=True, category="forge")
def forge_unpin(repo: str, pr: int | None = None) -> str:
    """Unpin a repo or pull request that forge_pin pinned to the current session."""
    return f"Unpinned {_pin_action('unpin', repo, pr)}."
