"""Resolve an agent's frontmatter `attachments:` entries into Attachment objects.

Handles the spec forms frontmatter accepts - plain paths, Jinja-templated paths,
globs, tier groupings, `assign:` bindings, `-name` removals - and the index-block
form that summarizes a directory instead of inlining it.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from tsugite.attachments.base import Attachment, AttachmentContentType
from tsugite.md_agents import AttachmentSpec
from tsugite.renderer import humanize_mtime
from tsugite.utils import has_glob_chars

logger = logging.getLogger(__name__)


def _render_path(template: str) -> Optional[str]:
    """Render a Jinja path template; return None on render failure."""
    from tsugite.renderer import AgentRenderer

    try:
        return AgentRenderer().render_string(template)
    except Exception as e:
        logger.debug("Failed to render attachment path %r: %s", template, e)
        return None


def split_attachment_removals(
    items: List[Union[str, AttachmentSpec]],
) -> Tuple[set, List[Union[str, AttachmentSpec]]]:
    """Split `-filename` removal markers out of an attachments list.

    The leading-dash convention is string-only legacy syntax; AttachmentSpec items
    always pass through unchanged.
    """
    removals = {t.lstrip("-") for t in items if isinstance(t, str) and t.startswith("-")}
    keep = [t for t in items if not (isinstance(t, str) and t.startswith("-"))]
    return removals, keep


def _resolve_path(rendered: str, workspace_path: Optional[Path]) -> Path:
    p = Path(rendered)
    if not p.is_absolute() and workspace_path:
        p = workspace_path / p
    return p


def _expand_glob(pattern: str, workspace_path: Optional[Path]) -> List[Path]:
    """Expand a glob pattern relative to workspace, alphabetically sorted."""
    p = Path(pattern)
    if p.is_absolute():
        base = Path(p.anchor)
        rel = str(p.relative_to(p.anchor))
    elif workspace_path:
        base = workspace_path
        rel = pattern
    else:
        base = Path.cwd()
        rel = pattern
    matches = sorted(base.glob(rel))
    return [m for m in matches if m.is_file()]


def resolve_agent_config_attachments(
    items: List[Union[str, AttachmentSpec]],
    workspace_path: Optional[Path] = None,
) -> Tuple[List[Attachment], Dict[str, Any]]:
    """Resolve agent config attachment items to Attachment objects and Jinja bindings.

    String items keep legacy semantics (Jinja-render the path, fetch the single file).
    AttachmentSpec items support globs, `assign:` bindings, and `attach: false`.

    Args:
        items: List of strings or AttachmentSpec objects from agent_config.attachments
        workspace_path: Optional workspace path for resolving relative paths

    Returns:
        Tuple of (attachments to inject, dict of {assign_name: bound_value}).
        Bound value is `str` for single concrete files, `list[dict]` for globs,
        and `None` for a missing single file.
    """
    if not items:
        return [], {}

    from tsugite.attachments.file import FileHandler

    file_handler = FileHandler()
    attachments: List[Attachment] = []
    bindings: Dict[str, Any] = {}

    for item in items:
        before = len(attachments)
        if isinstance(item, str):
            _resolve_string_item(item, workspace_path, file_handler, attachments)
        else:
            _resolve_spec(item, workspace_path, file_handler, attachments, bindings)
        # Tag everything this item produced with its cache tier (a spec's tier;
        # plain strings are always tier 0). The context turn renders one cache
        # block per tier so a volatile file only invalidates its own block.
        tier = getattr(item, "tier", 0)
        if tier:
            for att in attachments[before:]:
                att.tier = tier

    return attachments, bindings


def _resolve_string_item(
    template: str,
    workspace_path: Optional[Path],
    file_handler: "FileHandler",
    attachments: List[Attachment],
) -> None:
    """Legacy string handling: render Jinja, resolve path, fetch single file."""
    rendered = _render_path(template)
    if rendered is None:
        return
    resolved = _resolve_path(rendered, workspace_path)
    if not resolved.exists():
        logger.debug("Agent attachment not found (skipped): %s", resolved)
        return
    att = file_handler.fetch(str(resolved))
    if att:
        attachments.append(att)
        logger.debug("Loaded agent attachment: %s", resolved)


def _resolve_spec(
    spec: AttachmentSpec,
    workspace_path: Optional[Path],
    file_handler: "FileHandler",
    attachments: List[Attachment],
    bindings: Dict[str, Any],
) -> None:
    """Resolve a single AttachmentSpec, populating attachments and bindings."""
    rendered = _render_path(spec.path)
    if rendered is None:
        if spec.assign:
            bindings[spec.assign] = None
        return

    if spec.mode == "index":
        _resolve_index_spec(spec, rendered, workspace_path, attachments, bindings)
        return

    if has_glob_chars(rendered):
        matched = _expand_glob(rendered, workspace_path)
        if spec.assign:
            bindings[spec.assign] = []
        for path in matched:
            # When attach=False, we only need text content for the binding; skip
            # FileHandler.fetch so binaries aren't base64-encoded just to be discarded.
            if not spec.attach and spec.assign:
                try:
                    content = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                bindings[spec.assign].append({"path": str(path), "content": content})
                continue
            att = file_handler.fetch(str(path))
            if not att:
                continue
            if spec.attach:
                attachments.append(att)
            if spec.assign and att.content_type == AttachmentContentType.TEXT:
                bindings[spec.assign].append({"path": str(path), "content": att.content})
        return

    resolved = _resolve_path(rendered, workspace_path)
    if not resolved.exists():
        if spec.assign:
            bindings[spec.assign] = None
        logger.debug("Agent attachment not found (skipped): %s", resolved)
        return

    att = file_handler.fetch(str(resolved))
    if not att:
        if spec.assign:
            bindings[spec.assign] = None
        return
    if spec.attach:
        attachments.append(att)
    if spec.assign:
        bindings[spec.assign] = att.content if att.content_type == AttachmentContentType.TEXT else None


def _resolve_index_spec(
    spec: AttachmentSpec,
    rendered_path: str,
    workspace_path: Optional[Path],
    attachments: List[Attachment],
    bindings: Dict[str, Any],
) -> None:
    """Resolve a mode: index spec into a single index Attachment and/or assign binding."""
    if has_glob_chars(rendered_path):
        paths = _expand_glob(rendered_path, workspace_path)
    else:
        single = _resolve_path(rendered_path, workspace_path)
        paths = [single] if single.exists() and single.is_file() else []

    if len(paths) > spec.max_entries:
        logger.warning(
            "Attachment index for %r exceeds max_entries=%d, truncating from %d",
            spec.path,
            spec.max_entries,
            len(paths),
        )
        paths = paths[: spec.max_entries]

    entries = [_extract_index_entry(p, spec.index_format) for p in paths]

    if spec.assign:
        bindings[spec.assign] = entries

    if not entries:
        return

    if spec.attach:
        att_name = spec.name or _derive_index_name(rendered_path)
        att_content = _format_index_block(spec.path, entries, spec.index_format)
        attachments.append(
            Attachment(
                name=att_name,
                content=att_content,
                content_type=AttachmentContentType.TEXT,
                mime_type="text/plain",
                mode="index",
            )
        )


def _extract_index_entry(path: Path, fmt: str) -> Dict[str, Any]:
    """Build an index entry dict for a single file. Falls back gracefully on errors."""
    entry: Dict[str, Any] = {"path": str(path)}

    if fmt == "path_only":
        # Cheapest format: skip stat + read entirely. Only `path` is used downstream.
        entry["heading"] = ""
        return entry

    try:
        stat = path.stat()
        entry["size_bytes"] = stat.st_size
        entry["mtime"] = stat.st_mtime
    except OSError:
        entry["size_bytes"] = 0
        entry["mtime"] = 0

    try:
        with path.open("rb") as f:
            head = f.read(2048)
        text = head.decode("utf-8", errors="replace")
    except OSError:
        text = ""

    if fmt == "first_line":
        entry["heading"] = _first_nonempty_line(text)
    elif fmt == "frontmatter":
        entry.update(_parse_frontmatter_for_index(text, path))
    else:  # first_heading (default)
        entry["heading"] = _first_markdown_heading(text)

    return entry


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _first_markdown_heading(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def _parse_frontmatter_for_index(text: str, path: Path) -> Dict[str, Any]:
    """Extract title/description/tags from YAML frontmatter; fall back to path_only on error."""
    if not text.startswith("---"):
        return {"heading": ""}
    try:
        from tsugite.utils import parse_yaml_frontmatter

        meta, _body = parse_yaml_frontmatter(text, str(path))
    except Exception:
        return {"heading": ""}
    out: Dict[str, Any] = {
        "heading": meta.get("title", "") or meta.get("description", ""),
    }
    if "title" in meta:
        out["title"] = meta["title"]
    if "description" in meta:
        out["description"] = meta["description"]
    if "tags" in meta:
        out["tags"] = meta["tags"]
    return out


def _format_index_block(pattern: str, entries: List[Dict[str, Any]], fmt: str) -> str:
    """Render the index Attachment content with a one-line prelude + bullets."""
    prelude = (
        f'File index for "{pattern}" ({len(entries)} files). Use read_file(path=...) to read any of them when relevant.'
    )
    lines = [prelude, ""]
    for entry in entries:
        path = entry.get("path", "")
        if fmt == "path_only":
            lines.append(path)
            continue

        if fmt == "frontmatter":
            title = entry.get("title") or entry.get("heading") or ""
            desc = entry.get("description", "")
            if title and desc:
                bullet = f"{path} - {title}: {desc}"
            elif title:
                bullet = f"{path} - {title}"
            else:
                bullet = path
        else:
            heading = entry.get("heading", "")
            bullet = f"{path} - {heading}" if heading else path

        relative = humanize_mtime(entry.get("mtime"))
        if relative:
            bullet = f"{bullet} ({relative})"
        lines.append(bullet)
    return "\n".join(lines)


def _derive_index_name(pattern: str) -> str:
    """Derive a default index attachment name from a glob pattern.

    Uses the deepest non-glob path segment, suffixed with `_index`. Falls back to
    `attachment_index` if the pattern is purely glob characters.
    """
    parts = Path(pattern).parts
    for segment in reversed(parts):
        if segment and not has_glob_chars(segment) and segment not in ("/", "."):
            return f"{segment}_index"
    return "attachment_index"
