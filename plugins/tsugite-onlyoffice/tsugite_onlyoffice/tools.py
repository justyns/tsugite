"""Agent-facing OnlyOffice document tools.

This module is the `tsugite.plugins` entry point, so every process that lists
tools imports it, sandboxed subprocess executors included. Keeping tsugite_daemon
and starlette out of it at module scope keeps those processes off the daemon half
of the plugin, and the import stays one-directional: the adapter imports the
tools, never the other way. It gets its configuration from adapter-set module
state because `load_tool_plugins()` is called with no config argument.

Every tool takes a `path` relative to the configured documents directory, and
every one of them goes through the runtime, which is what jails the path and
coordinates the edit with whatever the document server is doing to the file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from tsugite.tools import tool

if TYPE_CHECKING:
    from tsugite_onlyoffice.docx import Document

_runtime = None


def set_onlyoffice_runtime(runtime) -> None:
    """Wire the daemon-owned OnlyOffice runtime into this module.

    Called from the adapter's `start()`; called with None from `stop()` to drop
    the reference.
    """
    global _runtime
    _runtime = runtime


def runtime_available() -> bool:
    """True when a daemon-side adapter wired itself in here."""
    return _runtime is not None


def _anchor(anchor: str) -> int | str:
    """Read an anchor as a paragraph number when it is one, and literally otherwise."""
    return int(anchor) if anchor.strip().isdigit() else anchor


def _through_runtime(call: Callable[[], dict]) -> dict:
    """Run a call against the runtime, turning every refusal into a tool error.

    A bad path is a ValueError and a save the document server never returned is a
    RuntimeError; both are things the agent can act on, so both come back as text
    rather than as a traceback.
    """
    if not runtime_available():
        return {"error": "onlyoffice runtime not available (not running in daemon mode)"}
    try:
        return call()
    except (ValueError, RuntimeError) as exc:
        return {"error": str(exc)}


def _edit(path: str, work: Callable[[Document], dict]) -> dict:
    """Apply an edit through the runtime, which coordinates it with any live session."""
    return _through_runtime(lambda: _runtime.edit_document(path, work))


@tool(require_daemon=True, category="onlyoffice")
def doc_read(path: str) -> dict:
    """Read a document as numbered paragraphs, with every comment thread on it.

    `text` is one line per paragraph, numbered from 1. Those numbers are anchors:
    passing "3" to `doc_insert` or `doc_comment` addresses the whole third
    paragraph, and any other string addresses its first literal occurrence.

    Comments come back in the order the document anchors them. A comment's `date`
    is whatever wrote it put there: the editor stamps its own local time and
    suffixes a `Z`, so a date from the editor and a date from these tools are not
    on the same clock. Read a date as a label, never as an ordering.

    Args:
        path: The document, relative to the configured documents directory.

    Returns:
        Dict with `path`, the numbered `text`, and `comments`. Each comment
        carries its `id`, `author`, `date`, `text`, the `anchor` text its range
        covers, whether it is `resolved`, and `parent`: the id of the comment it
        replies to, or None when it opens a thread. An `id` is derived from the
        comment's author, date and text, which is all a save leaves alone, so
        rewording a comment retires its id: a `doc_reply` or `doc_resolve` on the
        old one comes back as an error rather than reaching another thread. An
        `id` of None means the comment predates the thread part of the format and
        cannot be replied to or resolved.
    """

    def read() -> dict:
        document = _runtime.open_document(path)
        return {"path": path, "text": document.text(), "comments": document.comments()}

    return _through_runtime(read)


@tool(require_daemon=True, category="onlyoffice")
def doc_insert(path: str, anchor: str, text: str) -> dict:
    """Insert text into a document, after an anchor.

    The inserted text takes on the formatting of the run it lands in, so it reads
    as part of the surrounding sentence rather than as pasted plain text.

    Args:
        path: The document, relative to the configured documents directory.
        anchor: A paragraph number from `doc_read`, or literal text to insert after.
        text: What to insert. Include the spacing you want around it.

    Returns:
        Dict with `path` and `inserted`, or `{"error": ...}`.
    """

    def work(document: Document) -> dict:
        document.insert(_anchor(anchor), text)
        return {"path": path, "inserted": text}

    return _edit(path, work)


@tool(require_daemon=True, category="onlyoffice")
def doc_replace(path: str, target: str, replacement: str) -> dict:
    """Replace every occurrence of a literal string in a document.

    Args:
        path: The document, relative to the configured documents directory.
        target: The text to find. It has to lie within a single paragraph.
        replacement: What to put in its place. An empty string deletes.

    Returns:
        Dict with `path` and `replaced`, the number of occurrences changed, or
        `{"error": ...}`.
    """

    def work(document: Document) -> dict:
        return {"path": path, "replaced": document.replace(target, replacement)}

    return _edit(path, work)


@tool(require_daemon=True, category="onlyoffice")
def doc_comment(path: str, anchor: str, text: str) -> dict:
    """Comment on a span of a document, as a real docx comment.

    The comment is authored under the configured agent name, so a human reading
    the document in Word or in the editor sees who wrote it.

    Args:
        path: The document, relative to the configured documents directory.
        anchor: A paragraph number from `doc_read`, or literal text to comment on.
        text: The comment body.

    Returns:
        Dict with `path` and the new `comment_id`, or `{"error": ...}`.
    """

    def work(document: Document) -> dict:
        return {"path": path, "comment_id": document.comment(_anchor(anchor), text, _runtime.author)}

    return _edit(path, work)


@tool(require_daemon=True, category="onlyoffice")
def doc_reply(path: str, comment_id: str, text: str) -> dict:
    """Reply to a comment, in its thread and on its anchor.

    Args:
        path: The document, relative to the configured documents directory.
        comment_id: The comment being replied to, from `doc_read`.
        text: The reply body.

    Returns:
        Dict with `path`, the reply's own `comment_id`, and the `parent` it hangs
        under, or `{"error": ...}`.
    """

    def work(document: Document) -> dict:
        reply_id = document.reply(comment_id, text, _runtime.author)
        return {"path": path, "comment_id": reply_id, "parent": comment_id}

    return _edit(path, work)


@tool(require_daemon=True, category="onlyoffice")
def doc_resolve(path: str, comment_id: str) -> dict:
    """Mark a comment thread as resolved.

    Args:
        path: The document, relative to the configured documents directory.
        comment_id: The comment that opened the thread, from `doc_read`.

    Returns:
        Dict with `path`, `comment_id` and `resolved`, or `{"error": ...}`.
    """

    def work(document: Document) -> dict:
        document.resolve(comment_id)
        return {"path": path, "comment_id": comment_id, "resolved": True}

    return _edit(path, work)
