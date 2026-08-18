"""Locating, jailing, keying, and replacing the documents the plugin serves.

Every path arriving over HTTP or from a tool goes through `resolve_existing`,
except the save callback, which uses plain `resolve` so a save still lands if the
file vanished. The route layer is the only place these errors become status codes.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

EXTENSION = ".docx"
MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

# The document server rejects a key longer than 128 characters or carrying
# anything outside alphanumerics, "-" and "_", so a hex digest it is.
_KEY_LENGTH = 32


def is_document(path: Path) -> bool:
    """Whether a path names something this plugin will serve.

    The one predicate `resolve` refuses on and the listing filters by, so a listing
    never offers a row the routes then decline.
    """
    return path.suffix.lower() == EXTENSION


class OutsideDocumentsError(ValueError):
    """A caller-supplied path resolved outside the documents directory."""


class NoSuchDocumentError(ValueError):
    """The path is inside the documents directory, but there is no file there."""


class NotADocumentError(ValueError):
    """The path is inside the documents directory, but it is not a document."""


def resolve(documents_dir: Path, relative: str) -> Path:
    """Resolve a caller-supplied path inside the documents directory.

    Args:
        documents_dir: The jail root.
        relative: The path as it arrived from HTTP or from a tool call.

    Returns:
        The absolute path, which may or may not exist.

    Raises:
        OutsideDocumentsError: The path resolves outside the documents directory.
        NotADocumentError: The path names something other than a document.
    """
    root = documents_dir.resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise OutsideDocumentsError(f"path {relative!r} escapes the documents directory")
    # The jail alone would make the whole directory readable, and a file token is a
    # bearer credential that travels to the document server and its logs: minting one
    # for a private file hands it to a third party trusted only with documents. The
    # write path needs the same rule, or a callback picks the name of the file it creates.
    if not is_document(target):
        raise NotADocumentError(f"not a document: {relative}")
    return target


def resolve_existing(documents_dir: Path, relative: str) -> Path:
    """Resolve a caller-supplied path to a document that is actually there.

    Returns:
        The absolute path of an existing file.

    Raises:
        OutsideDocumentsError: The path resolves outside the documents directory.
        NotADocumentError: The path names something other than a document.
        NoSuchDocumentError: Nothing is at that path.
    """
    path = resolve(documents_dir, relative)
    if not path.is_file():
        raise NoSuchDocumentError(f"no such document: {relative}")
    return path


def canonical(documents_dir: Path, relative: str) -> str:
    """Reduce a caller-supplied path to the one spelling of the document it names.

    `notes.docx`, `./notes.docx` and `sub/../notes.docx` are one file, so per-document
    state keys on this. Keyed on what the caller typed instead, one document would get
    a lock and a document key per spelling, and two turns on it would neither serialize
    nor see each other.

    Returns:
        The path relative to the documents directory, in posix form.

    Raises:
        OutsideDocumentsError: The path resolves outside the documents directory.
        NotADocumentError: The path names something other than a document.
    """
    root = documents_dir.resolve()
    return resolve(root, relative).relative_to(root).as_posix()


def document_key(relative: str, path: Path, generation: int) -> str:
    """Derive the document server's cache key for a file's current contents.

    The server caches by key, so the key has to rotate whenever the bytes change.
    It also refuses to reopen a key whose session has ended, and a session can end
    without touching the file at all, so the generation carries what the file
    cannot: how many sessions on this document have already been retired.

    Args:
        relative: The document's `canonical` path relative to the documents
            directory. Seeding on the caller's own spelling would mint a key per
            spelling, which is the same bytes cached twice on the document server.
        path: The resolved file on disk.
        generation: How many editing sessions this document has already finished.

    Returns:
        A key the document server accepts.
    """
    stat = path.stat()
    seed = f"{relative}:{stat.st_mtime_ns}:{stat.st_size}:{generation}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:_KEY_LENGTH]


def list_documents(documents_dir: Path) -> list[dict]:
    """List every editable document under the documents directory.

    Returns:
        One `{"path", "size", "modified"}` entry per document, sorted by path.
    """
    root = documents_dir.resolve()
    if not root.is_dir():
        return []
    documents = []
    pending = deque([root])
    while pending:
        with os.scandir(pending.popleft()) as entries:
            for entry in entries:
                # A symlinked directory would walk out of the jail, and a listed
                # document whose path then fails `resolve` is worse than an absent one.
                if entry.is_symlink():
                    continue
                path = Path(entry.path)
                if entry.is_dir():
                    pending.append(path)
                elif is_document(path):
                    stat = entry.stat()
                    documents.append(
                        {
                            "path": path.relative_to(root).as_posix(),
                            "size": stat.st_size,
                            "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                        }
                    )
    return sorted(documents, key=lambda entry: entry["path"])


@contextmanager
def replacing(path: Path):
    """Replace a document in a single step, through a sibling file.

    An interrupted write would leave the editor's own save looking like a corrupt
    document, so the new bytes are written elsewhere and then take the name. A write
    that raises takes its half-written file with it.

    The temp name is unique per write, not per process: a save arriving from the
    document server does not hold the turn's lock, so two writers on one document
    can be in here at once, and a shared name means one of them renames the other's
    file out from under it.

    Yields:
        The path to write the new contents to.
    """
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(handle)
    tmp = Path(name)
    try:
        yield tmp
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def write_atomic(path: Path, content: bytes) -> None:
    """Replace a document's bytes in a single step."""
    with replacing(path) as tmp:
        tmp.write_bytes(content)
