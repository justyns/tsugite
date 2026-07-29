"""Turn delegation file paths into attachments, degrading non-inlinable files to a path hint.

Backs the ``files=`` parameter of ``spawn_agent`` (sync, handed to the subprocess
child) and ``spawn_job`` (daemon) so a delegating agent can pass real files -
images especially - to the child it delegates to, and the child's model sees them.

``can_inline_file`` is the single source of the "should this file ride inline as
model context vs. stay on disk" gate, called directly by both the daemon's upload
handlers and delegation here so uploaded and delegated files gate identically.
"""

import logging
from pathlib import Path
from typing import List, Tuple

from tsugite.attachments.base import (
    SUPPORTED_INLINE_IMAGE_MEDIA_TYPES,
    Attachment,
    AttachmentContentType,
)
from tsugite.attachments.file import FileHandler

logger = logging.getLogger(__name__)

MAX_TEXT_ATTACH_SIZE = 50 * 1024  # 50KB -- ~12K tokens
MAX_BINARY_ATTACH_SIZE = 10 * 1024 * 1024  # 10MB

_file_handler = FileHandler()


def can_inline_file(path: Path, size: int, supports_vision: bool = True) -> bool:
    """Whether a file should ride inline as model context vs. be left on disk.

    Text within the size cap inlines. Images inline only for a vision-capable
    model and only in a media type a mainstream vision API accepts (svg/bmp/tiff
    never inline). Other binaries inline within the size cap. Everything else
    stays on disk and reaches the child as a path hint instead.

    supports_vision defaults True so callers that can't resolve a model stay
    optimistic rather than stranding an image.
    """
    mime_type, content_type = _file_handler.detect_content_type(path)
    if content_type == AttachmentContentType.TEXT:
        return size <= MAX_TEXT_ATTACH_SIZE
    if path.suffix.lower() in FileHandler.BINARY_EXTENSIONS:
        if content_type == AttachmentContentType.IMAGE and (
            not supports_vision or mime_type not in SUPPORTED_INLINE_IMAGE_MEDIA_TYPES
        ):
            return False
        return size <= MAX_BINARY_ATTACH_SIZE
    return False


def resolve_delegation_files(files: List[str], workspace: Path) -> List[Path]:
    """Resolve each path against the workspace, rejecting traversal and misses.

    Mirrors the upload endpoint guard: a path resolving outside the workspace
    (absolute, ``../``, or via a symlink) or not pointing at an existing file
    raises ValueError so the caller can surface a clear error, not crash.
    """
    ws = Path(workspace).resolve()
    resolved: List[Path] = []
    for f in files:
        raw = Path(f)
        candidate = (raw if raw.is_absolute() else ws / raw).resolve()
        if not candidate.is_relative_to(ws):
            raise ValueError(f"File path escapes the workspace: {f}")
        if not candidate.is_file():
            raise ValueError(f"File not found in workspace: {f}")
        resolved.append(candidate)
    return resolved


def partition_delegation_files(paths: List[Path], supports_vision: bool) -> Tuple[List[Path], List[Path]]:
    """Split resolved paths into ``(inline, hint_only)`` using :func:`can_inline_file`."""
    inline: List[Path] = []
    hint_only: List[Path] = []
    for p in paths:
        try:
            size = p.stat().st_size
        except OSError:
            hint_only.append(p)
            continue
        (inline if can_inline_file(p, size, supports_vision) else hint_only).append(p)
    return inline, hint_only


def materialize_delegation_attachments(paths: List[Path]) -> List[Attachment]:
    """Fetch each path into an Attachment via FileHandler, skipping unreadable files."""
    attachments: List[Attachment] = []
    for p in paths:
        try:
            attachments.append(_file_handler.fetch(str(p)))
        except Exception as e:  # noqa: BLE001 -- one bad file shouldn't sink the batch
            logger.warning("Failed to attach delegated file %s: %s", p, e)
    return attachments


def format_delegation_hint(hint_paths: List[Path]) -> str:
    """Suffix telling the child where non-inlined delegated files live on disk."""
    if not hint_paths:
        return ""
    paths = ", ".join(str(p) for p in hint_paths)
    return f"\n\n[Delegated files available on disk (open to read their contents): {paths}]"
