"""Built-in context providers shipped with tsugite core (no plugin required).

Currently the "Workspace file" provider: its ``choices`` list the session
workspace's text files and its ``capture`` reads a picked file's text as one
context item. Pure core - stdlib plus the ``ctx`` dict (``workspace_dir``) only,
never a daemon import - so the same registry the daemon and CLI read always has
it. ``register_builtin_providers`` is called from ``context.ensure_loaded`` (a
called function, so it re-registers after a test registry reset).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional

from tsugite.attachments.base import Attachment
from tsugite.context import ContextChoice, ContextProvider, register_context_provider

# Directory names pruned from the walk. Any dot-prefixed dir (``.git``, ``.venv``,
# ...) is also pruned; these are the common non-hidden ones.
_SKIP_DIRS = {"node_modules", "__pycache__"}

# Extensions treated as "obvious binaries" and skipped when listing choices. The
# capture side additionally sniffs bytes, so a mislabelled text file still reads.
_BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tiff", ".svgz",
    ".pdf", ".zip", ".gz", ".tar", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".jar",
    ".exe", ".dll", ".so", ".dylib", ".o", ".a", ".obj", ".bin", ".wasm",
    ".pyc", ".pyo", ".class", ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv", ".flac", ".ogg", ".webm",
    ".sqlite", ".db", ".dat", ".lock",
}  # fmt: skip

_MAX_CHOICES = 500
_MAX_VALUE_CHARS = 4000
_MAX_FILE_BYTES = 1_000_000


def _workspace_root(ctx: dict) -> Optional[Path]:
    """The session workspace as a directory ``Path``, or None when absent."""
    wd = ctx.get("workspace_dir")
    if not wd:
        return None
    root = Path(wd)
    return root if root.is_dir() else None


def _iter_workspace_files(root: Path) -> Iterator[str]:
    """Yield ``/``-joined relative paths of text files under ``root``, pruning
    hidden dirs, ``node_modules`` / ``__pycache__``, dotfiles and binaries."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in _SKIP_DIRS]
        for name in filenames:
            if name.startswith(".") or Path(name).suffix.lower() in _BINARY_SUFFIXES:
                continue
            yield (Path(dirpath) / name).relative_to(root).as_posix()


def _file_choices(ctx: dict) -> list[ContextChoice]:
    root = _workspace_root(ctx)
    if root is None:
        return []
    rels = sorted(_iter_workspace_files(root))[:_MAX_CHOICES]
    return [ContextChoice(value=rel, label=rel) for rel in rels]


def _resolve_within(root: Path, arg: str) -> Optional[Path]:
    """Resolve ``root/arg`` and confirm it stays within ``root``. Returns the
    resolved path, or None if ``arg`` escapes via ``..`` / an absolute path (a
    ``pathlib`` join drops ``root`` when ``arg`` is absolute) or fails to
    resolve. Symlinks are resolved, so a link pointing outside is rejected."""
    try:
        root_resolved = root.resolve()
        candidate = (root_resolved / arg).resolve()
    except (OSError, ValueError, RuntimeError):
        return None
    return candidate if candidate.is_relative_to(root_resolved) else None


def _file_capture(arg: Optional[str], ctx: dict) -> list[Attachment]:
    if not arg:
        return []
    root = _workspace_root(ctx)
    if root is None:
        return []
    target = _resolve_within(root, arg)
    if target is None or not target.is_file():
        return []
    try:
        if target.stat().st_size > _MAX_FILE_BYTES:
            return []
        data = target.read_bytes()
    except OSError:
        return []
    if b"\x00" in data:
        return []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return []
    return [Attachment.context(key=f"file:{arg}", label=arg, value=text[:_MAX_VALUE_CHARS])]


WORKSPACE_FILE_PROVIDER = ContextProvider(
    key="file",
    label="Workspace file",
    icon="file",
    picker=True,
    capture=_file_capture,
    choices=_file_choices,
)


def register_builtin_providers() -> None:
    """Register every built-in context provider (idempotent by key)."""
    register_context_provider(WORKSPACE_FILE_PROVIDER)
