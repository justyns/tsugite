"""WorkspaceFilesMixin: agent-workspace file HTTP handlers for HTTPServer."""

import asyncio
import mimetypes
import shutil
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from tsugite.attachments.delegation import can_inline_file
from tsugite_daemon.adapters.http.helpers import (
    HTTPAgentAdapter,
    _deduplicate_dest,
    _file_handler,
    _is_text_mime,
)

# Defensive cap on a single recursive workspace listing so a pathologically large
# tree degrades (truncated flag on the response) instead of building an unbounded
# payload and stalling the client.
MAX_WORKSPACE_ENTRIES = 20000

# Cap on a raw workspace file served inline to the browser (image thumbnails +
# the lightbox). Larger than the text-view cap (downscaled photos routinely
# exceed 1MB) but still bounded, so a huge file can't be read into one Response.
MAX_WORKSPACE_RAW_SIZE = 10 * 1024 * 1024


class WorkspaceFilesMixin:
    def _workspace_routes(self) -> list:
        return [
            Route("/api/workspace", self._list_workspace_files, methods=["GET"]),
            Route("/api/workspace/content", self._read_workspace_file, methods=["GET"]),
            Route("/api/workspace/raw", self._read_workspace_raw, methods=["GET"]),
            Route("/api/workspace/content", self._save_workspace_file, methods=["PUT"]),
            Route("/api/workspace/attach", self._attach_workspace_file, methods=["POST"]),
        ]

    def _validate_workspace_path(
        self, adapter: "HTTPAgentAdapter", path_str: str
    ) -> tuple[Path, Optional[JSONResponse]]:
        """Validate a workspace file path stays within the workspace directory."""
        workspace_dir = adapter.runtime.workspace_dir
        try:
            resolved = (workspace_dir / path_str).resolve()
        except (ValueError, OSError):
            return Path(), JSONResponse({"error": "invalid path"}, status_code=400)
        if not resolved.is_relative_to(workspace_dir.resolve()):
            return Path(), JSONResponse({"error": "path outside workspace"}, status_code=403)
        return resolved, None

    def _walk_workspace_entries(
        self,
        target: Path,
        workspace_dir: Path,
        gitignore_spec,
        recursive: bool,
    ) -> tuple[list[dict], bool]:
        """Build the flat listing for `target`: one level, or the whole subtree.

        Guard rails, all shared with the one-level walk: symlinks are never
        followed or listed (S_ISLNK skip), so a symlink can neither escape the
        workspace nor cycle the walk; gitignored directories are pruned before
        descent, keeping .git and other ignored trees out of the recursive walk;
        and the total is capped at MAX_WORKSPACE_ENTRIES, returning a truncated
        flag rather than an unbounded response.
        """
        import stat as stat_mod

        entries: list[dict] = []
        truncated = False
        pending: deque[Path] = deque([target])
        while pending:
            current = pending.popleft()
            try:
                children = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except OSError:
                if current is target:
                    raise  # failing to list the requested dir itself stays a 500, as before
                continue  # an unreadable nested dir is skipped, not fatal to the whole walk
            for item in children:
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
                if len(entries) >= MAX_WORKSPACE_ENTRIES:
                    truncated = True
                    pending.clear()
                    break
                if is_dir:
                    entries.append({"path": rel, "name": item.name, "is_dir": True})
                    if recursive:
                        pending.append(item)
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
        return entries, truncated

    async def _list_workspace_files(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        workspace_dir = adapter.runtime.workspace_dir
        if not workspace_dir.is_dir():
            return JSONResponse({"entries": [], "subdir": "", "workspace_dir": str(workspace_dir)})

        subdir = request.query_params.get("subdir", "")
        recursive = request.query_params.get("recursive", "").lower() in ("1", "true", "yes")
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
        try:
            entries, truncated = self._walk_workspace_entries(target, workspace_dir, gitignore_spec, recursive)
        except OSError as e:
            return JSONResponse({"error": f"listing failed: {e}"}, status_code=500)

        payload = {"entries": entries, "subdir": subdir, "workspace_dir": str(workspace_dir)}
        if truncated:
            payload["truncated"] = True
        return JSONResponse(payload)

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
            content = await asyncio.to_thread(resolved.read_text, encoding="utf-8")
        except UnicodeDecodeError:
            return JSONResponse({"path": path_str, "content": None, "is_text": False, "size": st.st_size})
        except OSError as e:
            return JSONResponse({"error": f"read failed: {e}"}, status_code=500)

        return JSONResponse({"path": path_str, "content": content, "is_text": True})

    async def _read_workspace_raw(self, request: Request) -> Response:
        """Serve a workspace file's raw bytes with a guessed content-type.

        The text read endpoint (`workspace/content`) returns null for non-text,
        so image thumbnails and the lightbox read their bytes here instead. Same
        auth + path validation as the text read; size-capped; never JSON-wrapped.
        """
        adapter, err = self._get_adapter(request)
        if err:
            return err

        path_str = request.query_params.get("path", "")
        if not path_str:
            return JSONResponse({"error": "path parameter required"}, status_code=400)

        resolved, path_err = self._validate_workspace_path(adapter, path_str)
        if path_err:
            return path_err
        if not resolved.exists() or resolved.is_dir():
            return JSONResponse({"error": "file not found"}, status_code=404)

        st = resolved.stat()
        if st.st_size > MAX_WORKSPACE_RAW_SIZE:
            return JSONResponse(
                {"error": f"file too large (max {MAX_WORKSPACE_RAW_SIZE // (1024 * 1024)}MB)"},
                status_code=413,
            )

        try:
            data = await asyncio.to_thread(resolved.read_bytes)
        except OSError as e:
            return JSONResponse({"error": f"read failed: {e}"}, status_code=500)

        media_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        # Private so a shared cache never retains a workspace file; the short max-age
        # lets the lightbox reuse the thumbnail's bytes without a second round trip.
        return Response(content=data, media_type=media_type, headers={"Cache-Control": "private, max-age=3600"})

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
        # _is_text_mime is extension-based, so it also gates creation of new files.
        if not _is_text_mime(resolved):
            return JSONResponse({"error": "file type not editable"}, status_code=400)
        if resolved.is_dir():
            return JSONResponse({"error": "path is a directory"}, status_code=400)

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(resolved.write_text, content, encoding="utf-8")
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

        uploads_dir = adapter.runtime.workspace_dir / "uploads"
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
        mime_type, content_type = _file_handler.detect_content_type(dest)
        context_attach = can_inline_file(dest, file_size)

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
