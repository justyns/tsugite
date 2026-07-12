"""FilesMixin: files HTTP handlers for HTTPServer (split from adapters/http.py)."""

from pathlib import Path
from typing import Optional

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from tsugite.agent_inheritance import iter_agent_search_paths
from tsugite.skill_discovery import get_builtin_skills_path
from tsugite.utils import parse_yaml_frontmatter
from tsugite_daemon.adapters.http.helpers import (
    logger,
)


class FilesMixin:
    def _file_routes(self) -> list:
        return [
            Route("/api/agent-files", self._list_agent_files, methods=["GET"]),
            Route("/api/agent-files/content", self._read_agent_file, methods=["GET"]),
            Route("/api/agent-files/content", self._save_agent_file, methods=["PUT"]),
            Route("/api/skill-files", self._list_skill_files, methods=["GET"]),
            Route("/api/skill-files/content", self._read_skill_file, methods=["GET"]),
            Route("/api/skill-files/content", self._save_skill_file, methods=["PUT"]),
            Route("/api/skills/issues", self._list_skill_issues, methods=["GET"]),
        ]

    def _get_allowed_agent_dirs(self) -> list[tuple[Path, str, bool]]:
        """Return (directory, source_label, is_readonly) for all agent directories.

        Routes through `iter_agent_search_paths` so the search order + dedup
        logic match every other site (find_agent_file, repl_completer, etc.).
        Workspace agent dirs come from configured agents and feed in as
        extra_project_dirs.
        """
        extra_project_dirs: list[Path] = []
        for cfg in self.agent_configs.values():
            extra_project_dirs.extend([cfg.workspace_dir / ".tsugite", cfg.workspace_dir / "agents"])
        return [
            (entry.path, entry.source.value, entry.readonly)
            for entry in iter_agent_search_paths(extra_project_dirs=extra_project_dirs)
        ]

    def _validate_md_path(
        self, path_str: str, allowed_dirs: list[tuple[Path, str, bool]]
    ) -> tuple[Path, bool, Optional[JSONResponse]]:
        """Validate a markdown file path is within allowed directories.

        Returns (resolved_path, is_readonly, error_response_or_none).
        """
        try:
            resolved = Path(path_str).resolve()
        except (ValueError, OSError):
            return Path(), False, JSONResponse({"error": "invalid path"}, status_code=400)
        if resolved.suffix != ".md":
            return Path(), False, JSONResponse({"error": "only .md files allowed"}, status_code=400)
        for dir_path, _, readonly in allowed_dirs:
            try:
                resolved.relative_to(dir_path.resolve())
                return resolved, readonly, None
            except ValueError:
                continue
        return Path(), False, JSONResponse({"error": "path not in allowed directories"}, status_code=403)

    def _collect_md_files(self, allowed_dirs: list[tuple[Path, str, bool]], glob_pattern: str = "*.md") -> list[dict]:
        """Collect markdown files from allowed directories with frontmatter metadata."""
        files = []
        seen_paths: set[Path] = set()
        for dir_path, source, readonly in allowed_dirs:
            if not dir_path.is_dir():
                continue
            for md_file in sorted(dir_path.glob(glob_pattern)):
                resolved = md_file.resolve()
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                name, description = md_file.stem, ""
                try:
                    content = md_file.read_text(encoding="utf-8")
                    fm, _ = parse_yaml_frontmatter(content, str(md_file))
                    name = fm.get("name", md_file.stem)
                    description = fm.get("description", "")
                except Exception as e:
                    logger.warning("Failed to parse frontmatter %s: %s", md_file, e)
                files.append(
                    {
                        "path": str(resolved),
                        "name": name,
                        "source": source,
                        "readonly": readonly,
                        "description": description,
                    }
                )
        return files

    async def _read_md_file(self, request: Request, allowed_dirs: list[tuple[Path, str, bool]]) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        path_str = request.query_params.get("path", "")
        if not path_str:
            return JSONResponse({"error": "path parameter required"}, status_code=400)
        resolved, readonly, err = self._validate_md_path(path_str, allowed_dirs)
        if err:
            return err
        if not resolved.exists():
            return JSONResponse({"error": "file not found"}, status_code=404)
        try:
            content = resolved.read_text(encoding="utf-8")
        except OSError as e:
            return JSONResponse({"error": f"read failed: {e}"}, status_code=500)
        return JSONResponse({"path": str(resolved), "content": content, "readonly": readonly})

    async def _save_md_file(self, request: Request, allowed_dirs: list[tuple[Path, str, bool]]) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        path_str = body.get("path", "")
        content = body.get("content")
        if not path_str or content is None:
            return JSONResponse({"error": "path and content required"}, status_code=400)
        resolved, readonly, err = self._validate_md_path(path_str, allowed_dirs)
        if err:
            return err
        if readonly:
            return JSONResponse({"error": "file is read-only (builtin)"}, status_code=403)
        if not resolved.exists():
            return JSONResponse({"error": "file not found"}, status_code=404)
        try:
            resolved.write_text(content, encoding="utf-8")
        except OSError as e:
            return JSONResponse({"error": f"write failed: {e}"}, status_code=500)
        return JSONResponse({"status": "saved"})

    async def _list_agent_files(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        return JSONResponse({"files": self._collect_md_files(self._get_allowed_agent_dirs(), "*.md")})

    async def _read_agent_file(self, request: Request) -> JSONResponse:
        return await self._read_md_file(request, self._get_allowed_agent_dirs())

    async def _save_agent_file(self, request: Request) -> JSONResponse:
        return await self._save_md_file(request, self._get_allowed_agent_dirs())

    def _get_allowed_skill_dirs(self) -> list[tuple[Path, str, bool]]:
        """Return (directory, source_label, is_readonly) for all skill directories."""
        dirs: list[tuple[Path, str, bool]] = [(get_builtin_skills_path(), "builtin", True)]
        seen: set[Path] = set()
        for cfg in self.agent_configs.values():
            for subdir in [cfg.workspace_dir / ".tsugite" / "skills", cfg.workspace_dir / "skills"]:
                resolved = subdir.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    dirs.append((subdir, "project", False))
        dirs.append((Path.home() / ".config" / "tsugite" / "skills", "global", False))
        return dirs

    async def _list_skill_files(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        return JSONResponse({"files": self._collect_md_files(self._get_allowed_skill_dirs(), "**/*.md")})

    async def _read_skill_file(self, request: Request) -> JSONResponse:
        return await self._read_md_file(request, self._get_allowed_skill_dirs())

    async def _save_skill_file(self, request: Request) -> JSONResponse:
        return await self._save_md_file(request, self._get_allowed_skill_dirs())

    async def _list_skill_issues(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        from tsugite.tools.skills import get_failed_skills_list

        return JSONResponse({"issues": get_failed_skills_list()})
