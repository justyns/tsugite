"""HooksConfigMixin: agent hooks-config HTTP handlers for HTTPServer.

The file served here is the agent workspace's .tsugite/hooks.yaml, which the
daemon loads fresh on every hook firing - saves apply immediately.
"""

from pathlib import Path
from typing import Optional

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


class HooksConfigMixin:
    def _hooks_routes(self) -> list:
        return [
            Route("/api/agents/{agent}/hooks", self._get_hooks_config, methods=["GET"]),
            Route("/api/agents/{agent}/hooks", self._save_hooks_config, methods=["PUT"]),
        ]

    @staticmethod
    def _hooks_path(adapter) -> Path:
        return Path(adapter.agent_config.workspace_dir) / ".tsugite" / "hooks.yaml"

    @staticmethod
    def _parse_hooks_yaml(raw: str) -> tuple[Optional[dict], Optional[str]]:
        """Validate hooks YAML through the real loader models. Returns
        (phases, None) on success - {phase: [rule summaries]} - or (None, error)."""
        import yaml

        from tsugite.hooks import HooksConfig

        if not raw.strip():
            return {}, None
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as e:
            return None, f"invalid YAML: {e}"
        if not isinstance(data, dict):
            return None, "top-level mapping with a 'hooks' key required"
        if "hooks" not in data:
            return None, "missing top-level 'hooks' key"
        try:
            config = HooksConfig.model_validate(data["hooks"] or {})
        except Exception as e:
            return None, str(e)
        phases: dict = {}
        for phase in HooksConfig.model_fields:
            rules = getattr(config, phase)
            if not rules:
                continue
            phases[phase] = [
                {
                    "name": r.name,
                    "type": r.type,
                    "run": r.run if isinstance(r.run, str) else (" ".join(r.run) if r.run else None),
                    "agent": r.agent,
                    "tools": r.tools,
                    "match": r.match,
                    "wait": r.wait,
                    "capture_as": r.capture_as,
                    "only_interactive": r.only_interactive,
                }
                for r in rules
            ]
        return phases, None

    def _hooks_payload(self, adapter) -> dict:
        path = self._hooks_path(adapter)
        raw = path.read_text(encoding="utf-8") if path.exists() else ""
        phases, error = self._parse_hooks_yaml(raw)
        return {
            "path": str(path),
            "exists": path.exists(),
            "raw": raw,
            "phases": phases,
            "error": error,
        }

    async def _get_hooks_config(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err
        return JSONResponse(self._hooks_payload(adapter))

    async def _save_hooks_config(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        raw = body.get("raw")
        if not isinstance(raw, str):
            return JSONResponse({"error": "raw must be a string"}, status_code=400)
        _phases, error = self._parse_hooks_yaml(raw)
        if error:
            return JSONResponse({"error": error}, status_code=400)
        path = self._hooks_path(adapter)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(raw, encoding="utf-8")
        return JSONResponse(self._hooks_payload(adapter))
