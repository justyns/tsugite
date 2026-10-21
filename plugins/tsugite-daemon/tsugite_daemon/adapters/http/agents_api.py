"""AgentsApiMixin: /api/agents handlers for HTTPServer."""

from dataclasses import asdict
from pathlib import Path
from typing import Optional

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from tsugite_daemon.adapters.http.helpers import mounted_api_routes
from tsugite_daemon.agent_store import EDITABLE_FIELDS, Agent

NULLABLE_FIELDS = frozenset({"model", "reasoning_effort", "color"})


class AgentsApiMixin:
    def _agents_api_routes(self) -> list:
        return mounted_api_routes(
            "/api/agents",
            "agents",
            [
                Route("/", self._api_list_agents, methods=["GET"]),
                Route("/", self._api_create_agent, methods=["POST"]),
                Route("/{agent_id}", self._api_get_agent, methods=["GET"]),
                Route("/{agent_id}", self._api_update_agent, methods=["PATCH"]),
            ],
        )

    def _agent_fields(self, body: dict, current: Optional[Agent] = None) -> dict:
        """The body's fields, type-checked, with the workspace resolved. Raises ValueError."""
        if unknown := body.keys() - EDITABLE_FIELDS:
            raise ValueError(f"Unknown agent field(s): {', '.join(sorted(unknown))}")
        if current is not None and current.is_default and body.keys() & {"workspace", "template"}:
            raise ValueError("The default agent's workspace and template are set in the daemon config")
        for key in EDITABLE_FIELDS & body.keys():
            if not isinstance(body[key], str) and not (body[key] is None and key in NULLABLE_FIELDS):
                raise ValueError(f"{key} must be a string")
        fields = dict(body)
        if "workspace" in body or "template" in body:
            workspace = Path(body.get("workspace", current.workspace if current else "")).expanduser()
            if not workspace.is_absolute() or not workspace.is_dir():
                raise ValueError(f"workspace must be an absolute path to a directory: {workspace}")
            workspace = workspace.resolve()
            state_dir = self.adapter.session_store.agents.state_dir.resolve()
            if workspace == Path("/") or workspace.is_relative_to(state_dir):
                raise ValueError(f"workspace cannot be / or inside the daemon state dir: {workspace}")
            fields["workspace"] = str(workspace)
            template = body.get("template", current.template if current else "")
            if not self.adapter._resolve_agent_path(template, workspace):
                raise ValueError(f"template not found: {template}")
        return fields

    async def _api_list_agents(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        return JSONResponse({"agents": [asdict(a) for a in self.adapter.session_store.agents.list()]})

    async def _api_get_agent(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        agent = self.adapter.session_store.agents.get(request.path_params["agent_id"])
        if agent is None:
            return JSONResponse({"error": "unknown agent"}, status_code=404)
        return JSONResponse(asdict(agent))

    async def _api_create_agent(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        body = await self._optional_json_body(request)
        agent_id = body.pop("id", None)
        if not isinstance(agent_id, str) or not agent_id or "workspace" not in body:
            return JSONResponse({"error": "id and workspace are required"}, status_code=400)
        body.setdefault("name", agent_id)
        body.setdefault("template", self.adapter.runtime.agent_file)
        try:
            agent = self.adapter.session_store.agents.create(Agent(id=agent_id, **self._agent_fields(body)))
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse(asdict(agent), status_code=201)

    async def _api_update_agent(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        agents = self.adapter.session_store.agents
        current = agents.get(request.path_params["agent_id"])
        if current is None:
            return JSONResponse({"error": "unknown agent"}, status_code=404)
        body = await self._optional_json_body(request)
        try:
            agent = agents.update(current.id, **self._agent_fields(body, current))
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse(asdict(agent))
