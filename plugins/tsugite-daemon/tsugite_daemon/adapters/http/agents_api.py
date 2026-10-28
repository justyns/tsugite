"""AgentsApiMixin: /api/agents handlers for HTTPServer."""

import asyncio
from dataclasses import asdict
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from tsugite.workspace import Workspace
from tsugite_daemon.adapters.http.helpers import mounted_api_routes
from tsugite_daemon.agent_store import Agent
from tsugite_daemon.session_store import SessionSource

SETUP_FILES = ("IDENTITY.md", "USER.md", "AGENTS.md", "MEMORY.md")


def _setup_status(workspace: Path) -> dict:
    missing = [name for name in SETUP_FILES if not (workspace / name).exists()]
    return {
        "complete": not missing,
        "missing": missing,
        "fresh": len(missing) == len(SETUP_FILES),
        "workspace": str(workspace),
    }


class AgentsApiMixin:
    def _agents_api_routes(self) -> list:
        return [
            *mounted_api_routes(
                "/api/agents",
                "agents",
                [
                    Route("/", self._api_list_agents, methods=["GET"]),
                    Route("/", self._api_create_agent, methods=["POST"]),
                    Route("/{agent_id}", self._api_get_agent, methods=["GET"]),
                    Route("/{agent_id}", self._api_update_agent, methods=["PATCH"]),
                    Route("/{agent_id}/main", self._api_open_main_chat, methods=["POST"]),
                ],
            ),
            Route("/api/setup", self._api_setup_status, methods=["GET"]),
            Route("/api/setup", self._api_seed_setup, methods=["POST"]),
        ]

    async def _api_setup_status(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        workspace = Path(self.adapter.session_store.agents.default().workspace)
        return JSONResponse(_setup_status(workspace))

    async def _api_seed_setup(self, request: Request) -> JSONResponse:
        """Fill the default agent's workspace from the templates when it has none of the files."""
        if err := self._check_auth(request):
            return err
        body = await self._optional_json_body(request)
        workspace = Path(self.adapter.session_store.agents.default().workspace)
        if _setup_status(workspace)["fresh"]:
            await asyncio.to_thread(
                Workspace.create,
                workspace,
                user_name=body.get("user_name"),
                agent_name=body.get("agent_name"),
                about=body.get("about"),
            )
        return JSONResponse(_setup_status(workspace))

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
            agent = self.adapter.session_store.agents.create(Agent(id=agent_id, **self.adapter.agent_fields(body)))
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
            agent = agents.update(current.id, **self.adapter.agent_fields(body, current))
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse(asdict(agent))

    async def _api_open_main_chat(self, request: Request) -> JSONResponse:
        """The agent's main chat for the user, created on first open."""
        if err := self._check_auth(request):
            return err
        store = self.adapter.session_store
        agent = store.agents.get(request.path_params["agent_id"])
        if agent is None:
            return JSONResponse({"error": "unknown agent"}, status_code=404)
        body = await self._optional_json_body(request)
        user_id = self.adapter.resolve_http_user(body.get("user_id", "web-anonymous"))
        session = store.find_primary_session(user_id, agent.id)
        if session is None:
            session = store.create_default_session(
                user_id, title=agent.name, source=SessionSource.WEB.value, agent_id=agent.id
            )
            self.event_bus.emit("session_update", {"action": "created", "id": session.id})
        return JSONResponse(store.session_detail(session.id))
