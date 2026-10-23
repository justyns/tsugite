"""RoomsMixin: /api/rooms handlers for HTTPServer."""

from typing import Callable, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from tsugite_daemon.adapters.http.helpers import mounted_api_routes
from tsugite_daemon.room_store import DEFAULT_MAX_ROUNDS, Room
from tsugite_daemon.rooms import RoomConflictError, RoomLookupError


def _max_rounds_error(value) -> Optional[JSONResponse]:
    if not isinstance(value, int) or isinstance(value, bool):
        return JSONResponse({"error": "max_rounds must be an integer"}, status_code=400)
    return None


def _job_id_error(value) -> Optional[JSONResponse]:
    if value is not None and not isinstance(value, str):
        return JSONResponse({"error": "job_id must be a string"}, status_code=400)
    return None


class RoomsMixin:
    def _rooms_routes(self) -> list:
        return mounted_api_routes(
            "/api/rooms",
            "rooms",
            [
                Route("/", self._api_list_rooms, methods=["GET"]),
                Route("/", self._api_create_room, methods=["POST"]),
                Route("/{room_id}", self._api_get_room, methods=["GET"]),
                Route("/{room_id}", self._api_update_room, methods=["PATCH"]),
                Route("/{room_id}/close", self._api_close_room, methods=["POST"]),
                Route("/{room_id}/stop", self._api_stop_room, methods=["POST"]),
                Route("/{room_id}/participants", self._api_invite, methods=["POST"]),
                Route("/{room_id}/participants/{agent_id}", self._api_set_muted, methods=["PATCH"]),
                Route("/{room_id}/participants/{agent_id}", self._api_remove_participant, methods=["DELETE"]),
                Route("/{room_id}/messages", self._api_post_room_message, methods=["POST"]),
            ],
        )

    def _room_response(self, action: Callable[[], Room], status_code: int = 200) -> JSONResponse:
        rooms = self.session_runner.rooms
        try:
            room = action()
        except RoomLookupError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        except RoomConflictError as e:
            return JSONResponse({"error": str(e)}, status_code=409)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse(rooms.detail(room), status_code=status_code)

    async def _api_list_rooms(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        return JSONResponse({"rooms": self.session_runner.rooms.list_details()})

    async def _api_get_room(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        return self._room_response(lambda: self.session_runner.rooms.get(request.path_params["room_id"]))

    async def _api_create_room(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        body = await self._optional_json_body(request)
        title = body.get("title")
        if not isinstance(title, str) or not title.strip():
            return JSONResponse({"error": "title is required"}, status_code=400)
        max_rounds = body.get("max_rounds", DEFAULT_MAX_ROUNDS)
        if err := _max_rounds_error(max_rounds):
            return err
        job_id = body.get("job_id")
        if err := _job_id_error(job_id):
            return err
        agents = self.session_runner.store.agents
        return self._room_response(
            lambda: self.session_runner.rooms.create(
                title.strip(),
                user_id=self.adapter.resolve_http_user(body.get("user_id", "web-anonymous")),
                lead_agent_id=agents.effective_id(body.get("lead_agent_id")),
                max_rounds=max_rounds,
                job_id=job_id,
            ),
            status_code=201,
        )

    async def _api_update_room(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        body = await self._optional_json_body(request)
        room_id = request.path_params["room_id"]
        rooms = self.session_runner.rooms
        if "max_rounds" not in body and "job_id" not in body:
            return JSONResponse({"error": "max_rounds or job_id is required"}, status_code=400)
        if "max_rounds" in body and (err := _max_rounds_error(body["max_rounds"])):
            return err
        if "job_id" in body and (err := _job_id_error(body["job_id"])):
            return err

        def update() -> Room:
            if "job_id" in body:
                rooms.check_job_link(room_id, body["job_id"])
            room = rooms.set_max_rounds(room_id, body["max_rounds"]) if "max_rounds" in body else None
            return rooms.link_job(room_id, body["job_id"]) if "job_id" in body else room

        return self._room_response(update)

    async def _api_stop_room(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        return self._room_response(lambda: self.session_runner.rooms.stop(request.path_params["room_id"]))

    async def _api_close_room(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        return self._room_response(lambda: self.session_runner.rooms.close(request.path_params["room_id"]))

    async def _api_invite(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        body = await self._optional_json_body(request)
        agent_id = body.get("agent_id")
        if not isinstance(agent_id, str):
            return JSONResponse({"error": "agent_id must be a string"}, status_code=400)
        return self._room_response(
            lambda: self.session_runner.rooms.invite(request.path_params["room_id"], agent_id, body.get("brief")),
            status_code=201,
        )

    async def _api_set_muted(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        body = await self._optional_json_body(request)
        muted = body.get("muted")
        if not isinstance(muted, bool):
            return JSONResponse({"error": "muted must be a boolean"}, status_code=400)
        params = request.path_params
        return self._room_response(
            lambda: self.session_runner.rooms.set_muted(params["room_id"], params["agent_id"], muted)
        )

    async def _api_remove_participant(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        params = request.path_params
        return self._room_response(lambda: self.session_runner.rooms.remove(params["room_id"], params["agent_id"]))

    async def _api_post_room_message(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        body = await self._optional_json_body(request)
        content = body.get("content")
        if not isinstance(content, str) or not content.strip():
            return JSONResponse({"error": "content is required"}, status_code=400)
        return self._room_response(
            lambda: self.session_runner.rooms.post_user_message(request.path_params["room_id"], content),
            status_code=201,
        )
