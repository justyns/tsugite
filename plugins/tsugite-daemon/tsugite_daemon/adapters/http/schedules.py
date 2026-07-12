"""SchedulesMixin: schedules HTTP handlers for HTTPServer."""

from dataclasses import fields as dataclass_fields

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from tsugite_daemon.adapters.http.helpers import mounted_api_routes
from tsugite_daemon.scheduler import ScheduleEntry, entry_to_dict


class SchedulesMixin:
    def _schedule_routes(self) -> list:
        return [
            *mounted_api_routes(
                "/api/schedules",
                "schedules",
                [
                    Route("/", self._list_schedules, methods=["GET"]),
                    Route("/", self._create_schedule, methods=["POST"]),
                    # NB: cleanup literal must precede {schedule_id} -- Starlette matches in order.
                    Route("/cleanup", self._cleanup_schedules, methods=["POST"]),
                    Route("/{schedule_id}", self._get_schedule, methods=["GET"]),
                    Route("/{schedule_id}", self._update_schedule, methods=["PATCH"]),
                    Route("/{schedule_id}", self._delete_schedule, methods=["DELETE"]),
                    Route("/{schedule_id}/enable", self._enable_schedule, methods=["POST"]),
                    Route("/{schedule_id}/disable", self._disable_schedule, methods=["POST"]),
                    Route("/{schedule_id}/run", self._run_schedule, methods=["POST"]),
                    Route("/{schedule_id}/sessions", self._schedule_sessions, methods=["GET"]),
                ],
            ),
        ]

    def _schedule_action(self, schedule_id: str, action: str, status_label: str) -> JSONResponse:
        """Run a scheduler action (enable/disable/remove) with standard error handling."""
        try:
            getattr(self.scheduler, action)(schedule_id)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        self.event_bus.emit("schedule_update", {"action": status_label, "id": schedule_id})
        return JSONResponse({"status": status_label})

    async def _list_schedules(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_scheduler(request):
            return err
        return JSONResponse({"schedules": [entry_to_dict(e) for e in self.scheduler.list()]})

    async def _create_schedule(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_scheduler(request):
            return err

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        required = {"id", "agent", "prompt", "schedule_type"}
        missing = required - set(body.keys())
        if missing:
            return JSONResponse({"error": f"missing fields: {', '.join(missing)}"}, status_code=400)

        try:
            valid_fields = {f.name for f in dataclass_fields(ScheduleEntry)}
            entry = ScheduleEntry(**{k: v for k, v in body.items() if k in valid_fields})
            entry = self.scheduler.add(entry)
        except (ValueError, TypeError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)

        self.event_bus.emit("schedule_update", {"action": "created", "id": entry.id})
        return JSONResponse(entry_to_dict(entry), status_code=201)

    async def _get_schedule(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_scheduler(request):
            return err
        try:
            entry = self.scheduler.get(request.path_params["schedule_id"])
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        return JSONResponse(entry_to_dict(entry))

    async def _update_schedule(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_scheduler(request):
            return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        schedule_id = request.path_params["schedule_id"]
        allowed = {
            "prompt",
            "cron_expr",
            "run_at",
            "timezone",
            "agent",
            "schedule_type",
            "model",
            "agent_file",
            "max_turns",
            "execution_type",
            "command",
            "script_timeout",
            "target_session",
        }
        fields = {k: v for k, v in body.items() if k in allowed}
        if not fields:
            return JSONResponse({"error": "no updatable fields provided"}, status_code=400)

        try:
            entry = self.scheduler.update(schedule_id, **fields)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        self.event_bus.emit("schedule_update", {"action": "updated", "id": schedule_id})
        return JSONResponse(entry_to_dict(entry))

    async def _delete_schedule(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_scheduler(request):
            return err
        return self._schedule_action(request.path_params["schedule_id"], "remove", "removed")

    async def _enable_schedule(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_scheduler(request):
            return err
        return self._schedule_action(request.path_params["schedule_id"], "enable", "enabled")

    async def _disable_schedule(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_scheduler(request):
            return err
        return self._schedule_action(request.path_params["schedule_id"], "disable", "disabled")

    async def _run_schedule(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_scheduler(request):
            return err
        schedule_id = request.path_params["schedule_id"]
        try:
            self.scheduler.get(schedule_id)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        self.scheduler.fire_now(schedule_id)
        self.event_bus.emit("schedule_update", {"action": "triggered", "id": schedule_id})
        return JSONResponse({"status": "triggered", "schedule_id": schedule_id})

    async def _cleanup_schedules(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_scheduler(request):
            return err
        removed = self.scheduler.cleanup()
        return JSONResponse({"removed": removed, "count": len(removed)})

    async def _schedule_sessions(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_scheduler(request):
            return err
        schedule_id = request.path_params["schedule_id"]
        # Find any adapter to access the session store
        adapter = next(iter(self.adapters.values()), None)
        if not adapter:
            return JSONResponse({"error": "no adapters"}, status_code=500)
        sessions = adapter.session_store.list_sessions(parent_id=schedule_id, source="schedule")
        return JSONResponse(
            {
                "schedule_id": schedule_id,
                "sessions": [
                    {
                        "id": s.id,
                        "status": s.status,
                        "created_at": s.created_at,
                        "last_active": s.last_active,
                        "result": (s.result or "")[:500],
                        "error": s.error,
                    }
                    for s in sessions
                ],
            }
        )
