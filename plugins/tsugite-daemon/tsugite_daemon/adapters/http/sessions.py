"""SessionsMixin: sessions HTTP handlers for HTTPServer (split from adapters/http.py)."""

from typing import TYPE_CHECKING, Any, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from tsugite_daemon.adapters.http.helpers import (
    HTTPAgentAdapter,
)

if TYPE_CHECKING:
    pass


class SessionsMixin:
    def _session_routes(self) -> list:
        return [
            Route("/api/sessions/{session_id}/settings", self._session_settings_get, methods=["GET"]),
            Route("/api/sessions/{session_id}/settings", self._session_settings_patch, methods=["PATCH"]),
            Route("/api/sessions", self._api_list_sessions, methods=["GET"]),
            Route("/api/sessions", self._api_start_session, methods=["POST"]),
            Route("/api/sessions/{session_id}/metadata", self._api_get_metadata, methods=["GET"]),
            Route("/api/sessions/{session_id}/metadata", self._api_update_metadata, methods=["PATCH"]),
            Route("/api/sessions/{session_id}/metadata/{key}", self._api_delete_metadata, methods=["DELETE"]),
            Route("/api/sessions/{session_id}/scratchpad", self._api_get_scratchpad, methods=["GET"]),
            Route("/api/sessions/{session_id}/scratchpad", self._api_update_scratchpad, methods=["PUT"]),
            Route("/api/sessions/{session_id}", self._api_get_session, methods=["GET"]),
            Route("/api/sessions/{session_id}", self._api_update_session, methods=["PATCH"]),
            Route("/api/sessions/{session_id}/cancel", self._api_cancel_session, methods=["POST"]),
            Route("/api/sessions/{session_id}/restart", self._api_restart_session, methods=["POST"]),
            Route("/api/sessions/{session_id}/events", self._api_session_events, methods=["GET"]),
            Route("/api/sessions/{session_id}/pin", self._api_pin_session, methods=["POST"]),
            Route("/api/sessions/{session_id}/unpin", self._api_unpin_session, methods=["POST"]),
            Route("/api/sessions/pinned/reorder", self._api_reorder_pins, methods=["POST"]),
            # NB: clear-primary literal must precede {session_id}/set-primary -- Starlette matches in order.
            Route("/api/sessions/clear-primary", self._api_clear_primary, methods=["POST"]),
            Route("/api/sessions/{session_id}/set-primary", self._api_set_primary, methods=["POST"]),
            Route("/api/sessions/{session_id}/mark-viewed", self._api_mark_viewed, methods=["POST"]),
        ]

    def _resolve_session_for_settings(
        self, request: Request
    ) -> tuple[Optional[HTTPAgentAdapter], Optional[str], Optional[JSONResponse]]:
        """Look up (adapter, session_id) for a /api/sessions/{session_id}/settings call."""
        session_id = request.path_params["session_id"]
        for adapter in self.adapters.values():
            try:
                adapter.session_store.get_session(session_id)
                return adapter, session_id, None
            except ValueError:
                continue
        return None, None, JSONResponse({"error": f"unknown session: {session_id}"}, status_code=404)

    def _resolve_effort_or_400(
        self, adapter: "HTTPAgentAdapter", value: Any, session_id: Optional[str] = None
    ) -> tuple[Optional[str], Optional[JSONResponse]]:
        """Validate a reasoning_effort value against the session's resolved model."""
        if value is None:
            return None, None
        from tsugite.models import UnsupportedEffortError, resolve_reasoning_effort

        try:
            return resolve_reasoning_effort(self._resolve_session_model(adapter, session_id), value), None
        except UnsupportedEffortError as err:
            return None, JSONResponse({"error": str(err), "supported": err.supported}, status_code=400)

    def _session_settings_payload(self, adapter: "HTTPAgentAdapter", session_id: str) -> dict:
        session = adapter.session_store.get_session(session_id)
        return {
            "reasoning_effort": adapter.session_store.get_reasoning_effort(session_id),
            "model": adapter.session_store.get_model_override(session_id),
            "agent": session.agent if session else None,
        }

    async def _session_settings_get(self, request: Request) -> JSONResponse:
        adapter, session_id, err = self._resolve_session_for_settings(request)
        if err:
            return err
        return JSONResponse(self._session_settings_payload(adapter, session_id))

    async def _session_settings_patch(self, request: Request) -> JSONResponse:
        adapter, session_id, err = self._resolve_session_for_settings(request)
        if err:
            return err

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        if "reasoning_effort" in body:
            value, err_resp = self._resolve_effort_or_400(adapter, body["reasoning_effort"], session_id)
            if err_resp:
                return err_resp
            adapter.session_store.set_reasoning_effort(session_id, value)

        if "model" in body:
            raw = body["model"]
            if raw is None or raw == "":
                adapter.session_store.set_model_override(session_id, None)
            elif not isinstance(raw, str):
                return JSONResponse({"error": "model must be a string"}, status_code=400)
            else:
                from tsugite.models import get_provider_and_model

                try:
                    get_provider_and_model(raw)
                except Exception as exc:  # noqa: BLE001
                    return JSONResponse({"error": f"unknown model: {raw} ({exc})"}, status_code=400)
                adapter.session_store.set_model_override(session_id, raw)

        if "agent" in body:
            name = body["agent"]
            if not isinstance(name, str) or not name:
                return JSONResponse({"error": "agent must be a non-empty string"}, status_code=400)
            if name not in self.adapters:
                return JSONResponse({"error": f"unknown agent: {name}"}, status_code=400)
            adapter.session_store.set_agent_override(session_id, name)

        return JSONResponse(self._session_settings_payload(adapter, session_id))

    def _session_detail(self, session_id: str) -> dict:
        return self.session_runner.store.session_detail(session_id)

    async def _api_list_sessions(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        state = request.query_params.get("state")
        sessions = self.session_runner.store.list_sessions(status=state)
        return JSONResponse(
            {
                "sessions": [
                    {
                        "id": s.id,
                        "agent": s.agent,
                        "source": s.source,
                        "state": s.status,
                        "prompt": s.prompt or "",
                        "created_at": s.created_at,
                        "updated_at": s.last_active,
                        "error": s.error,
                        "title": s.title,
                        "metadata": s.metadata or {},
                        "is_primary": s.is_primary,
                    }
                    for s in sessions
                ]
            }
        )

    async def _api_start_session(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        prompt = body.get("prompt", "").strip()
        agent = body.get("agent", "")
        if not prompt or not agent:
            return JSONResponse({"error": "prompt and agent are required"}, status_code=400)
        if agent not in self.adapters:
            return JSONResponse({"error": f"unknown agent: {agent}"}, status_code=400)

        from tsugite_daemon.session_store import Session, SessionSource

        session = Session(
            id=body.get("session_id", ""),
            agent=agent,
            source=SessionSource.BACKGROUND.value,
            prompt=prompt,
            model=body.get("model"),
            agent_file=body.get("agent_file"),
            notify=body.get("notify", []),
        )

        try:
            result = self.session_runner.start_session(session)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

        self.event_bus.emit("session_update", {"action": "started", "id": result.id})
        return JSONResponse(
            {"id": result.id, "status": result.status, "created_at": result.created_at},
            status_code=201,
        )

    async def _api_get_session(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        try:
            return JSONResponse(self._session_detail(session_id))
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)

    async def _api_update_session(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        title = body.get("title")
        status = body.get("status")
        pinned = body.get("pinned")
        has_pin_position = "pin_position" in body
        pin_position = body.get("pin_position")
        last_viewed_at = body.get("last_viewed_at")
        if title is None and status is None and pinned is None and not has_pin_position and last_viewed_at is None:
            return JSONResponse({"error": "No updatable fields provided"}, status_code=400)
        runner = self.session_runner
        try:
            result = {}
            if title is not None:
                runner.rename_session(session_id, title)
                result["title"] = title
            if pinned is not None or has_pin_position:
                # When only pin_position is sent, keep the existing pinned state but route
                # through set_pin so siblings rebalance instead of leaving gaps.
                target_pinned = bool(pinned) if pinned is not None else runner.store.get_session(session_id).pinned
                session = runner.set_pin(session_id, target_pinned, position=pin_position if has_pin_position else None)
                result["pinned"] = session.pinned
                result["pin_position"] = session.pin_position
            if last_viewed_at is not None:
                session = runner.mark_viewed(session_id, ts=last_viewed_at or None)
                result["last_viewed_at"] = session.last_viewed_at
            if status is not None:
                if status != "completed":
                    return JSONResponse({"error": "Only 'completed' status is allowed"}, status_code=400)
                runner.store.update_session(session_id, status=status)
                self.event_bus.emit("session_update", {"action": "completed", "id": session_id})
                result["status"] = status
            return JSONResponse({"ok": True, **result})
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)

    @staticmethod
    async def _optional_json_body(request: Request) -> dict:
        """Best-effort JSON body parse; missing/invalid bodies become {}."""
        try:
            body = await request.json()
        except Exception:
            return {}
        return body if isinstance(body, dict) else {}

    async def _api_pin_session(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        body = await self._optional_json_body(request)
        try:
            session = self.session_runner.set_pin(session_id, True, position=body.get("position"))
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        return JSONResponse({"ok": True, "pinned": session.pinned, "pin_position": session.pin_position})

    async def _api_unpin_session(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        try:
            self.session_runner.set_pin(session_id, False)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        return JSONResponse({"ok": True})

    async def _api_reorder_pins(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        ids = body.get("ids") if isinstance(body, dict) else None
        if not isinstance(ids, list) or not all(isinstance(s, str) for s in ids):
            return JSONResponse({"error": "ids must be a list of strings"}, status_code=400)
        ordered = self.session_runner.reorder_pins(ids)
        return JSONResponse({"ok": True, "ordered": [s.id for s in ordered]})

    async def _api_set_primary(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        try:
            self.session_runner.set_primary_session(session_id)
        except ValueError as e:
            msg = str(e)
            status = 404 if "not found" in msg else 400
            return JSONResponse({"error": msg}, status_code=status)
        return JSONResponse({"ok": True, "id": session_id, "is_primary": True})

    async def _api_clear_primary(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        agent = request.query_params.get("agent")
        user_id = request.query_params.get("user_id")
        if not agent or not user_id:
            return JSONResponse({"error": "agent and user_id query params required"}, status_code=400)
        self.session_runner.clear_primary_session(user_id, agent)
        return JSONResponse({"ok": True})

    async def _api_mark_viewed(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        body = await self._optional_json_body(request)
        try:
            session = self.session_runner.mark_viewed(session_id, ts=body.get("ts"))
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        return JSONResponse({"ok": True, "last_viewed_at": session.last_viewed_at})

    async def _api_cancel_session(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        try:
            self.session_runner.cancel_session(session_id)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        self.event_bus.emit("session_update", {"action": "cancelled", "id": session_id})
        return JSONResponse({"status": "cancelled"})

    async def _api_restart_session(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        try:
            old = self.session_runner.store.get_session(session_id)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)

        from tsugite_daemon.session_store import Session, SessionSource, SessionStatus

        restartable = {SessionStatus.FAILED.value, SessionStatus.CANCELLED.value}
        if old.status not in restartable:
            return JSONResponse({"error": f"cannot restart session in '{old.status}' state"}, status_code=400)

        # Carry forward per-session overrides + metadata so a restarted worker
        # session still runs in its provisioned worktree AND the JobsOrchestrator
        # still recognises it via metadata.job_id.
        new_session = Session(
            id="",
            agent=old.agent,
            source=old.source or SessionSource.BACKGROUND.value,
            prompt=old.prompt,
            model=old.model,
            agent_file=old.agent_file,
            notify=old.notify,
            workspace_override=old.workspace_override,
            metadata=dict(old.metadata or {}),
        )
        try:
            result = self.session_runner.start_session(new_session)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        self.event_bus.emit("session_update", {"action": "restarted", "id": result.id})
        return JSONResponse(
            {"id": result.id, "status": result.status, "restarted_from": session_id},
            status_code=201,
        )

    async def _api_session_events(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        try:
            self.session_runner.store.get_session(session_id)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        events = self.session_runner.store.read_events(session_id)
        return JSONResponse({"events": events})

    async def _api_get_metadata(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        try:
            session = self.session_runner.store.get_session(session_id)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        return JSONResponse({"metadata": session.metadata or {}})

    async def _api_update_metadata(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "Body must be a JSON object"}, status_code=400)
        try:
            session = self.session_runner.update_session_metadata(session_id, body)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse({"ok": True, "metadata": session.metadata or {}})

    async def _api_delete_metadata(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        key = request.path_params["key"]
        try:
            session = self.session_runner.delete_session_metadata(session_id, key)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse({"ok": True, "metadata": session.metadata or {}})

    async def _api_get_scratchpad(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        try:
            session = self.session_runner.store.get_session(session_id)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        return JSONResponse({"scratchpad": session.scratchpad})

    async def _api_update_scratchpad(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_sessions(request):
            return err
        session_id = request.path_params["session_id"]
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        content = body.get("content", "")
        if not isinstance(content, str):
            return JSONResponse({"error": "content must be a string"}, status_code=400)
        from tsugite.tools.scratchpad import SCRATCHPAD_HARD_LIMIT

        if len(content) > SCRATCHPAD_HARD_LIMIT:
            return JSONResponse(
                {"error": f"Content exceeds hard limit of {SCRATCHPAD_HARD_LIMIT} chars"}, status_code=400
            )
        try:
            self.session_runner.store.update_session(session_id, scratchpad=content)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        self.event_bus.emit("session_update", {"action": "scratchpad_updated", "id": session_id})
        return JSONResponse({"ok": True, "scratchpad": content})
