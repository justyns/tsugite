"""HTTP API adapter with SSE streaming and webhook inbox."""

import asyncio
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from tsugite.daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite.daemon.config import AgentConfig, HTTPConfig, WebhookConfig
from tsugite.daemon.scheduler import ScheduleEntry
from tsugite.events.base import BaseEvent
from tsugite.history.models import CompactionSummary, Turn
from tsugite.history.storage import SessionStorage, get_history_dir
from tsugite.ui.jsonl import JSONLUIHandler

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

logger = logging.getLogger(__name__)


class SSEProgressHandler(JSONLUIHandler):
    """Converts agent events to SSE messages via an async queue."""

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.done = False
        self.has_final = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def handle_event(self, event: BaseEvent) -> None:
        """Handle event from agent thread — schedule onto the event loop."""
        super().handle_event(event)

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        if event_type == "final_result":
            self.has_final = True
        payload = {"type": event_type, **data}
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self.queue.put_nowait, payload)
        else:
            self.queue.put_nowait(payload)

    def signal_done(self):
        """Set done and wake up the generator."""
        self.done = True
        self.queue.put_nowait(None)

    async def event_generator(self):
        while True:
            try:
                data = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if self.done:
                    break
                yield ": keepalive\n\n"
                continue
            if data is None:
                break
            yield f"data: {json.dumps(data)}\n\n"
        # Drain remaining
        while not self.queue.empty():
            data = self.queue.get_nowait()
            if data is not None:
                yield f"data: {json.dumps(data)}\n\n"
        yield 'data: {"type": "done"}\n\n'


class HTTPAgentAdapter(BaseAdapter):
    """Per-agent adapter for HTTP. Lifecycle managed by HTTPServer."""

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class HTTPServer:
    """Runs a Starlette ASGI app with uvicorn for the HTTP API."""

    def __init__(
        self,
        config: HTTPConfig,
        adapters: dict[str, HTTPAgentAdapter],
        webhooks: list[WebhookConfig],
        agent_configs: dict[str, AgentConfig],
    ):
        self.config = config
        self.adapters = adapters
        self.webhooks = {w.token: w for w in webhooks}
        self.agent_configs = agent_configs
        self._auth_tokens = set(config.auth_tokens)
        self._server = None
        self.scheduler = None  # Set by Gateway after SchedulerAdapter is created
        self.app = self._build_app()

    def _check_auth(self, request: Request) -> Optional[JSONResponse]:
        if not self._auth_tokens:
            return None
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        if token not in self._auth_tokens:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return None

    def _get_adapter(self, request: Request) -> tuple[Optional[HTTPAgentAdapter], Optional[JSONResponse]]:
        """Authenticate and resolve the agent adapter from the request.

        Returns (adapter, None) on success, or (None, error_response) on failure.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return None, auth_err
        agent_name = request.path_params["agent"]
        adapter = self.adapters.get(agent_name)
        if not adapter:
            return None, JSONResponse({"error": f"unknown agent: {agent_name}"}, status_code=404)
        return adapter, None

    def _build_app(self) -> Starlette:
        routes = [
            Route("/api/health", self._health, methods=["GET"]),
            Route("/api/agents", self._list_agents, methods=["GET"]),
            Route("/api/agents/{agent}/sessions", self._list_sessions, methods=["GET"]),
            Route("/api/agents/{agent}/chat", self._chat, methods=["POST"]),
            Route("/api/agents/{agent}/status", self._status, methods=["GET"]),
            Route("/api/agents/{agent}/attachments", self._attachments, methods=["GET"]),
            Route("/api/agents/{agent}/history", self._history, methods=["GET"]),
            Route("/api/agents/{agent}/compact", self._compact, methods=["POST"]),
            Route("/api/schedules", self._list_schedules, methods=["GET"]),
            Route("/api/schedules", self._create_schedule, methods=["POST"]),
            Route("/api/schedules/{schedule_id}", self._get_schedule, methods=["GET"]),
            Route("/api/schedules/{schedule_id}", self._delete_schedule, methods=["DELETE"]),
            Route("/api/schedules/{schedule_id}/enable", self._enable_schedule, methods=["POST"]),
            Route("/api/schedules/{schedule_id}/disable", self._disable_schedule, methods=["POST"]),
            Route("/api/schedules/{schedule_id}/run", self._run_schedule, methods=["POST"]),
            Route("/webhook/{token}", self._webhook, methods=["POST"]),
            Route("/", self._serve_ui, methods=["GET"]),
        ]
        return Starlette(routes=routes)

    async def _health(self, request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "agents": list(self.adapters.keys())})

    async def _list_agents(self, request: Request) -> JSONResponse:
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        agents = [
            {
                "name": name,
                "agent_file": adapter.agent_config.agent_file,
                "workspace_dir": str(adapter.agent_config.workspace_dir),
            }
            for name, adapter in self.adapters.items()
        ]
        return JSONResponse({"agents": agents})

    async def _list_sessions(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err
        sessions_dir = adapter.agent_config.workspace_dir / "daemon_sessions"

        sessions = []
        if sessions_dir.is_dir():
            for path in sorted(sessions_dir.glob("*.json")):
                user_id = path.stem
                try:
                    data = json.loads(path.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                if user_id.isdigit():
                    label = f"Discord: {user_id}"
                elif user_id.startswith("web-"):
                    label = f"Web: {user_id}"
                else:
                    label = user_id
                sessions.append({
                    "user_id": user_id,
                    "label": label,
                    "conversation_id": data.get("conversation_id", ""),
                    "created_at": data.get("created_at", ""),
                })

        return JSONResponse({"sessions": sessions})

    async def _status(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        user_id = request.query_params.get("user_id", "web-anonymous")
        adapter.session_manager.get_or_create_session(user_id)
        session = adapter.session_manager.sessions.get(user_id)

        return JSONResponse({
            "model": adapter.resolve_model(),
            "tokens": session.cumulative_tokens if session else 0,
            "context_limit": adapter.session_manager.context_limit,
            "threshold": adapter.session_manager.compaction_threshold,
            "message_count": session.message_count if session else 0,
            "attachments": [
                {"name": a.name, "content_type": a.content_type.value, "mime_type": a.mime_type}
                for a in adapter.workspace_attachments
            ],
        })

    async def _attachments(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err
        attachments = []
        for a in adapter.workspace_attachments:
            entry = {"name": a.name, "content_type": a.content_type.value, "mime_type": a.mime_type}
            if a.content_type.value == "text":
                entry["content"] = a.content
            else:
                entry["size_bytes"] = len(a.content) if a.content else 0
            attachments.append(entry)
        return JSONResponse({"attachments": attachments})

    def _collect_turns(self, session_id: str) -> list:
        """Collect turns from a session and its compaction chain."""
        history_dir = get_history_dir()
        visited = set()

        # Walk the compaction chain, caching loaded storage objects
        chain = []
        current_id = session_id
        while current_id and current_id not in visited:
            visited.add(current_id)
            path = history_dir / f"{current_id}.jsonl"
            if not path.exists():
                break
            try:
                storage = SessionStorage.load(path)
                chain.append(storage)
                current_id = storage._meta.compacted_from if storage._meta else None
            except Exception:
                break

        # Reverse so oldest session is first; insert compaction markers between sessions
        chain.reverse()
        items = []
        for i, storage in enumerate(chain):
            try:
                turns = [r for r in storage.load_records() if isinstance(r, Turn)]
                items.extend(turns)
                if i < len(chain) - 1:
                    summary = None
                    for r in chain[i + 1].load_records():
                        if isinstance(r, CompactionSummary):
                            summary = r.summary
                            break
                    items.append({"marker": "compaction", "summary": summary})
            except Exception as e:
                logger.warning("Failed to load history: %s", e)

        return items

    async def _history(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        user_id = request.query_params.get("user_id", "web-anonymous")
        conversation_id = adapter.session_manager.get_or_create_session(user_id)

        turns = self._collect_turns(conversation_id)

        result_turns = []
        for item in turns:
            if isinstance(item, dict) and item.get("marker") == "compaction":
                entry = {"type": "compaction"}
                if item.get("summary"):
                    entry["summary"] = item["summary"]
                result_turns.append(entry)
                continue
            user_msg = ""
            for msg in item.messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    user_msg = content if isinstance(content, str) else str(content)
                    break
            result_turns.append({
                "user": user_msg,
                "assistant": item.final_answer or "",
                "timestamp": item.timestamp.isoformat() if item.timestamp else None,
                "tools_used": item.functions_called or [],
            })

        return JSONResponse({"conversation_id": conversation_id, "turns": result_turns})

    async def _compact(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        user_id = body.get("user_id", "web-anonymous")

        adapter.session_manager.get_or_create_session(user_id)
        session = adapter.session_manager.sessions.get(user_id)

        if not session or session.message_count == 0:
            return JSONResponse({"error": "no session to compact"}, status_code=404)

        old_conv_id = session.conversation_id
        try:
            await adapter._compact_session(user_id)
        except Exception as e:
            return JSONResponse({"error": f"compaction failed: {e}"}, status_code=500)

        new_session = adapter.session_manager.sessions.get(user_id)
        return JSONResponse({
            "status": "compacted",
            "old_conversation_id": old_conv_id,
            "new_conversation_id": new_session.conversation_id if new_session else None,
        })

    async def _chat(self, request: Request) -> Response:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        message = body.get("message", "").strip()
        if not message:
            return JSONResponse({"error": "message is required"}, status_code=400)

        user_id = body.get("user_id", "web-anonymous")

        channel_context = ChannelContext(
            source="http",
            channel_id=None,
            user_id=user_id,
            reply_to=f"http:{user_id}",
            metadata={"client_ip": request.client.host if request.client else "unknown"},
        )

        progress = SSEProgressHandler()
        progress.set_loop(asyncio.get_running_loop())
        custom_logger = SimpleNamespace(ui_handler=progress)

        async def run_agent():
            try:
                response = await adapter.handle_message(
                    user_id=user_id,
                    message=message,
                    channel_context=channel_context,
                    custom_logger=custom_logger,
                )
                # Only emit final_result if the EventBus didn't already
                # (FinalAnswerEvent fires during handle_message for normal completions,
                # but not for max_turns/error cases)
                if not progress.has_final:
                    progress._emit("final_result", {"result": response})

                # Emit session info for the web UI status bar
                session = adapter.session_manager.sessions.get(user_id)
                if session:
                    progress._emit("session_info", {
                        "tokens": session.cumulative_tokens,
                        "context_limit": adapter.session_manager.context_limit,
                        "threshold": adapter.session_manager.compaction_threshold,
                        "message_count": session.message_count,
                        "model": adapter.resolve_model(),
                        "attachments": [a.name for a in adapter.workspace_attachments],
                    })
            except Exception as e:
                logger.exception("[%s] Chat error", adapter.agent_name)
                progress._emit("error", {"error": str(e)})
            finally:
                progress.signal_done()

        asyncio.create_task(run_agent())

        return StreamingResponse(
            progress.event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    def _require_auth_and_scheduler(self, request: Request) -> Optional[JSONResponse]:
        """Check auth and scheduler availability. Returns error response or None."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        if not self.scheduler:
            return JSONResponse({"error": "scheduler not available"}, status_code=503)
        return None

    def _schedule_action(self, schedule_id: str, action: str, status_label: str) -> JSONResponse:
        """Run a scheduler action (enable/disable/remove) with standard error handling."""
        try:
            getattr(self.scheduler, action)(schedule_id)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        return JSONResponse({"status": status_label})

    async def _list_schedules(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_scheduler(request):
            return err
        return JSONResponse({"schedules": [asdict(e) for e in self.scheduler.list()]})

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
            entry = ScheduleEntry(**{k: v for k, v in body.items() if k in ScheduleEntry.__dataclass_fields__})
            entry = self.scheduler.add(entry)
        except (ValueError, TypeError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)

        return JSONResponse(asdict(entry), status_code=201)

    async def _get_schedule(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_scheduler(request):
            return err
        try:
            entry = self.scheduler.get(request.path_params["schedule_id"])
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        return JSONResponse(asdict(entry))

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
            entry = self.scheduler.get(schedule_id)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        asyncio.create_task(self.scheduler._fire_schedule(entry))
        return JSONResponse({"status": "triggered", "schedule_id": schedule_id})

    async def _webhook(self, request: Request) -> JSONResponse:
        token = request.path_params["token"]
        webhook = self.webhooks.get(token)
        if not webhook:
            return JSONResponse({"error": "invalid webhook token"}, status_code=404)

        if webhook.agent not in self.agent_configs:
            return JSONResponse({"error": "webhook agent not configured"}, status_code=500)

        try:
            raw = await request.body()
        except Exception:
            return JSONResponse({"error": "failed to read body"}, status_code=400)

        try:
            payload_data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload_data = raw.decode("utf-8", errors="replace")

        agent_config = self.agent_configs[webhook.agent]
        inbox_dir = agent_config.workspace_dir / "inbox" / "webhooks"
        inbox_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        filename = f"{now.strftime('%Y%m%dT%H%M%S')}-{webhook.source}.json"
        envelope = {
            "source": webhook.source,
            "agent": webhook.agent,
            "received_at": now.isoformat(),
            "headers": {k: v for k, v in request.headers.items() if k.lower() != "authorization"},
            "payload": payload_data,
        }
        (inbox_dir / filename).write_text(json.dumps(envelope, indent=2, default=str))

        return JSONResponse({"status": "accepted", "file": filename}, status_code=202)

    async def _serve_ui(self, request: Request) -> Response:
        ui_path = WEB_DIR / "index.html"
        if not ui_path.exists():
            return JSONResponse({"error": "web UI not found"}, status_code=404)
        return HTMLResponse(ui_path.read_text(), headers={"Cache-Control": "no-cache"})

    async def start(self):
        import uvicorn

        config = uvicorn.Config(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info",
        )
        self._server = uvicorn.Server(config)
        logger.info("HTTP API listening on http://%s:%d", self.config.host, self.config.port)
        await self._server.serve()

    async def stop(self):
        if self._server:
            self._server.should_exit = True
