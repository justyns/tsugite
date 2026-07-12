"""WebhooksMixin: webhooks HTTP handlers for HTTPServer (split from adapters/http.py)."""

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from tsugite_daemon.adapters.http.helpers import (
    MAX_WEBHOOK_BODY,
    logger,
)

if TYPE_CHECKING:
    pass


class WebhooksMixin:
    def _webhook_routes(self) -> list:
        return [
            Route("/api/webhooks", self._list_webhooks, methods=["GET"]),
            Route("/api/webhooks", self._create_webhook, methods=["POST"]),
            Route("/api/webhooks/{token}", self._delete_webhook, methods=["DELETE"]),
            Route("/webhook/{token}", self._webhook, methods=["POST"]),
        ]

    async def _list_webhooks(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        webhooks = [
            {"token": w.token, "agent": w.agent, "source": w.source, "created_at": w.created_at}
            for w in self.webhook_store.list()
        ]
        return JSONResponse({"webhooks": webhooks})

    async def _create_webhook(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        agent = body.get("agent", "")
        source = body.get("source", "")
        if not agent or not source:
            return JSONResponse({"error": "agent and source are required"}, status_code=400)
        # The source lands in inbox filenames - restrict it to a safe slug so a
        # slash/.. can't escape (or break) the inbox directory on delivery.
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", source):
            return JSONResponse(
                {"error": "source must be 1-64 chars of letters, digits, dot, underscore, or dash"},
                status_code=400,
            )
        if agent not in self.agent_configs:
            return JSONResponse({"error": f"unknown agent: {agent}"}, status_code=400)

        try:
            entry = self.webhook_store.add(agent=agent, source=source, token=body.get("token"))
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

        return JSONResponse(
            {
                "token": entry.token,
                "agent": entry.agent,
                "source": entry.source,
                "created_at": entry.created_at,
            },
            status_code=201,
        )

    async def _delete_webhook(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        token = request.path_params["token"]
        try:
            self.webhook_store.remove(token)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        return JSONResponse({"status": "removed"})

    async def _webhook(self, request: Request) -> JSONResponse:
        token = request.path_params["token"]
        client_ip = request.client.host if request.client else "unknown"
        webhook = self.webhook_store.get(token)
        if not webhook:
            logger.warning("Webhook rejected: invalid token [%s] from %s", token[:8], client_ip)
            return JSONResponse({"error": "invalid webhook token"}, status_code=404)

        logger.info("Received webhook [%s] from %s", token[:8], client_ip)

        if webhook.agent not in self.agent_configs:
            return JSONResponse({"error": "webhook agent not configured"}, status_code=500)

        try:
            raw = await request.body()
        except Exception:
            return JSONResponse({"error": "failed to read body"}, status_code=400)
        if len(raw) > MAX_WEBHOOK_BODY:
            return JSONResponse({"error": "payload too large"}, status_code=413)

        try:
            payload_data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload_data = raw.decode("utf-8", errors="replace")

        event_type = ""
        if isinstance(payload_data, dict):
            event_type = payload_data.get("event") or payload_data.get("type") or payload_data.get("action") or ""
        logger.info(
            "Webhook [%s] source: %s | event: %s | agent: %s",
            token[:8],
            webhook.source,
            event_type or "unknown",
            webhook.agent,
        )

        agent_config = self.agent_configs[webhook.agent]
        inbox_dir = agent_config.workspace_dir / "inbox" / "webhooks"
        inbox_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        # Sanitize defensively (pre-validation legacy records may carry anything)
        # and suffix a nonce: bursty sources deliver multiple events in the same
        # second, and write_text would silently clobber the earlier file.
        safe_source = re.sub(r"[^A-Za-z0-9._-]", "_", webhook.source)[:64] or "webhook"
        filename = f"{now.strftime('%Y%m%dT%H%M%S')}-{safe_source}-{uuid4().hex[:8]}.json"
        envelope = {
            "source": webhook.source,
            "agent": webhook.agent,
            "received_at": now.isoformat(),
            "headers": {k: v for k, v in request.headers.items() if k.lower() != "authorization"},
            "payload": payload_data,
        }
        # Off the event loop: a slow disk write here would stall every SSE
        # stream and request in the daemon.
        await asyncio.to_thread((inbox_dir / filename).write_text, json.dumps(envelope, indent=2, default=str))
        logger.info("Webhook [%s] saved to inbox: %s", token[:8], filename)

        return JSONResponse({"status": "accepted", "file": filename}, status_code=202)
