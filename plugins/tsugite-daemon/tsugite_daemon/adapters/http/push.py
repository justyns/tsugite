"""PushMixin: push HTTP handlers for HTTPServer (split from adapters/http.py)."""

from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

if TYPE_CHECKING:
    pass


class PushMixin:
    def _push_routes(self) -> list:
        return [
            Route("/api/push/vapid-key", self._push_vapid_key, methods=["GET"]),
            Route("/api/push/subscribe", self._push_subscribe, methods=["POST"]),
            Route("/api/push/unsubscribe", self._push_unsubscribe, methods=["POST"]),
        ]

    async def _push_vapid_key(self, request: Request) -> JSONResponse:
        if not self.vapid_public_key:
            return JSONResponse({"error": "web push not configured"}, status_code=404)
        return JSONResponse({"public_key": self.vapid_public_key})

    async def _push_subscribe(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        if not self.push_store:
            return JSONResponse({"error": "web push not configured"}, status_code=404)
        body = await request.json()
        if not body.get("endpoint"):
            return JSONResponse({"error": "missing endpoint"}, status_code=400)
        self.push_store.subscribe(body)
        return JSONResponse({"status": "subscribed"})

    async def _push_unsubscribe(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        if not self.push_store:
            return JSONResponse({"error": "web push not configured"}, status_code=404)
        body = await request.json()
        endpoint = body.get("endpoint")
        if not endpoint:
            return JSONResponse({"error": "missing endpoint"}, status_code=400)
        self.push_store.unsubscribe(endpoint)
        return JSONResponse({"status": "unsubscribed"})
