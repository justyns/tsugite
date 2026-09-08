"""PushMixin: push HTTP handlers for HTTPServer."""

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route


class PushMixin:
    def _push_routes(self) -> list:
        return [
            Mount(
                "/api/push",
                name="push",
                routes=[
                    Route("/vapid-key", self._push_vapid_key, methods=["GET"]),
                    Route("/subscribe", self._push_subscribe, methods=["POST"]),
                    Route("/unsubscribe", self._push_unsubscribe, methods=["POST"]),
                ],
            ),
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
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        if not body.get("endpoint"):
            return JSONResponse({"error": "missing endpoint"}, status_code=400)
        self.push_store.subscribe(body)
        return JSONResponse({"status": "subscribed"})

    async def _push_unsubscribe(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        if not self.push_store:
            return JSONResponse({"error": "web push not configured"}, status_code=404)
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        endpoint = body.get("endpoint")
        if not endpoint:
            return JSONResponse({"error": "missing endpoint"}, status_code=400)
        self.push_store.unsubscribe(endpoint)
        return JSONResponse({"status": "unsubscribed"})
