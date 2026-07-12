"""UsageMixin: usage HTTP handlers for HTTPServer."""

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route


class UsageMixin:
    def _usage_routes(self) -> list:
        return [
            Mount(
                "/api/usage",
                name="usage",
                routes=[
                    Route("/summary", self._usage_summary, methods=["GET"]),
                    Route("/agents", self._usage_agents, methods=["GET"]),
                    Route("/models", self._usage_models, methods=["GET"]),
                    Route("/total", self._usage_total, methods=["GET"]),
                ],
            ),
        ]

    def _get_usage_store(self):
        from tsugite.usage import get_usage_store

        return get_usage_store()

    def _parse_limit(self, request: Request, default: int = 10, cap: int = 100) -> int:
        try:
            return max(1, min(int(request.query_params.get("limit", str(default))), cap))
        except ValueError:
            return default

    async def _usage_summary(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        store = self._get_usage_store()
        period = request.query_params.get("period", "day")
        since = request.query_params.get("since")
        agent = request.query_params.get("agent")
        return JSONResponse(store.summary(agent=agent, period=period, since=since))

    async def _usage_agents(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        store = self._get_usage_store()
        since = request.query_params.get("since")
        limit = self._parse_limit(request)
        return JSONResponse(store.top_agents(since=since, limit=limit))

    async def _usage_models(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        store = self._get_usage_store()
        since = request.query_params.get("since")
        limit = self._parse_limit(request)
        return JSONResponse(store.top_models(since=since, limit=limit))

    async def _usage_total(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        store = self._get_usage_store()
        since = request.query_params.get("since")
        return JSONResponse(store.total(since=since))
