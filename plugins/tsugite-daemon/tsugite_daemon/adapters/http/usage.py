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
                    Route("/schedules", self._usage_schedules, methods=["GET"]),
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

    async def _usage_schedules(self, request: Request) -> JSONResponse:
        """Per-schedule usage breakdown for the Usage tab's scheduled-tasks table.

        One row per schedule (schedule_name, runs, total_tokens, total_cost,
        cache_creation_tokens, cache_read_tokens, last_run). Scheduled runs
        recorded before schedule attribution existed have schedule_name=null and
        aggregate into one unattributed bucket. Cache token counts are honest
        SUMs; a provider that doesn't report cache usage contributes 0, which is
        indistinguishable from a genuine zero.
        """
        if err := self._check_auth(request):
            return err
        store = self._get_usage_store()
        since = request.query_params.get("since")
        # Default high: this is a full per-schedule breakdown, not a top-N
        # leaderboard like agents/models. A user's schedule set is small.
        limit = self._parse_limit(request, default=100)
        return JSONResponse(store.by_schedule(since=since, limit=limit))
