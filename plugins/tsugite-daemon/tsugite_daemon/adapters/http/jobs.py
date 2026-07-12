"""JobsMixin: jobs HTTP handlers for HTTPServer (split from adapters/http.py)."""

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route


class JobsMixin:
    def _job_routes(self) -> list:
        return [
            Mount(
                "/api/jobs",
                name="jobs",
                routes=[
                    Route("/", self._api_list_jobs, methods=["GET"]),
                    Route("/{job_id}/cancel", self._api_cancel_job, methods=["POST"]),
                    Route("/{job_id}/mark-done", self._api_mark_job_done, methods=["POST"]),
                    Route("/{job_id}/retry", self._api_retry_job, methods=["POST"]),
                ],
            ),
            # Top-level (not under the /api/jobs mount): the new-job modal fetches
            # this to decide whether to show its executor dropdown.
            Route("/api/executors", self._api_list_executors, methods=["GET"]),
        ]

    async def _api_list_executors(self, request: Request) -> JSONResponse:
        """Job executors the new-job modal can pick from: the built-in "agent"
        plus any non-agent executors a plugin registered (e.g. "cc")."""
        if err := self._check_auth(request):
            return err
        registered = self.jobs_orchestrator.executor_names if self.jobs_orchestrator is not None else []
        return JSONResponse({"executors": ["agent", *registered]})

    async def _api_list_jobs(self, request: Request) -> JSONResponse:
        """Return Jobs for the Jobs tab, newest first. Optional ?state=<state>
        filter accepts a real Job state, the alias 'stuck' (= stuck + errored),
        or 'active' (= running + verifying). ?limit=N caps the response
        (default 100; 0 = unlimited)."""
        if err := self._check_auth(request):
            return err
        if not self.job_store:
            return JSONResponse({"error": "jobs orchestrator not available"}, status_code=503)
        jobs = self.job_store.list_all()
        state_filter = request.query_params.get("state")
        if state_filter:
            alias = {
                # The "needs you" set the tab badge counts: parked jobs AND ones
                # paused on a question.
                "stuck": frozenset({"stuck", "errored", "awaiting_input"}),
                "active": frozenset({"running", "verifying"}),
                "resolved": frozenset({"done", "cancelled"}),
            }
            allowed = alias.get(state_filter, frozenset({state_filter}))
            jobs = [j for j in jobs if j.state in allowed]
        try:
            limit = int(request.query_params.get("limit", "100"))
        except ValueError:
            limit = 100
        if limit > 0:
            jobs = jobs[:limit]
        return JSONResponse({"jobs": [j.to_payload() for j in jobs]})

    async def _api_cancel_job(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_jobs(request):
            return err
        job_id = request.path_params["job_id"]
        body = await self._optional_json_body(request)
        reason = body.get("reason") or "cancelled by user"
        try:
            await self.jobs_orchestrator.cancel_job(job_id, reason=reason)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        return JSONResponse({"status": "cancelled"})

    async def _api_mark_job_done(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_jobs(request):
            return err
        job_id = request.path_params["job_id"]
        body = await self._optional_json_body(request)
        reason = body.get("reason") or "marked done by user"
        try:
            await self.jobs_orchestrator.mark_done_manual(job_id, reason=reason)
        except ValueError as e:
            # 404 if unknown, 409 if not in STUCK state.
            status = 404 if "Unknown job" in str(e) else 409
            return JSONResponse({"error": str(e)}, status_code=status)
        return JSONResponse({"status": "done"})

    async def _api_retry_job(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_jobs(request):
            return err
        job_id = request.path_params["job_id"]
        body = await self._optional_json_body(request)
        hint = (body.get("hint") or "").strip()
        model = (body.get("model") or "").strip() or None
        verifier_model = (body.get("verifier_model") or "").strip() or None
        if not hint and not model:
            # Retrying purely to switch models is legitimate (usage-limit death);
            # an unchanged retry with neither is a no-op repeat.
            return JSONResponse({"error": "hint or model is required"}, status_code=400)
        reset_counter = bool(body.get("reset_counter", False))
        fresh_workspace = bool(body.get("fresh_workspace", False))
        try:
            await self.jobs_orchestrator.retry_with_hint(
                job_id,
                hint=hint,
                reset_counter=reset_counter,
                fresh_workspace=fresh_workspace,
                model=model,
                verifier_model=verifier_model,
            )
        except ValueError as e:
            status = 404 if "Unknown job" in str(e) else 409
            return JSONResponse({"error": str(e)}, status_code=status)
        return JSONResponse({"status": "running"})
