"""JobsMixin: jobs HTTP handlers for HTTPServer."""

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from tsugite_daemon.adapters.http.helpers import mounted_api_routes
from tsugite_daemon.job_store import JobState, JobStateTransitionError, UnknownJobError

_JOB_ERROR_STATUS: dict[type[Exception], int] = {
    UnknownJobError: 404,
    JobStateTransitionError: 409,
    ValueError: 400,
    RuntimeError: 500,
}


def _job_error_response(exc: Exception) -> JSONResponse | None:
    """The response for the nearest mapped class on the exception's MRO."""
    for cls in type(exc).__mro__:
        status = _JOB_ERROR_STATUS.get(cls)
        if status is not None:
            return JSONResponse({"error": str(exc)}, status_code=status)
    return None


class JobsMixin:
    def _job_routes(self) -> list:
        return [
            *mounted_api_routes(
                "/api/jobs",
                "jobs",
                [
                    Route("/", self._api_list_jobs, methods=["GET"]),
                    Route("/", self._api_create_job, methods=["POST"]),
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
        'active' (= queued + running + verifying), 'resolved' (= done + cancelled), or
        'open' (everything not resolved). ?limit=N caps the response
        (default 100; 0 = unlimited)."""
        if err := self._check_auth(request):
            return err
        if not self.job_store:
            return JSONResponse({"error": "jobs orchestrator not available"}, status_code=503)
        jobs = self.job_store.list_all()
        state_filter = request.query_params.get("state")
        if state_filter:
            alias = {
                "stuck": frozenset({"stuck", "errored", "awaiting_input"}),
                "active": frozenset({"queued", "running", "verifying"}),
                "resolved": frozenset({"done", "cancelled"}),
            }
            alias["open"] = frozenset(state.value for state in JobState) - alias["resolved"]
            states = frozenset(state.value for state in JobState)
            if state_filter not in alias and state_filter not in states:
                # Filtering for it would answer 200 with an empty list, which reads
                # as "no jobs in that state" rather than "no such state".
                known = ", ".join(sorted(states | set(alias)))
                return JSONResponse(
                    {"error": f"unknown state {state_filter!r} (expected one of: {known})"},
                    status_code=400,
                )
            allowed = alias.get(state_filter, frozenset({state_filter}))
            jobs = [j for j in jobs if j.state in allowed]
        try:
            limit = int(request.query_params.get("limit", "100"))
        except ValueError:
            limit = 100
        if limit > 0:
            jobs = jobs[:limit]
        return JSONResponse({"jobs": [j.to_payload() for j in jobs]})

    async def _api_create_job(self, request: Request) -> JSONResponse:
        """Structured job creation. Parses the same fields as the `/job` slash
        command (commands.cmd_job), calls JobsOrchestrator.create_and_start_job
        directly, and returns 201 with job.to_payload() so the caller gets the
        job_id synchronously - unlike POST /api/agents/{agent}/commands/job,
        which returns only a free-text confirmation string.

        Always provisions a fresh host session for the tile (no session_id field);
        the anchor logic lives in the shared cmd_job helper.
        """
        if err := self._require_auth_and_jobs(request):
            return err
        body = await self._optional_json_body(request)
        adapter = self.adapter
        if adapter is None:
            return JSONResponse({"error": "daemon runtime unavailable"}, status_code=503)
        user_id = (body.get("user_id") or "").strip()
        if not user_id:
            return JSONResponse({"error": "user_id is required"}, status_code=400)
        task = (body.get("task") or "").strip()
        if not task:
            return JSONResponse({"error": "task is required"}, status_code=400)

        from tsugite_daemon.commands import create_job_host_session, parse_acceptance_criteria

        parent_session_id = create_job_host_session(adapter, user_id, task)
        model_ladder = body.get("model_ladder") or None
        if isinstance(model_ladder, str):
            model_ladder = model_ladder.split("|")
        try:
            job, _started = await self.jobs_orchestrator.create_and_start_job(
                parent_session_id=parent_session_id,
                prompt=task,
                acceptance_criteria=parse_acceptance_criteria(body.get("acceptance_criteria")),
                repo=body.get("repo") or None,
                model=body.get("model") or None,
                model_ladder=model_ladder,
                agent=body.get("agent") or None,
                timeout_minutes=body.get("timeout_minutes") or 30,
                max_attempts=body.get("max_attempts"),
                notify_when=body.get("notify_when") or None,
                spawned_by="user-slash",
                executor=(body.get("executor") or "agent").strip() or "agent",
                effort=body.get("effort") or None,
            )
        except Exception as e:
            # Unmapped failures are spawn errors.
            return _job_error_response(e) or JSONResponse({"error": str(e)}, status_code=500)
        return JSONResponse(job.to_payload(), status_code=201)

    async def _api_cancel_job(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_jobs(request):
            return err
        job_id = request.path_params["job_id"]
        body = await self._optional_json_body(request)
        reason = body.get("reason") or "cancelled by user"
        try:
            await self.jobs_orchestrator.cancel_job(job_id, reason=reason)
        except Exception as e:
            if (resp := _job_error_response(e)) is None:
                raise
            return resp
        return JSONResponse({"status": "cancelled"})

    async def _api_mark_job_done(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_jobs(request):
            return err
        job_id = request.path_params["job_id"]
        body = await self._optional_json_body(request)
        reason = body.get("reason") or "marked done by user"
        try:
            await self.jobs_orchestrator.mark_done_manual(job_id, reason=reason)
        except Exception as e:
            if (resp := _job_error_response(e)) is None:
                raise
            return resp
        return JSONResponse({"status": "done"})

    async def _api_retry_job(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_jobs(request):
            return err
        job_id = request.path_params["job_id"]
        body = await self._optional_json_body(request)
        hint = (body.get("hint") or "").strip()
        model = (body.get("model") or "").strip() or None
        verifier_model = (body.get("verifier_model") or "").strip() or None
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
        except Exception as e:
            if (resp := _job_error_response(e)) is None:
                raise
            return resp
        return JSONResponse({"status": "running"})
