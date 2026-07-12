"""cc-driver hook decisions (pure) + the public hook route dispatch.

Route tests run a TestClient over Starlette(routes=adapter.get_public_http_routes())
with a fake orchestrator recording complete_worker/fail_worker. Payload shapes match
hook-payloads-reference.md.
"""

from starlette.applications import Starlette
from starlette.testclient import TestClient
from tsugite_cc_driver.adapter import CCDriverAdapter, CCDriverConfig
from tsugite_cc_driver.hooks import (
    decide_stop,
    decide_stop_failure,
    notification_attention,
)

MARKER = "CCDRIVER_GOAL_COMPLETE"


# ── fakes ──


class FakeJob:
    def __init__(self, job_id, state="running"):
        self.id = job_id
        self.state = state


class FakeJobStore:
    def __init__(self):
        self._jobs = {}

    def add(self, job):
        self._jobs[job.id] = job

    def get(self, job_id):
        return self._jobs.get(job_id)

    def update(self, job_id, **fields):
        job = self._jobs.get(job_id)
        for k, v in fields.items():
            setattr(job, k, v)
        return job


class FakeOrchestrator:
    def __init__(self):
        self._jobs = FakeJobStore()
        self.completed = []
        self.failed = []

    def get_job(self, job_id):
        return self._jobs.get(job_id)

    def attach_worker_terminal(self, job_id, terminal_id):
        self._jobs.update(job_id, worker_terminal_id=terminal_id)

    async def complete_worker(self, job_id, summary):
        self.completed.append((job_id, summary))

    async def fail_worker(self, job_id, error):
        self.failed.append((job_id, error))


class FakeBus:
    def __init__(self):
        self.emitted = []

    def emit(self, event, payload):
        self.emitted.append((event, payload))


def _wired_adapter(**cfg):
    adapter = CCDriverAdapter(CCDriverConfig(**cfg))
    orch = FakeOrchestrator()
    adapter.set_jobs_orchestrator(orch)
    adapter.event_bus = FakeBus()
    return adapter, orch


def _seed(adapter, orch, *, job_id="job-1", token="tok-1", state="running", continues=0):
    orch._jobs.add(FakeJob(job_id, state=state))
    st = adapter._drive_state.create(job_id, token)
    st.terminal_id = "term-1"
    st.consecutive_continues = continues
    return st


def _client(adapter):
    return TestClient(Starlette(routes=adapter.get_public_http_routes()))


def _stop_payload(**over):
    base = {
        "session_id": "cc-sess-1",
        "transcript_path": "/tmp/t.jsonl",
        "cwd": "/work",
        "hook_event_name": "Stop",
        "stop_hook_active": True,
        "last_assistant_message": "still working on it",
    }
    base.update(over)
    return base


# ── pure decide_stop ──


def test_decide_stop_marker_completes():
    d = decide_stop(
        _stop_payload(last_assistant_message=f"all done {MARKER}"),
        consecutive_continues=0,
        max_consecutive_continues=5,
        completion_marker=MARKER,
    )
    assert d.complete is True
    assert MARKER in d.summary
    assert d.response == {}, "a completed attempt must let claude stop"


def test_decide_stop_under_budget_blocks_with_exact_json():
    d = decide_stop(
        _stop_payload(),
        consecutive_continues=1,
        max_consecutive_continues=5,
        completion_marker=MARKER,
    )
    assert d.complete is False
    assert d.new_consecutive_continues == 2
    assert d.response["decision"] == "block"
    hso = d.response["hookSpecificOutput"]
    assert hso["hookEventName"] == "Stop"
    assert MARKER in hso["additionalContext"]
    assert "exactly" not in hso["additionalContext"].lower(), "must not be injection-shaped"


def test_decide_stop_budget_exhausted_completes():
    d = decide_stop(
        _stop_payload(),
        consecutive_continues=5,
        max_consecutive_continues=5,
        completion_marker=MARKER,
    )
    assert d.complete is True, "an exhausted continue budget ends the attempt for the verifier to judge"
    assert d.response == {}


def test_decide_stop_fresh_human_turn_resets_budget():
    # At the cap, but stop_hook_active=False (a human/CLI turn) resets to 0 -> block, not complete.
    d = decide_stop(
        _stop_payload(stop_hook_active=False),
        consecutive_continues=5,
        max_consecutive_continues=5,
        completion_marker=MARKER,
    )
    assert d.complete is False, "a fresh human turn must reset the continue budget"
    assert d.new_consecutive_continues == 1


def test_decide_stop_failure_extracts_reason():
    assert decide_stop_failure({"error": "API 500"}) == "API 500"
    assert "Stop failure" in decide_stop_failure({})


def test_notification_attention_only_for_permission_prompt():
    assert notification_attention({"notification_type": "permission_prompt", "message": "approve?"}) == "approve?"
    assert notification_attention({"notification_type": "other"}) is None


# ── route dispatch ──


def test_route_unknown_token_404(tmp_path):
    adapter, _orch = _wired_adapter(state_dir=str(tmp_path))
    resp = _client(adapter).post("/hook/nope", json=_stop_payload())
    assert resp.status_code == 404


def test_route_non_running_job_returns_empty(tmp_path):
    adapter, orch = _wired_adapter(state_dir=str(tmp_path))
    _seed(adapter, orch, state="done")
    resp = _client(adapter).post("/hook/tok-1", json=_stop_payload())
    assert resp.status_code == 200
    assert resp.json() == {}
    assert orch.completed == [] and orch.failed == []


def test_route_stop_marker_calls_complete_worker(tmp_path):
    adapter, orch = _wired_adapter(state_dir=str(tmp_path))
    _seed(adapter, orch)
    resp = _client(adapter).post("/hook/tok-1", json=_stop_payload(last_assistant_message=f"done {MARKER}"))
    assert resp.status_code == 200
    assert resp.json() == {}
    assert orch.completed == [("job-1", f"done {MARKER}")]


def test_route_stop_block_does_not_complete_and_increments(tmp_path):
    adapter, orch = _wired_adapter(state_dir=str(tmp_path))
    st = _seed(adapter, orch, continues=0)
    resp = _client(adapter).post("/hook/tok-1", json=_stop_payload())
    body = resp.json()
    assert body["decision"] == "block"
    assert orch.completed == []
    assert st.consecutive_continues == 1, "route must persist the incremented budget onto DriveState"


def test_route_stop_records_cc_session_id(tmp_path):
    adapter, orch = _wired_adapter(state_dir=str(tmp_path))
    st = _seed(adapter, orch)
    _client(adapter).post("/hook/tok-1", json=_stop_payload(session_id="cc-xyz", transcript_path="/t/x.jsonl"))
    assert st.cc_session_id == "cc-xyz"
    assert st.transcript_path == "/t/x.jsonl"


def test_route_stop_failure_calls_fail_worker(tmp_path):
    adapter, orch = _wired_adapter(state_dir=str(tmp_path))
    _seed(adapter, orch)
    resp = _client(adapter).post("/hook/tok-1", json={"hook_event_name": "StopFailure", "error": "model overloaded"})
    assert resp.status_code == 200
    assert orch.failed == [("job-1", "model overloaded")]


def test_route_notification_permission_prompt_emits_needs_attention(tmp_path):
    adapter, orch = _wired_adapter(state_dir=str(tmp_path))
    _seed(adapter, orch)
    _client(adapter).post(
        "/hook/tok-1",
        json={"hook_event_name": "Notification", "notification_type": "permission_prompt", "message": "approve?"},
    )
    assert adapter.event_bus.emitted == [("needs_attention", {"job_id": "job-1", "message": "approve?"})]
