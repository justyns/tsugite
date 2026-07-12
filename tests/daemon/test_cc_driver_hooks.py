"""cc-driver hook decisions (pure) + the public hook route dispatch.

Route tests run a TestClient over Starlette(routes=adapter.get_public_http_routes())
with a fake orchestrator recording complete_worker/fail_worker. Payload shapes match
hook-payloads-reference.md.
"""

from starlette.applications import Starlette
from starlette.testclient import TestClient
from tsugite_cc_driver.adapter import CCDriverAdapter, CCDriverConfig
from tsugite_cc_driver.hooks import (
    build_initial_prompt,
    decide_stop,
    decide_stop_failure,
    notification_attention,
)

MARKER = "CCDRIVER_GOAL_COMPLETE"
NEED_INPUT = "CCDRIVER_NEED_INPUT"


# ── fakes ──


class FakeJob:
    def __init__(self, job_id, state="running"):
        self.id = job_id
        self.state = state
        self.parent_session_id = "parent-1"


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
        self.paused = []
        self.resumed = []

    def get_job(self, job_id):
        return self._jobs.get(job_id)

    def attach_worker_terminal(self, job_id, terminal_id):
        self._jobs.update(job_id, worker_terminal_id=terminal_id)

    async def complete_worker(self, job_id, summary):
        self.completed.append((job_id, summary))

    async def fail_worker(self, job_id, error):
        self.failed.append((job_id, error))

    async def pause_worker(self, job_id, question):
        self.paused.append((job_id, question))

    async def resume_worker(self, job_id):
        self.resumed.append(job_id)
        return self._jobs.update(job_id, state="running")


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


def test_decide_stop_need_input_marker_pauses():
    d = decide_stop(
        _stop_payload(last_assistant_message=f"I can't guess this.\n{NEED_INPUT}: what is the codeword?"),
        consecutive_continues=1,
        max_consecutive_continues=5,
        completion_marker=MARKER,
    )
    assert d.complete is False
    assert d.needs_input == "what is the codeword?"
    assert d.response == {}, "a needs-input pause must let claude stop, not nudge it to guess"


def test_decide_stop_completion_marker_wins_over_need_input():
    d = decide_stop(
        _stop_payload(last_assistant_message=f"all done {MARKER}\n{NEED_INPUT}: stale question"),
        consecutive_continues=0,
        max_consecutive_continues=5,
        completion_marker=MARKER,
    )
    assert d.complete is True
    assert d.needs_input is None


def test_decide_stop_need_input_checked_before_budget_exhaustion():
    # At the cap AND asking for input: must pause, not force-complete into a
    # verification the worker already knows it can't pass.
    d = decide_stop(
        _stop_payload(last_assistant_message=f"{NEED_INPUT}: which environment should I target?"),
        consecutive_continues=5,
        max_consecutive_continues=5,
        completion_marker=MARKER,
    )
    assert d.complete is False, "needs-input at the nudge cap must pause, not complete"
    assert d.needs_input == "which environment should I target?"


def test_decide_stop_bare_need_input_marker_still_pauses():
    d = decide_stop(
        _stop_payload(last_assistant_message=f"blocked. {NEED_INPUT}:"),
        consecutive_continues=0,
        max_consecutive_continues=5,
        completion_marker=MARKER,
    )
    assert d.complete is False
    assert d.needs_input, "a bare marker with no question still needs a non-empty question for the notify"


def test_initial_prompt_teaches_both_markers():
    text = build_initial_prompt("do the thing", MARKER, needs_input_marker=NEED_INPUT)
    assert MARKER in text
    assert NEED_INPUT in text, "the worker can only ask for input if the protocol is in its prompt"


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


def test_route_stop_need_input_pauses_worker(tmp_path):
    adapter, orch = _wired_adapter(state_dir=str(tmp_path))
    _seed(adapter, orch)
    resp = _client(adapter).post(
        "/hook/tok-1", json=_stop_payload(last_assistant_message=f"{NEED_INPUT}: what is the codeword?")
    )
    assert resp.status_code == 200
    assert resp.json() == {}, "the pause must let claude stop"
    assert orch.paused == [("job-1", "what is the codeword?")]
    assert orch.completed == [] and orch.failed == []


def test_route_stop_on_awaiting_input_job_resumes_and_grades(tmp_path):
    # A human typed the answer straight into the live TUI: claude's next Stop
    # arrives while the job is paused. That turn must resume the job and be
    # graded normally (here: the completion marker ends the attempt) instead of
    # being dropped by the not-RUNNING guard.
    adapter, orch = _wired_adapter(state_dir=str(tmp_path))
    _seed(adapter, orch, state="awaiting_input")
    resp = _client(adapter).post(
        "/hook/tok-1",
        json=_stop_payload(stop_hook_active=False, last_assistant_message=f"wrote the file {MARKER}"),
    )
    assert resp.status_code == 200
    assert orch.resumed == ["job-1"], "a Stop during the pause is a human take-over turn - it must resume"
    assert orch.completed and orch.completed[0][0] == "job-1"


def test_route_non_stop_event_on_awaiting_input_job_does_not_resume(tmp_path):
    adapter, orch = _wired_adapter(state_dir=str(tmp_path))
    _seed(adapter, orch, state="awaiting_input")
    resp = _client(adapter).post(
        "/hook/tok-1",
        json={"hook_event_name": "Notification", "notification_type": "permission_prompt", "message": "ok?"},
    )
    assert resp.status_code == 200
    assert orch.resumed == [], "only a Stop (a finished turn) proves someone answered in the TUI"


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
    assert adapter.event_bus.emitted == [
        ("needs_attention", {"job_id": "job-1", "parent_session_id": "parent-1", "message": "approve?"})
    ], "the emit must carry parent_session_id so the UI can flag the owning session"


def test_route_stop_after_permission_prompt_emits_attention_cleared(tmp_path):
    # The only signal that a permission prompt was answered is the next Stop
    # (claude finished the turn) - the UI needs it to clear the persistent
    # "needs your input" marker the prompt set.
    adapter, orch = _wired_adapter(state_dir=str(tmp_path))
    _seed(adapter, orch)
    client = _client(adapter)
    client.post(
        "/hook/tok-1",
        json={"hook_event_name": "Notification", "notification_type": "permission_prompt", "message": "approve?"},
    )
    client.post("/hook/tok-1", json=_stop_payload())
    cleared = [e for e in adapter.event_bus.emitted if e[0] == "attention_cleared"]
    assert cleared == [("attention_cleared", {"job_id": "job-1", "parent_session_id": "parent-1"})]

    # A Stop with no prior prompt must NOT spam attention_cleared.
    adapter.event_bus.emitted.clear()
    client.post("/hook/tok-1", json=_stop_payload())
    assert [e for e in adapter.event_bus.emitted if e[0] == "attention_cleared"] == []
