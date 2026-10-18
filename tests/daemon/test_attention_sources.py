"""The four things that can wait on the user, and the records they open.

A needs-ack delivery, a blocking `ask_user`, a job parked in
awaiting_input/stuck/errored, and a send that failed after the turn ended each
open a record while they wait and clear it when they stop. Delivery and job
records outlive a daemon restart; an ask cannot, because the blocked call it
belonged to is gone.
"""

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from tsugite_daemon.adapters.http.sse import HTTPInteractionBackend, SSEProgressHandler
from tsugite_daemon.adapters.scheduler_adapter import SchedulerAdapter
from tsugite_daemon.attention_store import OWNER_SESSION, SOURCE_ASK, SOURCE_DELIVERY, SOURCE_ERROR, SOURCE_JOB
from tsugite_daemon.job_store import Job, JobState, JobStore
from tsugite_daemon.jobs_orchestrator import JobsOrchestrator
from tsugite_daemon.scheduler import ScheduleEntry
from tsugite_daemon.session_runner import SessionRunner
from tsugite_daemon.session_store import (
    METADATA_ORIGINATING_SESSION,
    Session,
    SessionSource,
    SessionStatus,
    SessionStore,
    attention_fields,
)

from tsugite.history.models import Event
from tsugite.history.reconstruction import events_to_messages


class _Bus:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def emit(self, name, payload):
        self.events.append((name, payload))

    def of(self, name: str, action: str | None = None) -> list[dict]:
        return [p for n, p in self.events if n == name and (action is None or p.get("action") == action)]


@pytest.fixture
def store(tmp_path, history_dir):
    store = SessionStore(tmp_path / "store.json")
    store.agents.ensure_default(tmp_path, "default")
    return store


@pytest.fixture
def bus():
    return _Bus()


@pytest.fixture
def runner(store, bus):
    return SessionRunner(store=store, adapter=None, event_bus=bus)


def _session(store: SessionStore, sid: str = "s1") -> str:
    store.create_session(Session(id=sid, source=SessionSource.INTERACTIVE.value, user_id="alice"))
    return sid


def _sources(store: SessionStore, sid: str) -> list[str]:
    return sorted(r.source for r in store.attention.open_records(sid))


class TestDeliverySource:
    def test_a_needs_ack_delivery_opens_a_record(self, store, runner):
        sid = _session(store)

        runner.deliver_to_session(sid, "approve?", source="job", kind="needs_ack", title="Approve")

        records = store.attention.open_records(sid)
        assert [r.source for r in records] == [SOURCE_DELIVERY]
        assert records[0].kind == "needs_ack"
        assert records[0].kind == "needs_ack"
        assert records[0].ref_id == store.get_session(sid).pending_delivery_ids[0]

    def test_an_fyi_delivery_opens_nothing(self, store, runner):
        sid = _session(store)

        runner.deliver_to_session(sid, "rent is due", source="schedule", kind="fyi")

        assert store.attention.open_records(sid) == []

    def test_acknowledging_one_card_clears_only_its_record(self, store, runner):
        sid = _session(store)
        runner.deliver_to_session(sid, "first", source="job", kind="needs_ack")
        runner.deliver_to_session(sid, "second", source="job", kind="needs_ack")
        first, second = store.get_session(sid).pending_delivery_ids

        runner.clear_attention(sid, first)

        assert [r.ref_id for r in store.attention.open_records(sid)] == [second]

    def test_dismissing_everything_clears_every_delivery_record(self, store, runner):
        sid = _session(store)
        runner.deliver_to_session(sid, "first", source="job", kind="needs_ack")
        runner.deliver_to_session(sid, "second", source="job", kind="needs_ack")

        runner.clear_attention(sid)

        assert store.attention.open_records(sid) == []

    def test_a_compaction_moves_the_record_to_the_successor(self, store, runner):
        sid = _session(store)
        runner.deliver_to_session(sid, "approve?", source="job", kind="needs_ack")

        successor = store.compact_session(sid)

        assert store.attention.open_records(sid) == []
        assert [r.source for r in store.attention.open_records(successor.id)] == [SOURCE_DELIVERY]

    def test_a_delivery_record_survives_a_restart(self, tmp_path, history_dir, store, runner):
        sid = _session(store)
        runner.deliver_to_session(sid, "approve?", source="job", kind="needs_ack")

        reopened = SessionStore(tmp_path / "store.json")

        assert [r.source for r in reopened.attention.open_records(sid)] == [SOURCE_DELIVERY]


class TestAskSource:
    def _backend(self, runner, sid):
        progress = SSEProgressHandler()
        progress.set_session_id(sid)
        return HTTPInteractionBackend(progress, session_runner=runner, session_id=sid)

    def test_a_blocking_ask_opens_a_record_and_answering_clears_it(self, store, runner):
        sid = _session(store)
        backend = self._backend(runner, sid)
        seen: dict = {}

        def answer():
            end = time.monotonic() + 5
            while time.monotonic() < end and not store.attention.open_records(sid):
                time.sleep(0.01)
            seen["while_blocked"] = store.attention.open_records(sid)
            backend.submit_response("main")

        t = threading.Thread(target=answer)
        t.start()
        backend.ask_user("Which branch?", "text")
        t.join()

        assert [r.source for r in seen["while_blocked"]] == [SOURCE_ASK]
        assert seen["while_blocked"][0].kind == "needs_answer"
        assert seen["while_blocked"][0].kind == "needs_answer"
        assert store.attention.open_records(sid) == []

    def test_an_approval_is_a_distinguishable_kind_of_ask(self, store, runner):
        """`request_approval` routes through the same ask_user, so only the kind separates them."""
        sid = _session(store)
        backend = self._backend(runner, sid)
        seen: dict = {}

        def answer():
            end = time.monotonic() + 5
            while time.monotonic() < end and not store.attention.open_records(sid):
                time.sleep(0.01)
            seen["kinds"] = [r.kind for r in store.attention.open_records(sid)]
            backend.submit_response("Approve")

        t = threading.Thread(target=answer)
        t.start()
        backend.ask_user("Fetch evil.test?", "approval", ["Approve", "Deny"])
        t.join()

        assert seen["kinds"] == ["needs_approval"]

    def test_a_timed_out_ask_clears_its_record(self, store, runner):
        sid = _session(store)
        backend = self._backend(runner, sid)
        backend.TIMEOUT = 0.05

        with pytest.raises(RuntimeError, match="Timed out"):
            backend.ask_user("Which branch?", "text")

        assert store.attention.open_records(sid) == []


class TestJobSource:
    @pytest.fixture
    def jobs(self, tmp_path):
        return JobStore(tmp_path / "jobs.json")

    @pytest.fixture
    def orchestrator(self, jobs, runner, bus):
        return JobsOrchestrator(jobs, runner, event_bus=bus)

    def _job(self, jobs, store, state=JobState.RUNNING.value) -> Job:
        _session(store)
        return jobs.add(Job(id="job-1", parent_session_id="s1", prompt="ship it", state=state))

    @pytest.mark.parametrize("state", [JobState.AWAITING_INPUT.value, JobState.STUCK.value, JobState.ERRORED.value])
    def test_a_parked_job_opens_a_record_on_its_parent_session(self, jobs, store, orchestrator, state):
        job = self._job(jobs, store)

        jobs.update_state(job.id, state)
        orchestrator._emit_job_event(jobs.get(job.id))

        records = store.attention.open_records("s1")
        assert [r.source for r in records] == [SOURCE_JOB]
        assert records[0].kind == state
        assert records[0].ref_id == "job-1"

    def test_leaving_the_parked_state_clears_the_record(self, jobs, store, orchestrator):
        job = self._job(jobs, store)
        jobs.update_state(job.id, JobState.STUCK.value)
        orchestrator._emit_job_event(jobs.get(job.id))

        jobs.update_state(job.id, JobState.RUNNING.value)
        orchestrator._emit_job_event(jobs.get(job.id))

        assert store.attention.open_records("s1") == []

    def test_a_job_already_parked_at_boot_gets_its_record(self, tmp_path, store, runner, bus):
        """A job stuck before the restart never transitions again, so load has to open it."""
        _session(store)
        jobs = JobStore(tmp_path / "jobs.json")
        jobs.add(Job(id="job-1", parent_session_id="s1", prompt="ship it", state=JobState.STUCK.value))

        JobsOrchestrator(JobStore(tmp_path / "jobs.json"), runner, event_bus=bus).reconcile_attention()

        assert [r.ref_id for r in store.attention.open_records("s1")] == ["job-1"]

    def test_boot_reconcile_leaves_unparked_jobs_alone(self, tmp_path, store, runner, bus):
        _session(store)
        jobs = JobStore(tmp_path / "jobs.json")
        jobs.add(Job(id="job-1", parent_session_id="s1", prompt="ship it", state=JobState.RUNNING.value))

        JobsOrchestrator(JobStore(tmp_path / "jobs.json"), runner, event_bus=bus).reconcile_attention()

        assert store.attention.open_records("s1") == []

    def test_boot_reconcile_does_not_duplicate_a_surviving_record(self, tmp_path, store, runner, bus):
        _session(store)
        jobs = JobStore(tmp_path / "jobs.json")
        jobs.add(Job(id="job-1", parent_session_id="s1", prompt="ship it", state=JobState.STUCK.value))
        JobsOrchestrator(jobs, runner, event_bus=bus).reconcile_attention()

        JobsOrchestrator(JobStore(tmp_path / "jobs.json"), runner, event_bus=bus).reconcile_attention()

        assert len(store.attention.open_records("s1")) == 1

    def test_pruning_an_errored_job_closes_its_record(self, tmp_path, store, runner, bus):
        _session(store)
        jobs = JobStore(tmp_path / "jobs.json", max_terminal_jobs=1)
        JobsOrchestrator(jobs, runner, event_bus=bus)
        old = jobs.add(Job(id="job-old", parent_session_id="s1", prompt="a", state=JobState.RUNNING.value))
        jobs.update_state(old.id, JobState.ERRORED.value)
        runner.open_attention("s1", source=SOURCE_JOB, ref_id=old.id, kind=JobState.ERRORED.value)
        newer = jobs.add(Job(id="job-new", parent_session_id="s1", prompt="b", state=JobState.RUNNING.value))
        bus.events.clear()

        jobs.update_state(newer.id, JobState.ERRORED.value)

        assert jobs.get(old.id) is None
        assert store.attention.open_records("s1") == []
        assert bus.of("session_update", "attention")

    def test_on_pruned_runs_outside_the_store_lock(self, tmp_path):
        jobs = JobStore(tmp_path / "jobs.json", max_terminal_jobs=1)
        seen = []
        jobs.on_pruned = lambda job_id: seen.append((job_id, jobs.get(job_id)))
        for job_id in ("job-old", "job-new"):
            jobs.add(Job(id=job_id, parent_session_id="s1", prompt="a", state=JobState.RUNNING.value))
            jobs.update_state(job_id, JobState.ERRORED.value)

        assert seen == [("job-old", None)]

    def test_boot_reconcile_closes_a_record_whose_job_is_gone(self, tmp_path, store, runner, bus):
        _session(store)
        jobs = JobStore(tmp_path / "jobs.json")
        jobs.add(Job(id="job-1", parent_session_id="s1", prompt="ship it", state=JobState.STUCK.value))
        runner.open_attention("s1", source=SOURCE_JOB, ref_id="job-gone", kind=JobState.ERRORED.value)
        runner.open_attention("s1", source=SOURCE_ERROR, ref_id="job-gone", kind="send_failed")

        JobsOrchestrator(jobs, runner, event_bus=bus).reconcile_attention()

        assert sorted((r.source, r.ref_id) for r in store.attention.open_records("s1")) == [
            (SOURCE_ERROR, "job-gone"),
            (SOURCE_JOB, "job-1"),
        ]
        assert bus.of("session_update", "attention")

    def test_a_running_job_opens_nothing(self, jobs, store, orchestrator):
        job = self._job(jobs, store)

        orchestrator._emit_job_event(jobs.get(job.id))

        assert store.attention.open_records("s1") == []


class TestScheduledParent:
    """A Job spawned by a scheduled run reports to the schedule's originating session."""

    @pytest.fixture
    def jobs(self, tmp_path):
        return JobStore(tmp_path / "jobs.json")

    def _scheduled_job(self, jobs, store, originating="origin", notify_when="stuck") -> Job:
        _session(store, "origin")
        store.create_session(
            Session(
                id="sched_run",
                source=SessionSource.SCHEDULE.value,
                parent_id="nightly",
                status=SessionStatus.COMPLETED.value,
                metadata={METADATA_ORIGINATING_SESSION: originating} if originating else {},
            )
        )
        return jobs.add(
            Job(id="job-1", parent_session_id="sched_run", prompt="ship it", state="stuck", notify_when=notify_when)
        )

    def test_attention_goes_to_the_originating_session(self, jobs, store, runner, bus):
        job = self._scheduled_job(jobs, store)

        JobsOrchestrator(jobs, runner, event_bus=bus)._emit_job_event(job)

        assert [r.source for r in store.attention.open_records("origin")] == [SOURCE_JOB]
        assert store.attention.open_records("sched_run") == []

    def test_attention_falls_back_to_the_run_without_an_originating_session(self, jobs, store, runner, bus):
        job = self._scheduled_job(jobs, store, originating=None)

        JobsOrchestrator(jobs, runner, event_bus=bus)._emit_job_event(job)

        assert [r.source for r in store.attention.open_records("sched_run")] == [SOURCE_JOB]
        assert store.attention.open_records("origin") == []

    @pytest.mark.asyncio
    async def test_the_notify_wake_up_goes_to_the_originating_session(self, jobs, store, runner, bus):
        job = self._scheduled_job(jobs, store)
        sent = []

        async def fake_reply(session_id, message, source="session", metadata=None):
            sent.append(session_id)
            return "ok"

        runner.reply_to_session = fake_reply
        orchestrator = JobsOrchestrator(jobs, runner, event_bus=bus)
        orchestrator._schedule_notify(job)
        await asyncio.gather(*orchestrator._notify_tasks)

        assert sent == ["origin"]

    def test_the_job_tile_event_goes_to_the_originating_session(self, jobs, store, runner, bus):
        job = self._scheduled_job(jobs, store)

        JobsOrchestrator(jobs, runner, event_bus=bus)._emit_job_event(job)

        assert [e["type"] for e in store.read_events("origin")] == ["job_status"]
        assert store.read_events("sched_run") == []

    def _barrier_orchestrator(self, jobs, store, runner, bus, sent):
        async def fake_reply(session_id, message, source="session", metadata=None):
            sent.append((session_id, source))
            return "ok"

        runner.reply_to_session = fake_reply
        job = self._scheduled_job(jobs, store, notify_when="all_done")
        jobs.update(job.id, state=JobState.DONE.value)
        return JobsOrchestrator(jobs, runner, event_bus=bus)

    @pytest.mark.asyncio
    async def test_the_batch_barrier_waits_on_the_run_session_turn(self, jobs, store, runner, bus):
        sent = []
        orchestrator = self._barrier_orchestrator(jobs, store, runner, bus, sent)
        store.update_session("sched_run", turn_in_flight=True)
        orchestrator.close_batch_barrier("sched_run")
        assert not orchestrator._notify_tasks

        jobs.add(
            Job(id="job-2", parent_session_id="sched_run", prompt="and this", state="done", notify_when="all_done")
        )
        orchestrator.close_batch_barrier("sched_run")
        assert not orchestrator._notify_tasks

        store.update_session("sched_run", turn_in_flight=False)
        orchestrator.on_turn_end("sched_run")
        await asyncio.gather(*orchestrator._notify_tasks)

        assert sent == [("origin", "jobs_all_complete")]
        assert all(j.barrier_notified for j in jobs.list_all())

    @pytest.mark.asyncio
    async def test_the_origin_turn_end_sends_the_runs_held_batch_once(self, jobs, store, runner, bus):
        sent = []
        orchestrator = self._barrier_orchestrator(jobs, store, runner, bus, sent)
        store.update_session("origin", turn_in_flight=True)
        orchestrator.close_batch_barrier("sched_run")
        assert not orchestrator._notify_tasks

        store.update_session("origin", turn_in_flight=False)
        orchestrator.on_turn_end("origin")
        orchestrator.on_turn_end("origin")
        await asyncio.gather(*orchestrator._notify_tasks)

        assert sent == [("origin", "jobs_all_complete")]

    @pytest.mark.asyncio
    async def test_a_compacted_origin_turn_end_sends_the_runs_held_batch(self, jobs, store, runner, bus):
        sent = []
        orchestrator = self._barrier_orchestrator(jobs, store, runner, bus, sent)
        _session(store, "origin2")
        store.update_session("origin", turn_in_flight=True)
        orchestrator.close_batch_barrier("sched_run")
        store.update_session("origin", turn_in_flight=False, superseded_by="origin2")
        orchestrator.on_turn_end("origin2")
        await asyncio.gather(*orchestrator._notify_tasks)

        assert [source for _, source in sent] == ["jobs_all_complete"]

    def test_a_pruned_run_session_falls_back_to_the_job_parent(self, jobs, store, runner, bus):
        job = self._scheduled_job(jobs, store)
        store.delete_session("sched_run")

        JobsOrchestrator(jobs, runner, event_bus=bus)._emit_job_event(job)

        assert [r.source for r in store.attention.open_records("sched_run")] == [SOURCE_JOB]
        assert store.attention.open_records("origin") == []

    @pytest.mark.asyncio
    async def test_a_finished_one_off_schedule_still_reports_to_its_origin(self, jobs, store, runner, bus, tmp_path):
        _session(store, "origin")
        app = MagicMock(agent_name="bot", session_store=store)
        app.resolve_model.return_value = "test-model"
        app.handle_message = AsyncMock(return_value="done")
        sched = SchedulerAdapter(adapter=app, schedules_path=tmp_path / "schedules.json")
        sched.set_session_runner(runner)
        entry = sched.scheduler.add(
            ScheduleEntry(
                id="once",
                prompt="p",
                schedule_type="once",
                run_at="2099-01-01T00:00:00Z",
                originating_session_id="origin",
            )
        )
        await sched.scheduler._fire_schedule(entry)
        run_id = entry.run_history[-1]["session_id"]
        job = jobs.add(Job(id="job-1", parent_session_id=run_id, prompt="ship it", state="stuck"))

        JobsOrchestrator(jobs, runner, event_bus=bus)._emit_job_event(job)

        assert sched.scheduler.list() == []
        assert [r.source for r in store.attention.open_records("origin")] == [SOURCE_JOB]
        assert store.attention.open_records(run_id) == []


class TestBootRecovery:
    def test_a_restart_drops_ask_records_and_keeps_the_rest(self, tmp_path, history_dir, store, runner):
        """An idle session, so the ask dies because it is an ask, not because the turn was flagged."""
        sid = _session(store)
        runner.deliver_to_session(sid, "approve?", source="job", kind="needs_ack")
        runner.open_attention(sid, source=SOURCE_ASK, ref_id="ask-1", kind="needs_answer")
        runner.open_attention(sid, source=SOURCE_JOB, ref_id="job-1", kind="stuck")

        reopened = SessionStore(tmp_path / "store.json")
        reopened.attention.clear_stale_asks()

        assert _sources(reopened, sid) == [SOURCE_DELIVERY, SOURCE_JOB]


class TestPayloads:
    def test_session_detail_carries_the_open_records(self, store, runner):
        sid = _session(store)
        runner.open_attention(sid, source=SOURCE_ASK, ref_id="ask-1", kind="needs_answer")

        detail = store.session_detail(sid)

        assert detail["needs_attention"] is True
        assert [(r["source"], r["ref_id"]) for r in detail["attention"]] == [(SOURCE_ASK, "ask-1")]

    def test_a_session_with_nothing_open_reports_no_attention(self, store):
        sid = _session(store)

        detail = store.session_detail(sid)

        assert detail["needs_attention"] is False
        assert detail["attention"] == []

    def test_an_ask_alone_makes_the_session_need_the_user(self, store, runner):
        """The old `bool(pending_deliveries)` answer would call this session idle."""
        sid = _session(store)
        runner.open_attention(sid, source=SOURCE_ASK, ref_id="ask-1", kind="needs_answer")

        assert store.session_detail(sid)["needs_attention"] is True
        assert store.get_session(sid).pending_deliveries == []


class TestBroadcast:
    """The broadcast carries the records, so a client patches its row rather than
    refetching the whole list to find out what changed."""

    def test_opening_a_record_announces_it(self, store, runner, bus):
        sid = _session(store)

        runner.open_attention(sid, source=SOURCE_ASK, ref_id="ask-1", kind="needs_answer")

        announced = bus.of("session_update", "attention")
        assert len(announced) == 1
        assert announced[0]["id"] == sid
        assert announced[0]["needs_attention"] is True
        assert [(r["source"], r["ref_id"]) for r in announced[0]["attention"]] == [(SOURCE_ASK, "ask-1")]

    def test_re_reporting_an_open_record_announces_nothing_new(self, store, runner, bus):
        """A parked job re-reports on every job event; each announcement would cost
        every client a refetch."""
        sid = _session(store)
        runner.open_attention(sid, source=SOURCE_JOB, ref_id="job-1", kind="stuck")

        runner.open_attention(sid, source=SOURCE_JOB, ref_id="job-1", kind="stuck")

        assert len(bus.of("session_update", "attention")) == 1

    def test_clearing_a_record_announces_the_remainder(self, store, runner, bus):
        sid = _session(store)
        runner.open_attention(sid, source=SOURCE_ASK, ref_id="ask-1", kind="needs_answer")

        runner.clear_attention_ref(SOURCE_ASK, "ask-1")

        last = bus.of("session_update", "attention")[-1]
        assert last["needs_attention"] is False
        assert last["attention"] == []


class TestWorklistIntegrity:
    """Two ways an open obligation could go missing from the worklist."""

    def test_deleting_a_session_takes_its_records_with_it(self, store):
        session = store.create_session(Session(id="s-del", source=SessionSource.INTERACTIVE.value, user_id="u1"))
        store.attention.open(
            owner_kind=OWNER_SESSION,
            owner_id=session.id,
            source=SOURCE_JOB,
            ref_id="job-1",
            kind="stuck",
        )

        store.delete_session(session.id)

        assert store.attention.open_records(session.id) == []

    def test_a_session_waiting_past_the_recency_limit_stays_in_the_list(self, store):
        """The limit bounds the quiet tail, never an obligation."""
        for i in range(5):
            store.create_session(
                Session(
                    id=f"s-{i}", source=SessionSource.INTERACTIVE.value, user_id="u1", last_active=f"2020-01-0{i + 1}"
                )
            )
        store.attention.open(
            owner_kind=OWNER_SESSION,
            owner_id="s-0",
            source=SOURCE_JOB,
            ref_id="job-1",
            kind="stuck",
        )

        listed = {s.id for s in store.list_sessions(limit=2)}

        assert "s-0" in listed


class TestOneAnswerEverywhere:
    """Every surface reports the same verdict, so acknowledging one obligation
    cannot claim the session is clear while another is still open."""

    def test_acknowledging_a_card_still_reports_an_open_ask(self, store, runner):
        sid = _session(store)
        runner.deliver_to_session(sid, "nightly report", source="schedule", kind="needs_ack")
        runner.open_attention(sid, source=SOURCE_ASK, ref_id="ask-1", kind="needs_answer")

        session = runner.clear_attention(sid)

        assert session.pending_deliveries == []
        assert [r.source for r in store.attention.open_records(sid)] == [SOURCE_ASK]
        assert store.session_detail(sid)["needs_attention"] is True

    def test_the_broadcast_reports_the_record_verdict(self, store, runner, bus):
        sid = _session(store)
        runner.deliver_to_session(sid, "nightly report", source="schedule", kind="needs_ack")
        runner.open_attention(sid, source=SOURCE_JOB, ref_id="job-1", kind="stuck")

        runner.clear_attention(sid)

        announced = bus.of("session_update", "attention")[-1]
        assert announced["needs_attention"] is True
        assert [r["source"] for r in announced["attention"]] == [SOURCE_JOB]


class TestErrorSource:
    """A send that fails after the turn is an obligation too: the chat gets an
    error block and the session list gets a record the user can acknowledge."""

    @staticmethod
    async def _boom(*args, **kwargs):
        raise RuntimeError("target is gone")

    def _notify_target(self, store, runner, monkeypatch) -> str:
        sid = _session(store, "s1")
        runner.add_notify_session(sid, _session(store, "s2"))
        monkeypatch.setattr(runner, "reply_to_session", self._boom)
        return sid

    @pytest.mark.asyncio
    async def test_a_failed_notify_opens_an_error_record(self, store, runner, monkeypatch):
        sid = self._notify_target(store, runner, monkeypatch)

        await runner._notify_finished(store.get_session(sid), SessionStatus.COMPLETED.value, "done")

        records = store.attention.open_records(sid)
        assert [(r.source, r.kind) for r in records] == [(SOURCE_ERROR, "send_failed")]
        assert attention_fields(records)["needs_attention"] is True

    @pytest.mark.asyncio
    async def test_the_chat_records_what_failed(self, store, runner, monkeypatch):
        sid = self._notify_target(store, runner, monkeypatch)

        await runner._notify_finished(store.get_session(sid), SessionStatus.COMPLETED.value, "done")

        errors = [e for e in store.read_events(sid) if e["type"] == "error"]
        assert len(errors) == 1
        assert "target is gone" in errors[0]["error"]

    @pytest.mark.asyncio
    async def test_the_failed_send_stays_out_of_the_next_prompt(self, store, runner, monkeypatch):
        """The model's own turn succeeded. Replaying the send failure into its
        context would make it resume work it already finished."""
        sid = self._notify_target(store, runner, monkeypatch)

        await runner._notify_finished(store.get_session(sid), SessionStatus.COMPLETED.value, "done")

        events = [Event(type=e["type"], ts=e["timestamp"], data=e) for e in store.read_events(sid)]
        assert [e.type for e in events].count("error") == 1
        assert all("target is gone" not in m["content"] for m in events_to_messages(events))

    @pytest.mark.asyncio
    async def test_acknowledging_clears_the_error_record(self, store, runner, monkeypatch):
        sid = self._notify_target(store, runner, monkeypatch)
        await runner._notify_finished(store.get_session(sid), SessionStatus.COMPLETED.value, "done")
        assert _sources(store, sid) == [SOURCE_ERROR]

        runner.clear_attention(sid)

        assert store.attention.open_records(sid) == []
        assert store.session_detail(sid)["needs_attention"] is False

    @pytest.mark.asyncio
    async def test_a_failed_completion_listener_opens_a_record(self, store, runner):
        sid = _session(store)
        runner.add_completion_listener(self._boom)

        await runner._dispatch_completion(store.get_session(sid), "done")

        assert [r.source for r in store.attention.open_records(sid)] == [SOURCE_ERROR]

    def _failing_notifier(self, store, bus, monkeypatch) -> tuple[SessionRunner, str]:
        runner = SessionRunner(
            store=store, adapter=None, event_bus=bus, notification_channels={"ntfy": {"url": "https://ntfy.test"}}
        )

        def boom(*args, **kwargs):
            raise RuntimeError("ntfy is down")

        monkeypatch.setattr("tsugite_daemon.session_runner.send_notification_nowait", boom)
        return runner, _session(store)

    def test_a_failed_delivery_notification_opens_a_record(self, store, bus, monkeypatch):
        runner, sid = self._failing_notifier(store, bus, monkeypatch)

        runner.deliver_to_session(sid, "approve?", source="job", kind="needs_ack", notify_channels=[("ntfy", {})])

        assert _sources(store, sid) == [SOURCE_DELIVERY, SOURCE_ERROR]

    def test_acknowledging_clears_delivery_and_error_together(self, store, bus, monkeypatch):
        runner, sid = self._failing_notifier(store, bus, monkeypatch)
        runner.deliver_to_session(sid, "approve?", source="job", kind="needs_ack", notify_channels=[("ntfy", {})])
        assert _sources(store, sid) == [SOURCE_DELIVERY, SOURCE_ERROR]

        runner.clear_attention(sid)

        assert store.attention.open_records(sid) == []

    def test_acknowledging_an_error_leaves_an_open_ask(self, store, runner):
        sid = _session(store)
        runner.open_attention(sid, source=SOURCE_ERROR, ref_id="err-1", kind="send_failed")
        runner.open_attention(sid, source=SOURCE_ASK, ref_id="ask-1", kind="needs_answer")

        runner.clear_attention(sid)

        assert [r.source for r in store.attention.open_records(sid)] == [SOURCE_ASK]
