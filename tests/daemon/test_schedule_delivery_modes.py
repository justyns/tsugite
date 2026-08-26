"""Delivery modes: where a scheduled run's result lands.

A schedule delivers its result into a session through the delivery primitive,
whether or not it has notification channels. `delivery_mode` picks the session:
the routed existing one, the session that spawned the schedule, or a dedicated
incident session that repeat runs of the same monitor share via `incident_key`.
"""

import asyncio
import re
from threading import Thread
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from tsugite_daemon.adapters.scheduler_adapter import SchedulerAdapter, resolve_delivery_sessions
from tsugite_daemon.config import NotificationChannelConfig
from tsugite_daemon.scheduler import ScheduleEntry, entry_to_dict
from tsugite_daemon.session_runner import SessionRunner
from tsugite_daemon.session_store import Session, SessionSource, SessionStatus, SessionStore


@pytest.fixture
def store(tmp_path, history_dir):
    return SessionStore(tmp_path / "store.json")


@pytest.fixture
def runner(store):
    return SessionRunner(store=store, adapter=None, event_bus=None)


@pytest.fixture
def adapter(store):
    a = MagicMock()
    a.agent_name = "bot"
    a.session_store = store
    a.resolve_model.return_value = "test-model"
    a.handle_message = AsyncMock(return_value="backup failed on disk 2")
    return a


@pytest.fixture
def sched_adapter(tmp_path, adapter, runner, request):
    channels = getattr(request, "param", {})
    sa = SchedulerAdapter(
        adapter=adapter,
        schedules_path=tmp_path / "schedules.json",
        notification_channels=channels,
    )
    sa.set_session_runner(runner)
    return sa


@pytest.fixture
def tool_loop():
    """Background event loop so the schedule tools can call the scheduler thread-safely."""
    from tsugite.tools.schedule import set_scheduler

    loop = asyncio.new_event_loop()
    thread = Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)
    loop.close()
    set_scheduler(None)


def _entry(**kwargs) -> ScheduleEntry:
    defaults = dict(id="backup-watch", prompt="check backups", schedule_type="cron", cron_expr="0 9 * * *")
    defaults.update(kwargs)
    return ScheduleEntry(**defaults)


def _chat(store: SessionStore, sid: str = "chat1", user_id: str = "alice") -> Session:
    return store.create_session(Session(id=sid, source=SessionSource.INTERACTIVE.value, user_id=user_id))


async def _drain_tasks() -> None:
    """Let the tasks the scheduler hook spawned run to completion."""
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if pending:
        await asyncio.gather(*pending)


def _deliveries(store: SessionStore, sid: str) -> list[dict]:
    return [e for e in store.read_events(sid) if e["type"] == "delivery"]


def _resolve_one(entry: ScheduleEntry, user_id: str, store: SessionStore) -> Session | None:
    """The single session a one-recipient run delivers into."""
    sessions = resolve_delivery_sessions(entry, [user_id], store)
    return sessions[0] if sessions else None


def _incident_sessions(store: SessionStore) -> list[Session]:
    return [s for s in store.list_sessions() if s.metadata.get("type") == "ops"]


def _calling_session(sid: str | None):
    return patch("tsugite_daemon.session_runner.get_current_session_id", return_value=sid)


def _compact_twice(store: SessionStore, sid: str) -> Session:
    """The live session left after `sid` is compacted twice."""
    return store.compact_session(store.compact_session(sid).id)


@pytest.fixture
def mock_scheduler(tool_loop):
    from tsugite.tools.schedule import set_scheduler

    sched = MagicMock()
    sched.add.side_effect = lambda entry: entry
    sched.get.return_value = _entry(id="t1")
    sched.update.side_effect = lambda sid, **fields: _entry(id=sid, **fields)
    set_scheduler(sched, tool_loop)
    return sched


class TestDeliveryFields:
    def test_defaults_route_to_the_existing_session(self):
        entry = _entry()
        assert entry.delivery_mode == "existing_session"
        assert entry.delivery_kind == "fyi"
        assert entry.incident_key is None
        assert entry.incident_title is None

    def test_persisted_roundtrip(self):
        data = entry_to_dict(_entry(delivery_mode="new_session", delivery_kind="needs_ack", incident_key="k"))
        assert data["delivery_mode"] == "new_session"
        restored = ScheduleEntry(**data)
        assert restored.delivery_kind == "needs_ack"
        assert restored.incident_key == "k"

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="delivery_mode"):
            _entry(delivery_mode="telepathy")

    def test_rejects_unknown_kind(self):
        with pytest.raises(ValueError, match="delivery_kind"):
            _entry(delivery_kind="shout")


class TestResolveDeliverySession:
    def test_existing_session_routes_like_target_session(self, store):
        chat = _chat(store)
        entry = _entry(originating_session_id=chat.id)

        resolved = _resolve_one(entry, "alice", store)
        assert resolved is not None
        assert resolved.id == chat.id

    def test_parent_session_ignores_the_primary(self, store):
        primary = _chat(store, "primary-chat")
        store.set_primary_session(primary.id)
        spawner = _chat(store, "spawner-chat")
        entry = _entry(delivery_mode="parent_session", originating_session_id=spawner.id)

        resolved = _resolve_one(entry, "alice", store)
        assert resolved is not None
        assert resolved.id == spawner.id

    def test_new_session_opens_an_incident_rather_than_reusing_the_chat(self, store):
        chat = _chat(store)
        entry = _entry(delivery_mode="new_session", originating_session_id=chat.id)

        resolved = _resolve_one(entry, "alice", store)
        assert resolved is not None
        assert resolved.id != chat.id
        assert resolved.source == SessionSource.SCHEDULE.value

    def test_auto_fyi_stays_in_the_existing_session(self, store):
        chat = _chat(store)
        entry = _entry(delivery_mode="auto", delivery_kind="fyi", originating_session_id=chat.id)

        resolved = _resolve_one(entry, "alice", store)
        assert resolved is not None
        assert resolved.id == chat.id

    def test_auto_needs_ack_opens_an_incident(self, store):
        chat = _chat(store)
        entry = _entry(delivery_mode="auto", delivery_kind="needs_ack", originating_session_id=chat.id)

        resolved = _resolve_one(entry, "alice", store)
        assert resolved is not None
        assert resolved.id != chat.id
        assert resolved.metadata.get("type") == "ops"

    def test_incident_carries_its_identifying_metadata(self, store):
        entry = _entry(
            delivery_mode="new_session",
            incident_key="backup-disk-2",
            incident_title="Backups failing",
        )

        resolved = _resolve_one(entry, "alice", store)
        assert resolved is not None
        assert resolved.title == "Backups failing"
        assert resolved.user_id == "alice"
        assert resolved.metadata["type"] == "ops"
        assert resolved.metadata["topic"] == "Backups failing"
        assert resolved.metadata["incident_key"] == "backup-disk-2"
        assert resolved.metadata["schedule_id"] == "backup-watch"

    def test_incident_uses_the_standard_session_id_dialect(self, store):
        """generate_session_id, not a hand-rolled dialect - id shape is what the
        rest of the system reads a session's provenance from."""
        entry = _entry(delivery_mode="new_session")

        resolved = _resolve_one(entry, "alice", store)
        assert resolved is not None
        assert re.fullmatch(r"\d{8}_\d{6}_session_[0-9a-f]{6}", resolved.id)

    def test_incident_title_defaults_to_the_schedule(self, store):
        entry = _entry(delivery_mode="new_session")

        resolved = _resolve_one(entry, "alice", store)
        assert resolved is not None
        assert resolved.title == "Incident: backup-watch"
        assert resolved.metadata["incident_key"] == entry.id

    def test_incident_key_dedupes_across_resolutions(self, store):
        entry = _entry(delivery_mode="new_session", incident_key="backup-disk-2")

        first = _resolve_one(entry, "alice", store)
        second = _resolve_one(entry, "alice", store)
        assert first is not None and second is not None
        assert first.id == second.id
        assert len(_incident_sessions(store)) == 1

    def test_without_an_incident_key_the_schedule_id_still_dedupes(self, store):
        """An unkeyed monitor collects into one incident session rather than
        opening one per firing."""
        entry = _entry(delivery_mode="new_session")

        first = _resolve_one(entry, "alice", store)
        second = _resolve_one(entry, "alice", store)
        assert first is not None and second is not None
        assert first.id == second.id
        assert first.metadata["incident_key"] == entry.id

    def test_a_finished_incident_does_not_absorb_the_next_one(self, store):
        entry = _entry(delivery_mode="new_session", incident_key="backup-disk-2")
        first = _resolve_one(entry, "alice", store)
        assert first is not None
        store.update_session(first.id, status=SessionStatus.COMPLETED.value)

        second = _resolve_one(entry, "alice", store)
        assert second is not None
        assert second.id != first.id


class TestScheduledRunDelivery:
    @pytest.mark.asyncio
    async def test_delivers_without_any_notify_channel(self, store, sched_adapter):
        chat = _chat(store)
        entry = _entry(originating_session_id=chat.id)

        await sched_adapter._run_agent(entry)

        delivered = _deliveries(store, chat.id)
        assert len(delivered) == 1
        assert delivered[0]["message"] == "backup failed on disk 2"
        assert delivered[0]["source"] == "schedule"
        assert delivered[0]["schedule_id"] == "backup-watch"

    @pytest.mark.asyncio
    async def test_inject_history_false_delivers_nothing(self, store, sched_adapter):
        chat = _chat(store)
        entry = _entry(originating_session_id=chat.id, inject_history=False)

        await sched_adapter._run_agent(entry)

        assert _deliveries(store, chat.id) == []

    @pytest.mark.asyncio
    async def test_needs_ack_marks_the_target_session(self, store, sched_adapter):
        chat = _chat(store)
        entry = _entry(originating_session_id=chat.id, delivery_kind="needs_ack")

        await sched_adapter._run_agent(entry)

        delivered = _deliveries(store, chat.id)
        assert len(delivered) == 1
        assert delivered[0]["kind"] == "needs_ack"
        assert store.get_session(chat.id).has_pending_deliveries is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "sched_adapter",
        [{"hook": NotificationChannelConfig(type="webhook", url="https://example.invalid/hook")}],
        indirect=True,
    )
    async def test_a_webhook_only_schedule_still_delivers(self, store, sched_adapter, monkeypatch):
        monkeypatch.setattr("tsugite_daemon.adapters.scheduler_adapter.send_notification", lambda *a, **k: None)
        chat = _chat(store)
        entry = _entry(originating_session_id=chat.id, notify=["hook"])

        await sched_adapter._run_agent(entry)

        assert len(_deliveries(store, chat.id)) == 1

    @pytest.mark.asyncio
    async def test_a_fresh_incident_session_announces_itself(self, store, sched_adapter, adapter):
        """The rail drops a session_update for an id it has never seen, so an
        incident that never broadcasts `created` stays invisible until a refetch."""
        entry = _entry(delivery_mode="new_session", incident_key="backup-disk-2")

        await sched_adapter._run_agent(entry)

        incidents = _incident_sessions(store)
        assert len(incidents) == 1
        created = call("session_update", {"action": "created", "id": incidents[0].id})
        assert created in adapter.event_bus.emit.call_args_list

    @pytest.mark.asyncio
    async def test_repeat_runs_share_one_incident_session(self, store, sched_adapter):
        chat = _chat(store)
        entry = _entry(
            originating_session_id=chat.id,
            delivery_mode="new_session",
            incident_key="backup-disk-2",
            incident_title="Backups failing",
        )

        await sched_adapter._run_agent(entry)
        await sched_adapter._run_agent(entry)

        incidents = _incident_sessions(store)
        assert len(incidents) == 1
        assert len(_deliveries(store, incidents[0].id)) == 2
        assert _deliveries(store, chat.id) == []


class TestToolPassthrough:
    def test_create_passes_delivery_fields(self, mock_scheduler):
        from tsugite.tools.schedule import schedule_create

        result = schedule_create(
            id="t1",
            prompt="hi",
            cron="0 9 * * *",
            delivery_mode="new_session",
            delivery_kind="needs_ack",
            incident_key="backup-disk-2",
        )
        assert result["delivery_mode"] == "new_session"
        assert result["delivery_kind"] == "needs_ack"
        assert result["incident_key"] == "backup-disk-2"

    def test_create_rejects_a_bad_mode(self, mock_scheduler):
        from tsugite.tools.schedule import schedule_create

        with pytest.raises(ValueError, match="delivery_mode"):
            schedule_create(id="t1", prompt="hi", cron="0 9 * * *", delivery_mode="telepathy")

    def test_update_passes_delivery_fields(self, mock_scheduler):
        from tsugite.tools.schedule import schedule_update

        result = schedule_update(id="t1", delivery_mode="auto", delivery_kind="needs_ack")
        assert result["delivery_mode"] == "auto"
        assert mock_scheduler.update.call_args.kwargs["delivery_kind"] == "needs_ack"

    def test_update_clears_the_incident_key_with_an_empty_string(self, mock_scheduler):
        from tsugite.tools.schedule import schedule_update

        schedule_update(id="t1", incident_key="")
        assert mock_scheduler.update.call_args.kwargs["incident_key"] is None

    def test_update_rejects_a_bad_kind(self, tmp_path, tool_loop):
        """Enforcement lives in Scheduler._validate_entry, which both add and
        update reach - a mock scheduler never gets there."""
        from tsugite_daemon.scheduler import Scheduler

        from tsugite.tools.schedule import schedule_update, set_scheduler

        scheduler = Scheduler(tmp_path / "real-schedules.json", AsyncMock())
        scheduler.add(_entry(id="t1"))
        set_scheduler(scheduler, tool_loop)

        with pytest.raises(ValueError, match="delivery_kind"):
            schedule_update(id="t1", delivery_kind="shout")

        assert scheduler.get("t1").delivery_kind == "fyi"

    def test_update_rejects_a_null_mode(self, tmp_path):
        """A null from the HTTP PATCH body is not a legal mode, and setting it
        would persist a value the resolver never matches."""
        from tsugite_daemon.scheduler import Scheduler

        scheduler = Scheduler(tmp_path / "real-schedules.json", AsyncMock())
        scheduler.add(_entry(id="t1"))

        with pytest.raises(ValueError, match="delivery_mode"):
            scheduler.update("t1", delivery_mode=None)

        assert scheduler.get("t1").delivery_mode == "existing_session"


class TestRepeatedFailureDelivery:
    """A schedule that keeps failing surfaces as a needs-ack card instead of only a log line."""

    @pytest.mark.asyncio
    async def test_crossing_the_failure_threshold_delivers_a_needs_ack_card(self, sched_adapter, store):
        target = store.create_session(Session(id="ops", source=SessionSource.INTERACTIVE.value, user_id=""))
        entry = _entry(target_session=target.id, notify_on_failure=2, consecutive_failures=2)

        sched_adapter.scheduler._maybe_notify_repeated_failure(entry)
        await _drain_tasks()

        cards = [e for e in store.read_events(target.id) if e["type"] == "delivery"]
        assert [c["kind"] for c in cards] == ["needs_ack"]
        assert "backup-watch" in cards[0]["message"]
        assert store.get_session(target.id).has_pending_deliveries is True

    @pytest.mark.asyncio
    async def test_below_the_threshold_delivers_nothing(self, sched_adapter, store):
        target = store.create_session(Session(id="ops", source=SessionSource.INTERACTIVE.value, user_id=""))
        entry = _entry(target_session=target.id, notify_on_failure=2, consecutive_failures=1)

        sched_adapter.scheduler._maybe_notify_repeated_failure(entry)
        await _drain_tasks()

        assert [e for e in store.read_events(target.id) if e["type"] == "delivery"] == []


class TestScheduleTargetsTheCallingChat:
    """An agent in a chat can point a schedule at that chat without knowing its id."""

    def test_create_records_the_calling_session(self, mock_scheduler):
        from tsugite.tools.schedule import schedule_create

        with _calling_session("chat1"):
            result = schedule_create(id="t1", prompt="hi", cron="0 9 * * *")

        assert result["originating_session_id"] == "chat1"

    def test_create_resolves_current_to_a_concrete_id(self, mock_scheduler):
        from tsugite.tools.schedule import schedule_create

        with _calling_session("chat1"):
            result = schedule_create(id="t1", prompt="hi", cron="0 9 * * *", target_session="current")

        assert result["target_session"] == "chat1"

    def test_create_rejects_current_outside_a_session(self, mock_scheduler):
        from tsugite.tools.schedule import schedule_create

        with _calling_session(None), pytest.raises(ValueError, match="current"):
            schedule_create(id="t1", prompt="hi", cron="0 9 * * *", target_session="current")

    def test_background_task_records_the_calling_session(self, mock_scheduler):
        from tsugite.tools.schedule import background_task

        with _calling_session("chat1"):
            background_task(prompt="dig into the logs")

        assert mock_scheduler.add.call_args[0][0].originating_session_id == "chat1"

    def test_background_task_resolves_current_to_a_concrete_id(self, mock_scheduler):
        from tsugite.tools.schedule import background_task

        with _calling_session("chat1"):
            background_task(prompt="dig into the logs", target_session="current")

        assert mock_scheduler.add.call_args[0][0].target_session == "chat1"

    @pytest.mark.asyncio
    async def test_a_schedule_made_in_a_chat_posts_back_into_it(self, store, sched_adapter, mock_scheduler):
        from tsugite.tools.schedule import schedule_create

        chat = _chat(store)
        with _calling_session(chat.id):
            schedule_create(id="backup-watch", prompt="check backups", cron="0 9 * * *", target_session="originating")
        entry = mock_scheduler.add.call_args[0][0]

        await sched_adapter._run_agent(entry)

        assert len(_deliveries(store, chat.id)) == 1


class TestDeliveryFollowsCompaction:
    """A conversation that compacted twice is still one conversation."""

    @pytest.mark.asyncio
    async def test_a_bare_id_target_reaches_the_live_session(self, store, sched_adapter):
        chat = _chat(store)
        second = _compact_twice(store, chat.id)
        entry = _entry(target_session=chat.id)

        await sched_adapter._run_agent(entry)

        assert len(_deliveries(store, second.id)) == 1
        assert _deliveries(store, chat.id) == []

    @pytest.mark.asyncio
    async def test_a_current_target_reaches_the_live_session(self, store, sched_adapter, mock_scheduler):
        from tsugite.tools.schedule import schedule_create

        chat = _chat(store)
        with _calling_session(chat.id):
            schedule_create(id="backup-watch", prompt="check backups", cron="0 9 * * *", target_session="current")
        entry = mock_scheduler.add.call_args[0][0]
        second = _compact_twice(store, chat.id)

        await sched_adapter._run_agent(entry)

        assert len(_deliveries(store, second.id)) == 1

    @pytest.mark.asyncio
    async def test_a_named_target_reaches_the_live_session(self, store, sched_adapter):
        named = store.claim_aliased_session("daily")
        store.update_session(named.id, user_id="alice")
        second = _compact_twice(store, named.id)
        entry = _entry(target_session="name:daily", originating_session_id=named.id)

        await sched_adapter._run_agent(entry)

        assert len(_deliveries(store, second.id)) == 1


class TestDeliveryWithoutNotifyChannels:
    """No channel names a user, so the schedule routes by its own sessions."""

    @pytest.mark.asyncio
    async def test_the_primary_chat_still_receives_the_result(self, store, sched_adapter):
        chat = _chat(store)
        store.set_primary_session(chat.id)

        await sched_adapter._run_agent(_entry())

        assert len(_deliveries(store, chat.id)) == 1

    @pytest.mark.asyncio
    async def test_a_bare_id_target_needs_no_identifiable_user(self, store, sched_adapter):
        target = _chat(store, "ops", user_id="")

        await sched_adapter._run_agent(_entry(target_session=target.id))

        assert len(_deliveries(store, target.id)) == 1


class TestDeliveryStaysWithinItsAudience:
    """A run's result reaches the sessions it is addressed to, once each."""

    @pytest.mark.asyncio
    async def test_the_spawning_users_chat_receives_it_and_nobody_elses(self, store, sched_adapter):
        alice = _chat(store, "alice-chat", "alice")
        bob = _chat(store, "bob-chat", "bob")
        store.set_primary_session(alice.id)
        store.set_primary_session(bob.id)

        await sched_adapter._run_agent(_entry(originating_session_id=alice.id))

        assert len(_deliveries(store, alice.id)) == 1
        assert _deliveries(store, bob.id) == []

    @pytest.mark.asyncio
    async def test_an_unattributable_run_reaches_nobody_rather_than_everybody(self, store, sched_adapter):
        """No channel and no spawning session names a person, and two people could
        be meant, so the result is withheld rather than shown to both."""
        alice = _chat(store, "alice-chat", "alice")
        bob = _chat(store, "bob-chat", "bob")
        store.set_primary_session(alice.id)
        store.set_primary_session(bob.id)

        await sched_adapter._run_agent(_entry())

        assert _deliveries(store, alice.id) == []
        assert _deliveries(store, bob.id) == []

    @pytest.mark.asyncio
    async def test_a_named_session_target_is_delivered_once_however_many_owners(self, store, sched_adapter):
        for name in ("alice", "bob", "carol"):
            store.set_primary_session(_chat(store, f"{name}-chat", name).id)
        ops = _chat(store, "ops", user_id="")

        await sched_adapter._run_agent(_entry(target_session=ops.id, delivery_kind="needs_ack"))

        assert len(_deliveries(store, ops.id)) == 1
        assert len(store.get_session(ops.id).pending_deliveries) == 1

    @pytest.mark.asyncio
    async def test_a_finished_session_is_never_given_an_obligation(self, store, sched_adapter):
        ops = _chat(store, "ops", user_id="")
        store.update_session(ops.id, status=SessionStatus.COMPLETED.value)

        await sched_adapter._run_agent(_entry(target_session=ops.id, delivery_kind="needs_ack"))

        assert _deliveries(store, ops.id) == []
        assert store.get_session(ops.id).has_pending_deliveries is False

    @pytest.mark.asyncio
    async def test_one_incident_session_per_run_not_one_per_owner(self, store, sched_adapter):
        for name in ("alice", "bob"):
            store.set_primary_session(_chat(store, f"{name}-chat", name).id)

        await sched_adapter._run_agent(
            _entry(delivery_mode="new_session", incident_key="disk-2", delivery_kind="needs_ack")
        )

        assert len(_incident_sessions(store)) == 1

    @pytest.mark.asyncio
    async def test_an_unresolvable_target_delivers_nothing(self, store, sched_adapter):
        store.set_primary_session(_chat(store, "alice-chat", "alice").id)

        await sched_adapter._run_agent(_entry(target_session="none"))

        assert _incident_sessions(store) == []
        assert _deliveries(store, "alice-chat") == []
