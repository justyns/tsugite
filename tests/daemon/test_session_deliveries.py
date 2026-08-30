"""Deliveries: cards pushed into a session's history without starting a turn.

A delivery is how a schedule, job, or another session hands the user something
in an existing conversation. It appends one `delivery` event and bumps the
session so the sidebar shows it as unread.

A needs-ack delivery also records an obligation on the session itself, one entry
per card. The obligation is live session state re-rendered into `<message_context>`
every turn, so it outlives a compaction that summarizes its card out of the
timeline. It is discharged by the agent calling `session_acknowledge` or by an
explicit dismiss - never by the mere arrival of another message.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient
from tsugite_daemon.adapters.base import BaseAdapter, ChannelContext
from tsugite_daemon.config import HTTPConfig, RuntimeDefaults
from tsugite_daemon.session_runner import SessionRunner
from tsugite_daemon.session_store import Session, SessionSource, SessionStatus, SessionStore
from tsugite_daemon.webhook_store import WebhookStore


class _Bus:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def emit(self, name, payload):
        self.events.append((name, payload))

    def of(self, name: str, action: str | None = None) -> list[dict]:
        return [p for n, p in self.events if n == name and (action is None or p.get("action") == action)]


@pytest.fixture
def store(tmp_path, history_dir):
    return SessionStore(tmp_path / "store.json")


@pytest.fixture
def bus():
    return _Bus()


@pytest.fixture
def runner(store, bus):
    return SessionRunner(store=store, adapter=None, event_bus=bus)


def _session(store: SessionStore, sid: str = "s1", **kwargs) -> str:
    store.create_session(Session(id=sid, source=SessionSource.INTERACTIVE.value, user_id="alice", **kwargs))
    return sid


def _deliveries(store: SessionStore, sid: str) -> list[dict]:
    return [e for e in store.read_events(sid) if e["type"] == "delivery"]


class TestDeliveryEvent:
    def test_delivery_lands_in_history_with_its_metadata(self, store, runner):
        sid = _session(store)
        runner.deliver_to_session(
            sid,
            "rent is due tomorrow",
            source="schedule",
            kind="fyi",
            title="Rent",
            metadata={"schedule_id": "rent-watch"},
        )

        delivered = _deliveries(store, sid)
        assert len(delivered) == 1
        event = delivered[0]
        assert event["message"] == "rent is due tomorrow"
        assert event["source"] == "schedule"
        assert event["kind"] == "fyi"
        assert event["title"] == "Rent"
        assert event["schedule_id"] == "rent-watch"

    def test_delivery_does_not_start_a_turn(self, store, runner):
        sid = _session(store)
        runner.deliver_to_session(sid, "fyi", source="job")

        assert len(_deliveries(store, sid)) == 1
        assert [e["type"] for e in store.read_events(sid)] == ["delivery"]
        assert store.get_session(sid).turn_in_flight is False

    def test_delivery_makes_the_session_unread(self, store, runner):
        sid = _session(store)
        store.mark_viewed(sid)

        runner.deliver_to_session(sid, "something happened", source="job")

        session = store.get_session(sid)
        assert session.last_active > session.last_viewed_at

    def test_a_finished_session_is_never_given_a_card(self, store, runner):
        """An ended session can never answer, so an obligation there is undischargeable."""
        sid = _session(store)
        store.update_session(sid, status=SessionStatus.COMPLETED.value)

        runner.deliver_to_session(sid, "approve?", source="job", kind="needs_ack")

        assert _deliveries(store, sid) == []
        assert store.get_session(sid).has_pending_deliveries is False

    def test_unknown_kind_is_rejected(self, store, runner):
        sid = _session(store)
        with pytest.raises(ValueError):
            runner.deliver_to_session(sid, "hi", source="job", kind="urgent")


class TestAttention:
    def test_needs_ack_sets_needs_attention(self, store, runner):
        sid = _session(store)
        runner.deliver_to_session(sid, "approve the deploy?", source="job", kind="needs_ack")

        assert store.get_session(sid).has_pending_deliveries is True

    def test_fyi_leaves_attention_clear(self, store, runner):
        sid = _session(store)
        runner.deliver_to_session(sid, "build finished", source="job", kind="fyi")

        assert len(_deliveries(store, sid)) == 1
        assert store.get_session(sid).has_pending_deliveries is False

    def test_reading_a_session_does_not_clear_attention(self, store, runner):
        sid = _session(store)
        runner.deliver_to_session(sid, "approve?", source="job", kind="needs_ack")

        store.mark_viewed(sid)

        assert store.get_session(sid).has_pending_deliveries is True

    def test_clearing_attention_does_not_re_mark_unread(self, store):
        sid = _session(store)
        store.record_delivery(
            sid, {"type": "delivery", "delivery_id": "dlv-1", "message": "m", "kind": "needs_ack"}, needs_ack=True
        )
        assert store.get_session(sid).has_pending_deliveries is True
        store.mark_viewed(sid)

        store.clear_attention(sid)

        session = store.get_session(sid)
        assert session.has_pending_deliveries is False
        assert session.last_active <= session.last_viewed_at

    def test_attention_survives_a_daemon_restart(self, store, runner, tmp_path):
        sid = _session(store)
        runner.deliver_to_session(sid, "approve?", source="job", kind="needs_ack")

        reopened = SessionStore(tmp_path / "store.json")

        assert reopened.get_session(sid).has_pending_deliveries is True

    def test_attention_follows_compaction_to_the_successor(self, store, runner):
        sid = _session(store)
        runner.deliver_to_session(sid, "approve?", source="job", kind="needs_ack")

        successor = store.compact_session(sid)

        assert successor.has_pending_deliveries is True
        assert store.get_session(sid).has_pending_deliveries is False

    def test_runner_clear_attention_broadcasts(self, store, runner, bus):
        sid = _session(store)
        runner.deliver_to_session(sid, "approve?", source="job", kind="needs_ack")

        runner.clear_attention(sid)

        assert store.get_session(sid).has_pending_deliveries is False
        assert bus.of("session_update", "attention_cleared") == [
            {"action": "attention_cleared", "id": sid, "pending_deliveries": []}
        ]


class TestObligationSurvivesTheRecencyCap:
    """`list_sessions` truncates to the newest `limit` rows and every needs-you
    surface reads that capped list, so an unanswered card must survive the
    window the way a pin does."""

    def _seed(self, store, runner, *, newer: int) -> str:
        """One long-idle session holding a card, buried under `newer` fresh ones."""
        sid = _session(store, "old-obligation")
        runner.deliver_to_session(sid, "approve the deploy?", source="job", kind="needs_ack")
        store.get_session(sid).last_active = "2026-01-01T00:00:00+00:00"
        for i in range(newer):
            _session(store, f"recent-{i}")
        return sid

    def test_a_card_outside_the_window_is_still_listed(self, store, runner):
        sid = self._seed(store, runner, newer=5)

        ids = [s.id for s in store.list_sessions(limit=3)]

        assert sid in ids, "the recency window evicted a session with an unanswered card"

    def test_the_window_still_bounds_sessions_with_nothing_outstanding(self, store, runner):
        self._seed(store, runner, newer=5)

        rows = store.list_sessions(limit=3)

        assert len([s for s in rows if not s.has_pending_deliveries]) == 3

    def test_a_card_inside_the_window_is_not_listed_twice(self, store, runner):
        for i in range(3):
            _session(store, f"recent-{i}")
        sid = _session(store, "fresh-obligation")
        runner.deliver_to_session(sid, "approve?", source="job", kind="needs_ack")

        rows = store.list_sessions(limit=3)

        assert [s.id for s in rows].count(sid) == 1

    def test_discharging_the_card_returns_the_session_to_the_window(self, store, runner):
        sid = self._seed(store, runner, newer=5)

        runner.clear_attention(sid)

        ids = [s.id for s in store.list_sessions(limit=3)]
        assert sid not in ids


class TestMidTurnDeferral:
    def test_delivery_during_a_turn_waits_for_turn_end(self, store, runner):
        sid = _session(store)
        store.begin_turn(sid)

        runner.deliver_to_session(sid, "first", source="schedule")
        runner.deliver_to_session(sid, "second", source="job", kind="needs_ack")
        assert _deliveries(store, sid) == []

        store.end_turn(sid)

        assert [e["message"] for e in _deliveries(store, sid)] == ["first", "second"]
        assert store.get_session(sid).has_pending_deliveries is True

    def test_delivery_deferred_across_a_compaction_lands_on_the_successor(self, store, runner):
        sid = _session(store)
        store.begin_turn(sid)
        runner.deliver_to_session(sid, "backup failed", source="schedule", kind="needs_ack")

        # Mid-turn compaction hands the marker over: the old session's turn ends,
        # the session rotates, and the successor carries the turn to completion.
        store.end_turn(sid, notify_listeners=False)
        successor = store.compact_session(sid)
        store.begin_turn(successor.id)
        store.end_turn(successor.id)

        assert _deliveries(store, sid) == []
        assert [e["message"] for e in _deliveries(store, successor.id)] == ["backup failed"]
        assert store.get_session(successor.id).has_pending_deliveries is True

    @pytest.mark.asyncio
    async def test_a_reply_resumes_the_successor_of_a_compacted_session(self, store, bus):
        """A job wake-up or a schedule's completion callback carries the id it saw
        when it started; by the time it fires the chat may have rotated, so the turn
        belongs in the conversation the user still has open."""
        adapter = MagicMock()
        adapter.handle_message = AsyncMock(return_value="ok")
        runner = SessionRunner(store=store, adapter=adapter, event_bus=bus)
        sid = _session(store)
        successor = store.compact_session(sid)

        await runner.reply_to_session(sid, "the job finished", source="job_complete")

        context = adapter.handle_message.await_args.kwargs["channel_context"]
        assert context.metadata["conv_id_override"] == successor.id

    def test_deleting_a_session_drops_what_it_was_holding(self, store, runner):
        """A purged session's turn never ends, so what it held goes with it."""
        sid = _session(store)
        store.begin_turn(sid)
        runner.deliver_to_session(sid, "held", source="job")

        store.delete_session(sid)

        assert store.take_deferred_deliveries(sid) == []
        assert store.sessions_holding_deliveries() == []

    def test_deferral_is_per_session(self, store, runner):
        busy = _session(store, "busy")
        idle = _session(store, "idle")
        store.begin_turn(busy)

        runner.deliver_to_session(busy, "later", source="job")
        runner.deliver_to_session(idle, "now", source="job")

        assert _deliveries(store, busy) == []
        assert [e["message"] for e in _deliveries(store, idle)] == ["now"]

        store.end_turn(busy)
        assert [e["message"] for e in _deliveries(store, busy)] == ["later"]


class TestFanoutDoesNotBlockTheLoop:
    @pytest.mark.asyncio
    async def test_flushing_a_deferred_needs_ack_delivery_leaves_the_loop_free(self, store, bus):
        """The turn-end flush runs inline on the daemon's loop, so a fanout that
        waits on that same loop would freeze the daemon until it timed out."""
        from tsugite.tools.notify import set_notifier

        sent: list[str] = []

        async def notifier(message, channel_configs, url="/"):
            sent.append(url)
            return {}

        set_notifier(notifier, asyncio.get_running_loop())
        try:
            channels = {"push": MagicMock()}
            runner = SessionRunner(store=store, adapter=None, event_bus=bus, notification_channels=channels)
            sid = _session(store)
            store.begin_turn(sid)
            runner.deliver_to_session(
                sid, "approve?", source="job", kind="needs_ack", notify_channels=list(channels.items())
            )

            started = time.monotonic()
            store.end_turn(sid)
            elapsed = time.monotonic() - started

            assert elapsed < 1
            for _ in range(50):
                if sent:
                    break
                await asyncio.sleep(0.01)
            assert sent == [f"#chats?sessionId={sid}"]
        finally:
            set_notifier(None, None)


class TestBroadcasts:
    def test_delivery_emits_event_and_session_update(self, store, runner, bus):
        sid = _session(store)
        runner.deliver_to_session(
            sid,
            "approve?",
            source="schedule",
            kind="needs_ack",
            title="Deploy",
            metadata={"schedule_id": "deploy-watch"},
        )

        [event] = bus.of("session_event")
        assert event["session_id"] == sid
        assert event["event_type"] == "delivery"
        assert event["message"] == "approve?"
        assert event["kind"] == "needs_ack"
        assert event["title"] == "Deploy"
        # The card shows its schedule live, not only after a replay.
        assert event["schedule_id"] == "deploy-watch"

        assert bus.of("session_update", "delivered") == [
            {
                "action": "delivered",
                "id": sid,
                "pending_deliveries": [event["delivery_id"]],
            }
        ]

    def test_fyi_delivery_reports_no_attention(self, store, runner, bus):
        sid = _session(store)
        runner.deliver_to_session(sid, "done", source="job")

        assert bus.of("session_update", "delivered") == [{"action": "delivered", "id": sid, "pending_deliveries": []}]


class TestTurnEndListener:
    def test_listener_fires_with_the_session_id(self, store):
        sid = _session(store)
        seen: list[str] = []
        store.set_turn_end_listener(seen.append)

        store.begin_turn(sid)
        store.end_turn(sid)

        assert seen == [sid]

    def test_compactions_marker_handoff_does_not_fire_it(self, store):
        """The turn moves to the successor rather than ending, so nothing flushes."""
        sid = _session(store)
        seen: list[str] = []
        store.set_turn_end_listener(seen.append)

        store.begin_turn(sid)
        store.end_turn(sid, notify_listeners=False)

        assert seen == []


class TestIncidentSessions:
    def test_finds_the_session_stamped_with_the_key(self, store):
        _session(store, "other")
        incident = _session(store, "incident")
        store.update_session(incident, metadata={"incident_key": "disk-full"})

        found = store.find_incident_session("alice", "disk-full")

        assert found is not None and found.id == incident

    def test_ignores_finished_and_superseded_sessions(self, store):
        done = _session(store, "done", status=SessionStatus.COMPLETED.value)
        store.update_session(done, metadata={"incident_key": "disk-full"})
        gone = _session(store, "gone")
        store.update_session(gone, metadata={"incident_key": "disk-full"}, superseded_by="elsewhere")

        assert store.find_incident_session("alice", "disk-full") is None

    def test_unknown_key_finds_nothing(self, store):
        sid = _session(store)
        store.update_session(sid, metadata={"incident_key": "disk-full"})

        assert store.find_incident_session("alice", "other-key") is None

    def test_incident_key_is_not_agent_writable(self, store):
        sid = _session(store)
        with pytest.raises(ValueError):
            store.set_metadata_bulk(sid, {"incident_key": "hijacked"})

    def test_incident_key_survives_compaction(self, store):
        sid = _session(store)
        store.update_session(sid, metadata={"incident_key": "disk-full"})

        successor = store.compact_session(sid)

        assert successor.metadata.get("incident_key") == "disk-full"


class TestExternalFanout:
    @pytest.fixture
    def channels(self):
        from tsugite_daemon.config import NotificationChannelConfig

        return {"push": NotificationChannelConfig(type="web-push")}

    def test_needs_ack_notifies_once_with_a_deep_link(self, store, bus, channels):
        runner = SessionRunner(store=store, adapter=None, event_bus=bus, notification_channels=channels)
        sid = _session(store)

        with patch("tsugite_daemon.session_runner.send_notification_nowait") as send:
            runner.deliver_to_session(
                sid,
                "approve the deploy?",
                source="job",
                kind="needs_ack",
                title="Deploy",
                notify_channels=list(channels.items()),
            )

        assert send.call_count == 1
        message, resolved = send.call_args.args
        assert "approve the deploy?" in message
        assert "Deploy" in message
        assert resolved == [("push", channels["push"])]
        assert send.call_args.kwargs["url"] == f"#chats?sessionId={sid}"

    def test_an_unaddressed_delivery_does_not_ping_every_configured_channel(self, store, bus, channels):
        """The registry is not the audience: a card names the channels it pings, so
        one landing in another person's session cannot reach this one's."""
        runner = SessionRunner(store=store, adapter=None, event_bus=bus, notification_channels=channels)
        sid = _session(store)

        with patch("tsugite_daemon.session_runner.send_notification_nowait") as send:
            runner.deliver_to_session(sid, "approve?", source="job", kind="needs_ack")

        assert send.call_count == 0

    def test_fyi_does_not_notify(self, store, bus, channels):
        runner = SessionRunner(store=store, adapter=None, event_bus=bus, notification_channels=channels)
        sid = _session(store)

        with patch("tsugite_daemon.session_runner.send_notification_nowait") as send:
            runner.deliver_to_session(sid, "build finished", source="job")

        assert send.call_count == 0

    def test_a_dead_channel_does_not_break_the_delivery(self, store, bus, channels):
        runner = SessionRunner(store=store, adapter=None, event_bus=bus, notification_channels=channels)
        sid = _session(store)

        with patch("tsugite_daemon.session_runner.send_notification_nowait", side_effect=RuntimeError("channel down")):
            runner.deliver_to_session(
                sid, "approve?", source="job", kind="needs_ack", notify_channels=list(channels.items())
            )

        assert len(_deliveries(store, sid)) == 1
        assert store.get_session(sid).has_pending_deliveries is True


class _StubAdapter(BaseAdapter):
    def get_platform_name(self) -> str:
        return "test"

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


@pytest.fixture
def adapter(tmp_path, store, runner, monkeypatch):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "agent.md").write_text("---\nname: agent\n---\n\nHi.\n")
    adapter = _StubAdapter(RuntimeDefaults(workspace_dir=ws, agent_file=str(ws / "agent.md")), store)

    monkeypatch.setattr(adapter, "_resolve_agent_path", lambda: ws / "agent.md")
    monkeypatch.setattr(adapter, "_build_message_context", lambda msg, *a, **kw: msg)
    monkeypatch.setattr(adapter, "_build_agent_context", lambda *a, **kw: {})
    monkeypatch.setattr(adapter, "_save_history", lambda **kw: None)
    monkeypatch.setattr(adapter, "_update_skill_ttl", lambda *a, **kw: None)
    monkeypatch.setattr(
        "tsugite_daemon.adapters.base.run_agent",
        lambda *a, **kw: MagicMock(token_count=0, cost=0, execution_steps=[]),
    )
    return adapter


def _ctx(sid: str) -> ChannelContext:
    return ChannelContext(
        source="http",
        channel_id=None,
        user_id="alice",
        reply_to="http:alice",
        metadata={"conv_id_override": sid},
    )


class TestRepliesDoNotDischargeAnObligation:
    @pytest.mark.asyncio
    async def test_an_unrelated_reply_leaves_the_obligation_outstanding(self, store, runner, adapter):
        """The feature exists so a glance cannot kill an obligation; a reply
        about the weather must not kill it either."""
        sid = _session(store)
        runner.deliver_to_session(sid, "rent is due friday", source="schedule", kind="needs_ack")

        await adapter.handle_message("alice", "what's the weather like?", _ctx(sid))

        assert store.get_session(sid).has_pending_deliveries is True


@pytest.fixture
def server(tmp_path, store, runner):
    ws = tmp_path / "http-workspace"
    ws.mkdir()
    agent_config = RuntimeDefaults(workspace_dir=ws, agent_file="default")

    from tsugite_daemon.adapters.http import HTTPAgentAdapter, HTTPServer
    from tsugite_daemon.auth import TokenStore

    with patch("tsugite.workspace.Workspace") as mock_ws_cls:
        from tsugite.workspace import WorkspaceNotFoundError

        mock_ws_cls.load.side_effect = WorkspaceNotFoundError("not found")
        adapter = HTTPAgentAdapter(runtime=agent_config, session_store=store)

    srv = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8379),
        adapter=adapter,
        webhook_store=WebhookStore(tmp_path / "webhooks.json"),
        token_store=TokenStore(tmp_path / "tokens.json"),
    )
    srv.session_runner = runner
    return srv


@pytest.fixture
def client(server):
    return TestClient(server.app)


@pytest.fixture
def auth(server):
    _stored, raw = server._token_store.create_admin_token(name="test-token")
    return {"Authorization": f"Bearer {raw}"}


class TestHTTPSurface:
    def test_session_row_carries_needs_attention(self, store, runner, client, auth):
        sid = _session(store)
        runner.deliver_to_session(sid, "approve?", source="job", kind="needs_ack")

        resp = client.get("/api/chat/sessions", headers=auth)

        assert resp.status_code == 200
        rows = {r["id"]: r for r in resp.json()["sessions"]}
        assert rows[sid]["needs_attention"] is True

    def test_dismiss_endpoint_clears_the_flag(self, store, runner, client, auth):
        sid = _session(store)
        runner.deliver_to_session(sid, "approve?", source="job", kind="needs_ack")

        resp = client.post(f"/api/sessions/{sid}/dismiss-attention", json={}, headers=auth)

        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert store.get_session(sid).has_pending_deliveries is False

    def test_dismiss_reaches_the_session_a_compacted_chat_became(self, store, runner, client, auth):
        """A client holding the pre-compaction id must still discharge the card."""
        sid = _session(store)
        runner.deliver_to_session(sid, "approve?", source="job", kind="needs_ack")
        successor = store.compact_session(sid)

        resp = client.post(f"/api/sessions/{sid}/dismiss-attention", json={}, headers=auth)

        assert resp.status_code == 200
        assert resp.json()["needs_attention"] is False
        assert store.get_session(successor.id).has_pending_deliveries is False

    def test_dismiss_unknown_session_404s(self, client, auth):
        resp = client.post("/api/sessions/missing/dismiss-attention", json={}, headers=auth)
        assert resp.status_code == 404

    def test_dismiss_requires_auth(self, store, client):
        sid = _session(store)
        resp = client.post(f"/api/sessions/{sid}/dismiss-attention", json={})
        assert resp.status_code == 401


class TestPerCardObligations:
    def test_each_needs_ack_delivery_gets_its_own_id(self, store, runner):
        sid = _session(store)
        runner.deliver_to_session(sid, "rent is due friday", source="schedule", kind="needs_ack")
        runner.deliver_to_session(sid, "approve the deploy?", source="job", kind="needs_ack")

        first, second = (e.get("delivery_id") for e in _deliveries(store, sid))

        assert first and second and first != second

    def test_dismissing_one_card_leaves_the_other_outstanding(self, store, runner, client, auth):
        sid = _session(store)
        runner.deliver_to_session(sid, "rent is due friday", source="schedule", kind="needs_ack")
        runner.deliver_to_session(sid, "approve the deploy?", source="job", kind="needs_ack")
        first = _deliveries(store, sid)[0].get("delivery_id")

        resp = client.post(f"/api/sessions/{sid}/dismiss-attention", json={"delivery_id": first}, headers=auth)

        assert resp.status_code == 200
        assert store.get_session(sid).has_pending_deliveries is True

    def test_dismissing_the_last_card_clears_the_session(self, store, runner, client, auth):
        sid = _session(store)
        runner.deliver_to_session(sid, "rent is due friday", source="schedule", kind="needs_ack")
        runner.deliver_to_session(sid, "approve the deploy?", source="job", kind="needs_ack")
        for event in _deliveries(store, sid):
            client.post(
                f"/api/sessions/{sid}/dismiss-attention", json={"delivery_id": event["delivery_id"]}, headers=auth
            )

        assert store.get_session(sid).has_pending_deliveries is False

    def test_dismissing_without_an_id_clears_every_obligation(self, store, runner, client, auth):
        """The session-row menu's clear-all: one action for a chat the person
        has read through and wants quiet."""
        sid = _session(store)
        runner.deliver_to_session(sid, "rent is due friday", source="schedule", kind="needs_ack")
        runner.deliver_to_session(sid, "approve the deploy?", source="job", kind="needs_ack")

        client.post(f"/api/sessions/{sid}/dismiss-attention", json={}, headers=auth)

        assert store.get_session(sid).has_pending_deliveries is False

    def test_an_fyi_delivery_creates_no_obligation(self, store, runner):
        sid = _session(store)
        runner.deliver_to_session(sid, "build finished", source="job", kind="fyi")

        assert store.get_session(sid).pending_deliveries == []

    def test_the_session_row_carries_what_is_still_outstanding(self, store, runner, client, auth):
        """The chat needs per-card truth to gate each card's own dismiss control."""
        sid = _session(store)
        runner.deliver_to_session(sid, "rent is due friday", source="schedule", kind="needs_ack")
        runner.deliver_to_session(sid, "build finished", source="job", kind="fyi")
        outstanding = [e["delivery_id"] for e in _deliveries(store, sid) if e["kind"] == "needs_ack"]

        resp = client.get("/api/chat/sessions", headers=auth)

        row = {r["id"]: r for r in resp.json()["sessions"]}[sid]
        assert row["pending_deliveries"] == outstanding

    def test_an_fyi_landing_on_an_owed_session_leaves_the_obligation(self, store, runner, bus):
        """A card's own kind is not the session's answer."""
        sid = _session(store)
        runner.deliver_to_session(sid, "rent is due friday", source="schedule", kind="needs_ack")
        runner.deliver_to_session(sid, "build finished", source="job", kind="fyi")

        fyi = bus.of("session_update", "delivered")[-1]

        assert fyi["pending_deliveries"] == store.get_session(sid).pending_delivery_ids
        assert store.session_detail(sid)["needs_attention"] is True

    def test_session_detail_names_outstanding_deliveries_the_way_the_row_does(self, store, runner):
        sid = _session(store)
        runner.deliver_to_session(sid, "rent is due friday", source="schedule", kind="needs_ack")

        detail = store.session_detail(sid)

        assert detail["needs_attention"] is True
        assert detail["pending_deliveries"] == store.get_session(sid).pending_delivery_ids

    def test_a_dismiss_broadcast_reports_what_is_left(self, store, runner, bus):
        sid = _session(store)
        runner.deliver_to_session(sid, "rent is due friday", source="schedule", kind="needs_ack")
        runner.deliver_to_session(sid, "approve the deploy?", source="job", kind="needs_ack")
        first, second = (e["delivery_id"] for e in _deliveries(store, sid))

        runner.clear_attention(sid, delivery_id=first)

        assert bus.of("session_update", "attention_cleared") == [
            {"action": "attention_cleared", "id": sid, "pending_deliveries": [second]}
        ]
        assert bus.of("session_update", "attention")[-1]["needs_attention"] is True


@pytest.fixture
def context_adapter(tmp_path, store, runner, monkeypatch):
    """An adapter whose `_build_message_context` is the real one."""
    ws = tmp_path / "ctx-workspace"
    ws.mkdir()
    (ws / "agent.md").write_text("---\nname: agent\n---\n\nHi.\n")
    adapter = _StubAdapter(RuntimeDefaults(workspace_dir=ws, agent_file=str(ws / "agent.md")), store)
    return adapter


class TestAgentSeesWhatIsOutstanding:
    def test_message_context_lists_every_outstanding_obligation(self, store, runner, context_adapter):
        sid = _session(store)
        runner.deliver_to_session(sid, "rent is due friday", source="schedule", kind="needs_ack", title="Rent")
        runner.deliver_to_session(sid, "approve the deploy?", source="job", kind="needs_ack")
        first, second = (e["delivery_id"] for e in _deliveries(store, sid))

        rendered = context_adapter._build_message_context("hi", _ctx(sid), "alice")

        assert "<pending_deliveries>" in rendered
        assert first in rendered and second in rendered
        assert "rent is due friday" in rendered
        assert 'source="schedule"' in rendered
        assert "Rent" in rendered

    def test_message_context_omits_the_block_when_nothing_is_outstanding(self, store, runner, context_adapter):
        sid = _session(store)
        runner.deliver_to_session(sid, "build finished", source="job", kind="fyi")

        rendered = context_adapter._build_message_context("hi", _ctx(sid), "alice")

        assert "<pending_deliveries>" not in rendered

    def test_a_dismissed_obligation_leaves_the_context(self, store, runner, context_adapter):
        sid = _session(store)
        runner.deliver_to_session(sid, "rent is due friday", source="schedule", kind="needs_ack")
        runner.clear_attention(sid)

        rendered = context_adapter._build_message_context("hi", _ctx(sid), "alice")

        assert "rent is due friday" not in rendered

    def test_message_context_follows_the_compaction_that_just_happened(self, store, runner, context_adapter):
        """A mid-turn compaction rotates the session after the caller resolved
        it, so the channel context still names the session the turn started on."""
        sid = _session(store)
        runner.deliver_to_session(sid, "rent is due friday", source="schedule", kind="needs_ack")
        store.compact_session(sid)

        rendered = context_adapter._build_message_context("hi", _ctx(sid), "alice")

        assert "rent is due friday" in rendered


def _compaction_patches():
    """Drive a real compaction: a real event split against a tiny retention
    budget, with only the summarizer and the hooks stubbed out."""

    async def fake_summarize(messages, model=None, max_context_tokens=None, progress_callback=None):
        return "Summary of the earlier conversation."

    return [
        patch("tsugite_daemon.memory.get_context_limit", return_value=400),
        patch("tsugite_daemon.memory.infer_compaction_model", return_value="anthropic:claude-3-haiku-20240307"),
        patch("tsugite_daemon.memory.summarize_session", new=fake_summarize),
        patch("tsugite.hooks.fire_compact_hooks", new_callable=AsyncMock, return_value=[]),
    ]


def _seed_turns(session_id: str, count: int, first: int = 0) -> None:
    from tests.history_helpers import load_history_session

    storage = load_history_session(session_id)
    for i in range(first, first + count):
        storage.record("user_input", text=f"question {i} " + ("padding " * 40))
        storage.record("model_response", raw_content=f"answer {i} " + ("padding " * 40))


async def _compact(adapter, session_id: str):
    from contextlib import ExitStack

    with ExitStack() as stack:
        for p in _compaction_patches():
            stack.enter_context(p)
        return await adapter._compact_session(session_id, reason="token_threshold")


class TestObligationOutlivesCompaction:
    @pytest.mark.asyncio
    async def test_an_obligation_survives_two_compactions_that_summarize_its_card_away(
        self, store, runner, context_adapter, history_dir
    ):
        """Compaction copies only the retained window forward, so a card older
        than that window is summarized into prose and never replayed again. The
        obligation is session state, not a timeline event, so it stays."""
        from tests.history_helpers import load_history_session, seed_history_session

        sid = _session(store)
        seed_history_session(sid, agent="test-agent", model="anthropic:claude-sonnet-4-5")
        _seed_turns(sid, 1)
        runner.deliver_to_session(sid, "rent is due friday", source="schedule", kind="needs_ack", title="Rent")
        _seed_turns(sid, 5, first=1)

        first = await _compact(context_adapter, sid)
        assert first is not None
        _seed_turns(first.id, 5, first=10)
        second = await _compact(context_adapter, first.id)
        assert second is not None

        replayed = load_history_session(second.id).load_events()
        assert not any(e.type == "delivery" for e in replayed), "the card must be outside the retained window"

        assert store.get_session(second.id).has_pending_deliveries is True
        rendered = context_adapter._build_message_context("hi", _ctx(second.id), "alice")
        assert "rent is due friday" in rendered


class TestSessionAcknowledge:
    @pytest.fixture
    def tools(self, store, runner):
        """The session tools bound to a runner on a background event loop."""
        import threading

        from tsugite.tools import sessions as session_tools

        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        session_tools.set_session_runner(runner, loop)
        yield session_tools
        session_tools.set_session_runner(None)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()

    def test_a_daemon_agent_gets_it_under_the_sessions_category(self, tools):
        """The default agent pulls `@sessions`; a tool that never lands in that
        category is one the agent cannot call."""
        from tsugite.tools import get_tools_by_category, set_daemon_mode

        set_daemon_mode(True)
        try:
            assert "session_acknowledge" in get_tools_by_category("sessions")
        finally:
            set_daemon_mode(False)

    def test_acknowledging_discharges_the_obligation(self, store, runner, tools):
        sid = _session(store)
        runner.deliver_to_session(sid, "rent is due friday", source="schedule", kind="needs_ack")

        result = tools.session_acknowledge(session_id=sid)

        assert result["needs_attention"] is False
        assert store.get_session(sid).has_pending_deliveries is False

    def test_acknowledging_one_card_leaves_the_other_outstanding(self, store, runner, tools):
        sid = _session(store)
        runner.deliver_to_session(sid, "rent is due friday", source="schedule", kind="needs_ack")
        runner.deliver_to_session(sid, "approve the deploy?", source="job", kind="needs_ack")
        first, second = (e["delivery_id"] for e in _deliveries(store, sid))

        result = tools.session_acknowledge(delivery_id=first, session_id=sid)

        assert result["pending_deliveries"] == [second]
        assert store.get_session(sid).has_pending_deliveries is True

    def test_it_defaults_to_the_session_the_agent_is_running_in(self, store, runner, tools):
        sid = _session(store)
        runner.deliver_to_session(sid, "rent is due friday", source="schedule", kind="needs_ack")

        with patch.object(tools, "get_current_session_id", return_value=sid):
            tools.session_acknowledge()

        assert store.get_session(sid).has_pending_deliveries is False

    def test_it_reaches_the_session_a_compacted_chat_became(self, store, runner, tools):
        sid = _session(store)
        runner.deliver_to_session(sid, "rent is due friday", source="schedule", kind="needs_ack")
        first = store.compact_session(sid)
        second = store.compact_session(first.id)

        with patch.object(tools, "get_current_session_id", return_value=sid):
            tools.session_acknowledge()

        assert store.get_session(second.id).has_pending_deliveries is False

    def test_it_broadcasts_so_open_clients_stop_saying_needs_you(self, store, runner, bus, tools):
        sid = _session(store)
        runner.deliver_to_session(sid, "rent is due friday", source="schedule", kind="needs_ack")

        tools.session_acknowledge(session_id=sid)

        assert bus.of("session_update", "attention_cleared") == [
            {"action": "attention_cleared", "id": sid, "pending_deliveries": []}
        ]


class TestTheAgentIsToldWhenToAcknowledge:
    def _render_default_agent(self, **context) -> str:
        from pathlib import Path

        from tsugite.renderer import AgentRenderer

        body = Path("tsugite/builtin_agents/default.md").read_text().split("---\n", 2)[2]
        return AgentRenderer().render(body, {"user_prompt": "hi", "is_interactive": False, "tools": [], **context})

    def test_an_outstanding_obligation_puts_the_instruction_in_the_prompt(self):
        rendered = self._render_default_agent(is_daemon=True, has_pending_deliveries=True)

        assert "session_acknowledge" in rendered

    def test_no_obligation_leaves_the_prompt_alone(self):
        rendered = self._render_default_agent(is_daemon=True, has_pending_deliveries=False)

        assert "session_acknowledge" not in rendered


class TestHeldDeliveriesSurviveADaemonDeath:
    def test_a_card_held_mid_turn_is_still_owed_after_a_restart(self, tmp_path, history_dir):
        path = tmp_path / "restart_store.json"
        store = SessionStore(path)
        runner = SessionRunner(store=store, adapter=None, event_bus=None)
        sid = _session(store)
        store.begin_turn(sid)
        runner.deliver_to_session(sid, "rent is due", source="schedule", kind="needs_ack")

        reopened = SessionStore(path)
        SessionRunner(store=reopened, adapter=None, event_bus=None).flush_held_deliveries()

        assert reopened.get_session(sid).has_pending_deliveries is True
        assert len(_deliveries(reopened, sid)) == 1

    @pytest.mark.asyncio
    async def test_a_needs_ack_card_held_across_a_restart_still_notifies(self, tmp_path, history_dir):
        """The gateway wires the notifier after it builds the runner, so a boot flush
        that runs during construction has nobody to tell."""
        from tsugite_daemon.config import NotificationChannelConfig

        from tsugite.tools.notify import set_notifier

        path = tmp_path / "restart_store.json"
        channels = {"push": NotificationChannelConfig(type="web-push")}
        store = SessionStore(path)
        runner = SessionRunner(store=store, adapter=None, event_bus=None, notification_channels=channels)
        sid = _session(store)
        store.begin_turn(sid)
        runner.deliver_to_session(
            sid, "approve the deploy?", source="job", kind="needs_ack", notify_channels=list(channels.items())
        )

        sent: list[str] = []

        async def notifier(message, channel_configs, url="/"):
            sent.append(message)
            return {}

        reopened = SessionStore(path)
        rebooted = SessionRunner(store=reopened, adapter=None, event_bus=None, notification_channels=channels)
        set_notifier(notifier, asyncio.get_running_loop())
        try:
            rebooted.flush_held_deliveries()
            for _ in range(50):
                if sent:
                    break
                await asyncio.sleep(0.01)
        finally:
            set_notifier(None, None)

        assert sent == ["approve the deploy?"]

    def test_a_card_is_not_held_for_a_turn_that_already_ended(self, store):
        """The hold decision and the hold share one lock, so a turn that ends first
        cannot strand the card behind a flush that already ran."""
        sid = _session(store)
        store.begin_turn(sid)
        store.end_turn(sid)

        assert store.hold_delivery(sid, {"delivery_id": "dlv-1"}) is False
        assert store.take_deferred_deliveries(sid) == []

    def test_a_card_is_held_while_the_turn_runs(self, store):
        sid = _session(store)
        store.begin_turn(sid)

        assert store.hold_delivery(sid, {"delivery_id": "dlv-1"}) is True
        assert store.take_deferred_deliveries(sid) == [{"delivery_id": "dlv-1"}]


class TestResumableTargets:
    """A background session created to be talked to keeps taking cards after its
    first turn completes, matching what `session_reply` already allows."""

    def _resumable(self, store: SessionStore, sid: str, status: str) -> str:
        store.create_session(
            Session(
                id=sid,
                source=SessionSource.BACKGROUND.value,
                status=status,
                resumable=True,
            )
        )
        return sid

    def test_a_completed_resumable_session_still_gets_cards(self, store, runner):
        sid = self._resumable(store, "chatty", SessionStatus.COMPLETED.value)

        runner.deliver_to_session(sid, "the build is green", source="job", kind="fyi")

        assert len(_deliveries(store, sid)) == 1

    def test_a_failed_resumable_session_gets_none(self, store, runner):
        sid = self._resumable(store, "broken", SessionStatus.FAILED.value)

        runner.deliver_to_session(sid, "the build is green", source="job", kind="fyi")

        assert _deliveries(store, sid) == []
