"""Tests for GET /api/activity - the cross-cutting recent-activity feed."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPServer
from tsugite_daemon.auth import TokenStore
from tsugite_daemon.config import HTTPConfig
from tsugite_daemon.job_store import Job, JobStore
from tsugite_daemon.scheduler import ScheduleEntry
from tsugite_daemon.session_store import Session, SessionStore
from tsugite_daemon.webhook_store import WebhookStore

from tsugite.history import get_history_backend

# A fixed day so every seeded timestamp orders deterministically.
DAY = "2026-08-17T"


def at(clock: str) -> str:
    return f"{DAY}{clock}+00:00"


@pytest.fixture
def session_store(tmp_path):
    store = SessionStore(tmp_path / "session_store.json")
    store.create_session(Session(id="chat-1", title="Morning triage"))
    store.create_session(Session(id="chat-2", title="Deploy check"))
    store.create_session(Session(id="run-1", prompt="poll the feeds"))
    return store


@pytest.fixture
def job_store(tmp_path):
    store = JobStore(tmp_path / "jobs.json")
    store.add(
        Job(
            id="job-done",
            parent_session_id="chat-1",
            prompt="ship the release",
            state="done",
            agent="demo",
            resolved_at=at("09:30:00"),
            updated_at=at("09:30:00"),
        )
    )
    store.add(
        Job(
            id="job-errored",
            parent_session_id="chat-2",
            prompt="rebuild the index",
            state="errored",
            error="worker died",
            agent="demo",
            resolved_at=at("09:10:00"),
            updated_at=at("09:10:00"),
        )
    )
    store.add(
        Job(
            id="job-running",
            parent_session_id="chat-2",
            prompt="still going",
            state="running",
            agent="demo",
            updated_at=at("09:45:00"),
        )
    )
    return store


@pytest.fixture
def scheduler():
    entry = ScheduleEntry(id="feeds", prompt="poll the feeds", schedule_type="cron", cron_expr="*/5 * * * *")
    entry.run_history = [
        {"timestamp": at("08:00:00"), "status": "success", "error": None, "session_id": "run-0"},
        {"timestamp": at("09:00:00"), "status": "error", "error": "timed out", "session_id": "run-1"},
    ]
    return SimpleNamespace(list=lambda: [entry])


@pytest.fixture
def history(history_dir):
    """Seed the shared event log: a chat that ended two turns, plus a compaction."""
    backend = get_history_backend()
    chat1 = backend.create(agent_name="demo", model="test:model", session_id="chat-1")
    chat1.record("session_end", ts=at("09:05:00"), status="success")
    chat1.record("session_end", ts=at("09:35:00"), status="success")
    chat2 = backend.create(agent_name="demo", model="test:model", session_id="chat-2")
    chat2.record("compaction", ts=at("09:20:00"), reason="auto", replaced_count=12, retained_count=3)
    run1 = backend.create(agent_name="watcher", model="test:model", session_id="run-1")
    run1.record("session_error", ts=at("09:00:00"), error="timed out")
    return backend


@pytest.fixture
def token_store(tmp_path):
    return TokenStore(tmp_path / "tokens.json")


@pytest.fixture
def test_token(token_store):
    _st, raw = token_store.create_admin_token(name="activity-token")
    return raw


@pytest.fixture
def server(tmp_path, session_store, job_store, scheduler, token_store, history):
    s = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8484),
        adapter=None,
        webhook_store=WebhookStore(tmp_path / "webhooks.json"),
        token_store=token_store,
    )
    s.job_store = job_store
    s.scheduler = scheduler
    s.session_runner = SimpleNamespace(store=session_store)
    return s


@pytest.fixture
def client(server):
    return TestClient(server.app)


def fetch(client, token, **params) -> dict:
    resp = client.get("/api/activity", params=params, headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_requires_auth(client):
    assert client.get("/api/activity").status_code == 401


def test_merges_every_source_newest_first(client, test_token):
    entries = fetch(client, test_token)["entries"]

    assert [(e["type"], e["title"], e["timestamp"]) for e in entries] == [
        ("session", "Morning triage", at("09:35:00")),
        ("job", "ship the release", at("09:30:00")),
        ("compaction", "Deploy check", at("09:20:00")),
        ("job", "rebuild the index", at("09:10:00")),
        ("session", "poll the feeds", at("09:00:00")),
        ("schedule", "feeds", at("09:00:00")),
        ("schedule", "feeds", at("08:00:00")),
    ]
    assert len({e["id"] for e in entries}) == len(entries)


def test_a_chat_that_only_emits_session_end_still_appears(client, test_token):
    """Interactive chats never emit session_complete; the feed must not miss them."""
    entries = fetch(client, test_token)["entries"]

    chats = [e for e in entries if e["session_id"] == "chat-1" and e["type"] == "session"]
    assert len(chats) == 1
    chat = chats[0]
    assert chat["title"] == "Morning triage"
    assert chat["status"] == "ok"
    assert chat["label"] == "completed"


def test_a_successful_chat_summarizes_its_last_answer(client, test_token, history):
    """A success session_end carries no text; the summary comes from the final_result
    recorded just before it (the real ordering: final_result, then session_end)."""
    chat = history.create(agent_name="demo", model="test:model", session_id="chat-answered")
    chat.record("final_result", ts=at("09:52:00"), result="Deployed 3 services\nAll healthy", turns=2)
    chat.record("session_end", ts=at("09:53:00"), status="success")

    entries = fetch(client, test_token, types="session")["entries"]

    row = next(e for e in entries if e["session_id"] == "chat-answered")
    assert row["timestamp"] == at("09:53:00")
    assert row["summary"] == "Deployed 3 services"


def test_a_cancelled_chat_reads_as_cancelled(client, test_token, history):
    """A user-stopped turn records session_end status="cancelled"; the row must say so."""
    chat = history.create(agent_name="demo", model="test:model", session_id="chat-stopped")
    chat.record("session_end", ts=at("09:50:00"), status="cancelled", error_message="Cancelled by user")

    entries = fetch(client, test_token, types="session")["entries"]

    row = next(e for e in entries if e["session_id"] == "chat-stopped")
    assert row["status"] == "cancelled"
    assert row["label"] == "cancelled"
    assert row["summary"] == "Cancelled by user"


def test_a_max_turns_run_reads_as_failed(client, test_token, history):
    """session_end status="interrupted" means max_turns was hit, not a user cancel."""
    chat = history.create(agent_name="demo", model="test:model", session_id="chat-capped")
    chat.record("session_end", ts=at("09:51:00"), status="interrupted", error_message="max_turns (30) reached")

    entries = fetch(client, test_token, types="session")["entries"]

    row = next(e for e in entries if e["session_id"] == "chat-capped")
    assert row["status"] == "error"
    assert row["label"] == "failed"


def test_a_failed_background_run_carries_its_error(client, test_token):
    entries = fetch(client, test_token)["entries"]

    run = next(e for e in entries if e["session_id"] == "run-1" and e["type"] == "session")
    assert run["status"] == "error"
    assert run["label"] == "failed"
    assert run["summary"] == "timed out"


def test_entries_carry_a_link_target(client, test_token):
    entries = fetch(client, test_token)["entries"]
    newest = {}
    for entry in entries:
        newest.setdefault(entry["type"], entry)

    assert newest["job"]["job_id"] == "job-done"
    assert newest["job"]["session_id"] == "chat-1"
    assert newest["schedule"]["schedule_id"] == "feeds"
    assert newest["schedule"]["session_id"] == "run-1"
    assert newest["compaction"]["session_id"] == "chat-2"


def test_a_compaction_reads_as_its_own_row(client, test_token):
    entries = fetch(client, test_token, types="compaction")["entries"]

    assert [(e["title"], e["label"], e["summary"], e["session_id"]) for e in entries] == [
        ("Deploy check", "compacted", "12 turns compacted, 3 kept", "chat-2")
    ]


def test_terminal_jobs_only(client, test_token):
    entries = fetch(client, test_token, types="job")["entries"]

    assert [e["job_id"] for e in entries] == ["job-done", "job-errored"]
    assert entries[1]["label"] == "errored"
    assert entries[1]["summary"] == "worker died"


def test_a_chatty_session_does_not_crowd_out_older_ones(client, test_token, history):
    """A session that ends many turns must not starve the feed: the newest-per-session
    collapse happens in SQL, not over a fixed window of scanned events."""
    chat1 = history.load("chat-1")
    for minute in range(12):
        chat1.record("session_end", ts=at(f"10:{minute:02d}:00"), status="success")

    entries = fetch(client, test_token, types="session", limit=2)["entries"]

    assert [(e["session_id"], e["timestamp"]) for e in entries] == [
        ("chat-1", at("10:11:00")),
        ("run-1", at("09:00:00")),
    ]


def test_type_filter_restricts_the_feed(client, test_token):
    entries = fetch(client, test_token, types="schedule,compaction")["entries"]

    assert {e["type"] for e in entries} == {"schedule", "compaction"}


def test_unknown_type_is_rejected(client, test_token):
    resp = client.get("/api/activity?types=bogus", headers={"Authorization": f"Bearer {test_token}"})
    assert resp.status_code == 400
    assert "bogus" in resp.json()["error"]


def test_unparseable_limit_is_rejected(client, test_token):
    """A garbage limit 400s rather than silently serving the default window."""
    resp = client.get("/api/activity?limit=abc", headers={"Authorization": f"Bearer {test_token}"})
    assert resp.status_code == 400
    assert "limit" in resp.json()["error"]


def test_limit_bounds_the_feed(client, test_token):
    data = fetch(client, test_token, limit=2)

    assert [e["title"] for e in data["entries"]] == ["Morning triage", "ship the release"]


def test_missing_subsystems_degrade_to_an_empty_feed(tmp_path, token_store, test_token, history_dir):
    """No runner / jobs / scheduler wired (a bare test server): 200 with nothing, not 503."""
    s = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8485),
        adapter=None,
        webhook_store=WebhookStore(tmp_path / "webhooks.json"),
        token_store=token_store,
    )
    resp = TestClient(s.app).get("/api/activity", headers={"Authorization": f"Bearer {test_token}"})
    assert resp.status_code == 200
    assert resp.json() == {"entries": []}
