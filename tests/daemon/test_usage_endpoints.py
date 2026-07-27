"""Tests for the /api/usage/* endpoints backing the Usage tab.

Covers the new per-schedule breakdown route and the cache-split fields now
surfaced by the summary/total responses. The handlers read the process-wide
UsageStore singleton, so the fixture points it at a seeded temp DB.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPServer
from tsugite_daemon.auth import TokenStore
from tsugite_daemon.config import HTTPConfig
from tsugite_daemon.webhook_store import WebhookStore

import tsugite.usage.store as usage_store_mod
from tsugite.usage.store import UsageStore

SCHEDULER_SOURCE = "scheduler"


@pytest.fixture
def usage_db(tmp_path):
    """Replace the UsageStore singleton with a seeded temp-DB instance."""
    store = UsageStore(db_path=tmp_path / "usage.db")
    store.record(
        source=SCHEDULER_SOURCE,
        schedule_name="morning-report",
        total_tokens=3000,
        cost_usd=0.30,
        cache_creation_tokens=100,
        cache_read_tokens=900,
    )
    store.record(source=SCHEDULER_SOURCE, schedule_name="morning-report", total_tokens=2000, cost_usd=0.20)
    store.record(source=SCHEDULER_SOURCE, schedule_name=None, total_tokens=500, cost_usd=0.05)
    store.record(
        source="daemon", agent="odyn", total_tokens=1000, cost_usd=0.10, cache_creation_tokens=10, cache_read_tokens=40
    )
    prev = usage_store_mod._instance
    usage_store_mod._instance = store
    yield store
    usage_store_mod._instance = prev


@pytest.fixture
def token_store(tmp_path):
    return TokenStore(tmp_path / "tokens.json")


@pytest.fixture
def test_token(token_store):
    _st, raw = token_store.create_admin_token(name="usage-token")
    return raw


@pytest.fixture
def client(tmp_path, token_store, usage_db):
    server = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8385),
        adapters={},
        webhook_store=WebhookStore(tmp_path / "webhooks.json"),
        agent_configs={},
        token_store=token_store,
    )
    return TestClient(server.app)


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


class TestSchedulesEndpoint:
    def test_requires_auth(self, client):
        assert client.get("/api/usage/schedules").status_code == 401

    def test_returns_per_schedule_rows(self, client, test_token):
        resp = client.get("/api/usage/schedules", headers=_auth(test_token))
        assert resp.status_code == 200
        rows = resp.json()
        by_name = {r["schedule_name"]: r for r in rows}
        assert by_name["morning-report"]["runs"] == 2
        assert by_name["morning-report"]["total_tokens"] == 5000
        assert by_name["morning-report"]["cache_read_tokens"] == 900
        # Legacy scheduler runs with no marker aggregate under a null bucket.
        assert None in by_name

    def test_excludes_non_scheduler_usage(self, client, test_token):
        rows = client.get("/api/usage/schedules", headers=_auth(test_token)).json()
        # The source="daemon" row must not appear as a schedule.
        assert all(r["runs"] > 0 for r in rows)
        assert sum(r["runs"] for r in rows) == 3

    def test_default_limit_returns_full_breakdown_not_top_ten(self, usage_db, client, test_token):
        # A full per-schedule breakdown, unlike the top-N agents/models lists:
        # a user with many schedules must see all of them, not just 10.
        for i in range(15):
            usage_db.record(source=SCHEDULER_SOURCE, schedule_name=f"sched-{i}", total_tokens=100, cost_usd=0.01)
        rows = client.get("/api/usage/schedules", headers=_auth(test_token)).json()
        assert len([r for r in rows if (r["schedule_name"] or "").startswith("sched-")]) == 15


class TestCacheSplitInResponses:
    def test_total_exposes_cache_fields(self, client, test_token):
        total = client.get("/api/usage/total", headers=_auth(test_token)).json()
        assert total["cache_creation_tokens"] == 110
        assert total["cache_read_tokens"] == 940

    def test_summary_exposes_cache_fields(self, client, test_token):
        rows = client.get("/api/usage/summary", headers=_auth(test_token)).json()
        assert rows
        assert "cache_creation_tokens" in rows[0]
        assert "cache_read_tokens" in rows[0]
