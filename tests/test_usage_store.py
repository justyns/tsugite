"""Tests for UsageStore aggregation: per-schedule breakdown and cache split.

The store is SQLite-backed with dedicated columns (schedule_name, source,
cache_creation_tokens, cache_read_tokens); these tests lock the aggregation
queries that surface them in the daemon Usage tab.
"""

from __future__ import annotations

import pytest

from tsugite.usage.store import UsageStore

SCHEDULER_SOURCE = "scheduler"


@pytest.fixture
def store(tmp_path):
    return UsageStore(db_path=tmp_path / "usage.db")


def _record(store, **kw):
    """Record with sensible defaults so each test only states what it cares about."""
    defaults = dict(agent="odyn", model="claude_code:opus", total_tokens=1000, cost_usd=0.10)
    defaults.update(kw)
    store.record(**defaults)


class TestBySchedule:
    def test_groups_scheduler_runs_by_schedule_name(self, store):
        _record(store, source=SCHEDULER_SOURCE, schedule_name="morning-report", cost_usd=0.20, total_tokens=2000)
        _record(store, source=SCHEDULER_SOURCE, schedule_name="morning-report", cost_usd=0.30, total_tokens=3000)
        _record(store, source=SCHEDULER_SOURCE, schedule_name="nightly-digest", cost_usd=0.10, total_tokens=1000)

        rows = store.by_schedule()
        by_name = {r["schedule_name"]: r for r in rows}

        assert by_name["morning-report"]["runs"] == 2
        assert by_name["morning-report"]["total_tokens"] == 5000
        assert by_name["morning-report"]["total_cost"] == pytest.approx(0.50)
        assert by_name["nightly-digest"]["runs"] == 1

    def test_ordered_by_cost_descending(self, store):
        _record(store, source=SCHEDULER_SOURCE, schedule_name="cheap", cost_usd=0.05)
        _record(store, source=SCHEDULER_SOURCE, schedule_name="pricey", cost_usd=5.00)

        rows = store.by_schedule()

        assert [r["schedule_name"] for r in rows] == ["pricey", "cheap"]

    def test_reports_last_run_timestamp(self, store):
        _record(store, source=SCHEDULER_SOURCE, schedule_name="daily", timestamp="2026-07-10T08:00:00+00:00")
        _record(store, source=SCHEDULER_SOURCE, schedule_name="daily", timestamp="2026-07-15T08:00:00+00:00")

        row = store.by_schedule()[0]

        assert row["last_run"] == "2026-07-15T08:00:00+00:00"

    def test_includes_cache_split_per_schedule(self, store):
        _record(
            store,
            source=SCHEDULER_SOURCE,
            schedule_name="daily",
            cache_creation_tokens=100,
            cache_read_tokens=900,
        )
        _record(
            store,
            source=SCHEDULER_SOURCE,
            schedule_name="daily",
            cache_creation_tokens=50,
            cache_read_tokens=500,
        )

        row = store.by_schedule()[0]

        assert row["cache_creation_tokens"] == 150
        assert row["cache_read_tokens"] == 1400

    def test_excludes_non_scheduler_sources(self, store):
        _record(store, source="daemon", schedule_name=None)
        _record(store, source="cli", schedule_name=None)
        _record(store, source=SCHEDULER_SOURCE, schedule_name="daily")

        rows = store.by_schedule()

        assert [r["schedule_name"] for r in rows] == ["daily"]

    def test_unattributed_bucket_for_legacy_scheduler_runs(self, store):
        # Runs recorded before the schedule_id marker existed: source is
        # "scheduler" but schedule_name is NULL. They must show as their own
        # bucket, never be dropped or mis-merged into a named schedule.
        _record(store, source=SCHEDULER_SOURCE, schedule_name=None, cost_usd=0.40)
        _record(store, source=SCHEDULER_SOURCE, schedule_name="daily", cost_usd=0.10)

        rows = store.by_schedule()
        by_name = {r["schedule_name"]: r for r in rows}

        assert None in by_name
        assert by_name[None]["runs"] == 1
        assert by_name[None]["total_cost"] == pytest.approx(0.40)

    def test_since_filter(self, store):
        _record(store, source=SCHEDULER_SOURCE, schedule_name="old", timestamp="2026-06-01T00:00:00+00:00")
        _record(store, source=SCHEDULER_SOURCE, schedule_name="new", timestamp="2026-07-15T00:00:00+00:00")

        rows = store.by_schedule(since="2026-07-01")

        assert [r["schedule_name"] for r in rows] == ["new"]

    def test_limit_caps_rows(self, store):
        for i in range(5):
            _record(store, source=SCHEDULER_SOURCE, schedule_name=f"sched-{i}", cost_usd=float(i))

        rows = store.by_schedule(limit=2)

        assert len(rows) == 2

    def test_empty_when_no_scheduler_runs(self, store):
        _record(store, source="daemon")

        assert store.by_schedule() == []


class TestCacheSplit:
    def test_summary_sums_cache_columns(self, store):
        _record(store, cache_creation_tokens=100, cache_read_tokens=900, timestamp="2026-07-15T08:00:00+00:00")
        _record(store, cache_creation_tokens=50, cache_read_tokens=500, timestamp="2026-07-15T09:00:00+00:00")

        row = store.summary(period="day")[0]

        assert row["cache_creation_tokens"] == 150
        assert row["cache_read_tokens"] == 1400

    def test_total_sums_cache_columns(self, store):
        _record(store, cache_creation_tokens=100, cache_read_tokens=900)
        _record(store, cache_creation_tokens=50, cache_read_tokens=500)

        total = store.total()

        assert total["cache_creation_tokens"] == 150
        assert total["cache_read_tokens"] == 1400

    def test_top_models_includes_cache_columns(self, store):
        _record(store, model="claude_code:opus", cache_creation_tokens=100, cache_read_tokens=900)

        row = store.top_models()[0]

        assert row["cache_creation_tokens"] == 100
        assert row["cache_read_tokens"] == 900

    def test_cache_defaults_to_zero_when_provider_reports_none(self, store):
        # A provider that doesn't report cache fields records the column default
        # (0). The aggregation can't distinguish this from a genuine zero - the
        # store treats both as 0, which the endpoint docstring documents.
        _record(store)

        total = store.total()

        assert total["cache_creation_tokens"] == 0
        assert total["cache_read_tokens"] == 0
