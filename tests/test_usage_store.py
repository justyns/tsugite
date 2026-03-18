"""Thorough tests for the UsageStore SQLite backend."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tsugite.usage import UsageRecord, UsageStore


@pytest.fixture
def store(tmp_path):
    """Create a UsageStore with a temporary DB."""
    return UsageStore(tmp_path / "usage.db")


@pytest.fixture
def sample_record():
    """Create a sample UsageRecord."""
    return UsageRecord(
        timestamp="2026-03-15T10:00:00+00:00",
        agent="chat",
        model="openai:gpt-4o",
        session_id="session-abc123",
        schedule_id=None,
        input_tokens=1000,
        output_tokens=500,
        total_tokens=1500,
        cached_tokens=200,
        cost=0.03,
        duration_ms=2500,
    )


def _make_record(
    agent="chat",
    model="openai:gpt-4o",
    timestamp="2026-03-15T10:00:00+00:00",
    input_tokens=1000,
    output_tokens=500,
    total_tokens=1500,
    cached_tokens=0,
    cost=0.03,
    duration_ms=2500,
    session_id=None,
    schedule_id=None,
):
    return UsageRecord(
        timestamp=timestamp,
        agent=agent,
        model=model,
        session_id=session_id,
        schedule_id=schedule_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        cost=cost,
        duration_ms=duration_ms,
    )


# ── Schema & Initialization ──


class TestSchemaAndInit:
    def test_db_created_lazily_on_first_record(self, tmp_path):
        db_path = tmp_path / "usage.db"
        store = UsageStore(db_path)
        assert not db_path.exists()

        store.record(_make_record())
        assert db_path.exists()

    def test_table_exists_after_init(self, store, sample_record):
        store.record(sample_record)
        import sqlite3

        conn = sqlite3.connect(str(store._path))
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='usage_log'")
        assert cursor.fetchone() is not None
        conn.close()

    def test_schema_version_set(self, store, sample_record):
        store.record(sample_record)
        import sqlite3

        conn = sqlite3.connect(str(store._path))
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == 1
        conn.close()

    def test_read_returns_empty_when_db_does_not_exist(self, tmp_path):
        store = UsageStore(tmp_path / "nonexistent.db")
        assert store.query() == []
        assert store.summary() == {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "total_cost": 0.0,
            "run_count": 0,
            "avg_cost_per_run": 0.0,
        }
        assert store.aggregate() == []


# ── record() — Insert ──


class TestRecord:
    def test_record_single_entry(self, store, sample_record):
        store.record(sample_record)
        records = store.query()
        assert len(records) == 1
        r = records[0]
        assert r.agent == "chat"
        assert r.model == "openai:gpt-4o"
        assert r.input_tokens == 1000
        assert r.output_tokens == 500
        assert r.total_tokens == 1500
        assert r.cached_tokens == 200
        assert r.cost == 0.03
        assert r.duration_ms == 2500
        assert r.session_id == "session-abc123"
        assert r.schedule_id is None
        assert r.id is not None

    def test_record_with_none_optional_fields(self, store):
        record = UsageRecord(
            timestamp="2026-03-15T10:00:00+00:00",
            agent="test",
            model="openai:gpt-4o",
        )
        store.record(record)
        records = store.query()
        assert len(records) == 1
        r = records[0]
        assert r.session_id is None
        assert r.schedule_id is None
        assert r.duration_ms is None
        assert r.input_tokens == 0
        assert r.output_tokens == 0
        assert r.cost == 0.0

    def test_record_multiple_agents_and_models(self, store):
        store.record(_make_record(agent="chat", model="openai:gpt-4o", cost=0.05))
        store.record(_make_record(agent="digest", model="anthropic:claude-3-sonnet", cost=0.01))
        store.record(_make_record(agent="chat", model="openai:gpt-4o-mini", cost=0.002))
        records = store.query()
        assert len(records) == 3

    def test_cost_stored_as_float(self, store):
        store.record(_make_record(cost=0.000123))
        records = store.query()
        assert abs(records[0].cost - 0.000123) < 1e-10


# ── query() — Filtered Retrieval ──


class TestQuery:
    def _populate(self, store):
        store.record(_make_record(agent="chat", model="openai:gpt-4o", timestamp="2026-03-10T10:00:00+00:00", cost=0.01, schedule_id=None))
        store.record(_make_record(agent="digest", model="anthropic:claude-3-sonnet", timestamp="2026-03-12T10:00:00+00:00", cost=0.02, schedule_id="morning"))
        store.record(_make_record(agent="chat", model="openai:gpt-4o", timestamp="2026-03-14T10:00:00+00:00", cost=0.03, schedule_id=None))
        store.record(_make_record(agent="digest", model="openai:gpt-4o", timestamp="2026-03-15T10:00:00+00:00", cost=0.04, schedule_id="morning"))

    def test_query_all(self, store):
        self._populate(store)
        records = store.query()
        assert len(records) == 4

    def test_filter_by_agent(self, store):
        self._populate(store)
        records = store.query(agent="chat")
        assert len(records) == 2
        assert all(r.agent == "chat" for r in records)

    def test_filter_by_schedule_id(self, store):
        self._populate(store)
        records = store.query(schedule_id="morning")
        assert len(records) == 2
        assert all(r.schedule_id == "morning" for r in records)

    def test_filter_by_since(self, store):
        self._populate(store)
        records = store.query(since="2026-03-13T00:00:00+00:00")
        assert len(records) == 2

    def test_filter_by_until(self, store):
        self._populate(store)
        records = store.query(until="2026-03-11T00:00:00+00:00")
        assert len(records) == 1

    def test_filter_by_model(self, store):
        self._populate(store)
        records = store.query(model="anthropic:claude-3-sonnet")
        assert len(records) == 1
        assert records[0].agent == "digest"

    def test_combined_filters(self, store):
        self._populate(store)
        records = store.query(agent="digest", schedule_id="morning", since="2026-03-14T00:00:00+00:00")
        assert len(records) == 1
        assert records[0].cost == 0.04

    def test_limit_caps_results(self, store):
        self._populate(store)
        records = store.query(limit=2)
        assert len(records) == 2

    def test_results_ordered_by_timestamp_desc(self, store):
        self._populate(store)
        records = store.query()
        timestamps = [r.timestamp for r in records]
        assert timestamps == sorted(timestamps, reverse=True)


# ── summary() — Aggregate Totals ──


class TestSummary:
    def test_correct_totals(self, store):
        store.record(_make_record(input_tokens=100, output_tokens=50, total_tokens=150, cost=0.01))
        store.record(_make_record(input_tokens=200, output_tokens=100, total_tokens=300, cost=0.02))
        data = store.summary()
        assert data["run_count"] == 2
        assert data["total_tokens"] == 450
        assert data["input_tokens"] == 300
        assert data["output_tokens"] == 150
        assert abs(data["total_cost"] - 0.03) < 1e-10
        assert abs(data["avg_cost_per_run"] - 0.015) < 1e-10

    def test_filtered_by_agent(self, store):
        store.record(_make_record(agent="chat", cost=0.01))
        store.record(_make_record(agent="digest", cost=0.02))
        data = store.summary(agent="chat")
        assert data["run_count"] == 1
        assert abs(data["total_cost"] - 0.01) < 1e-10

    def test_filtered_by_schedule_id(self, store):
        store.record(_make_record(schedule_id="morning", cost=0.01))
        store.record(_make_record(schedule_id=None, cost=0.02))
        data = store.summary(schedule_id="morning")
        assert data["run_count"] == 1

    def test_filtered_by_since(self, store):
        store.record(_make_record(timestamp="2026-03-01T10:00:00+00:00", cost=0.01))
        store.record(_make_record(timestamp="2026-03-15T10:00:00+00:00", cost=0.02))
        data = store.summary(since="2026-03-10T00:00:00+00:00")
        assert data["run_count"] == 1
        assert abs(data["total_cost"] - 0.02) < 1e-10

    def test_empty_result(self, store):
        data = store.summary(agent="nonexistent")
        assert data["run_count"] == 0
        assert data["total_cost"] == 0.0
        assert data["avg_cost_per_run"] == 0.0

    def test_cost_across_different_models(self, store):
        store.record(_make_record(model="openai:gpt-4o", cost=0.05))
        store.record(_make_record(model="openai:gpt-4o-mini", cost=0.002))
        store.record(_make_record(model="anthropic:claude-3-sonnet", cost=0.01))
        data = store.summary()
        assert abs(data["total_cost"] - 0.062) < 1e-10
        assert data["run_count"] == 3


# ── aggregate() — Time-Period Grouping ──


class TestAggregate:
    def test_group_by_day(self, store):
        store.record(_make_record(timestamp="2026-03-15T08:00:00+00:00", cost=0.01, total_tokens=100))
        store.record(_make_record(timestamp="2026-03-15T14:00:00+00:00", cost=0.02, total_tokens=200))
        store.record(_make_record(timestamp="2026-03-16T10:00:00+00:00", cost=0.03, total_tokens=300))
        rows = store.aggregate(group_by="day")
        assert len(rows) == 2
        assert rows[0]["period"] == "2026-03-15"
        assert rows[0]["run_count"] == 2
        assert rows[0]["total_tokens"] == 300
        assert abs(rows[0]["total_cost"] - 0.03) < 1e-10
        assert rows[1]["period"] == "2026-03-16"
        assert rows[1]["run_count"] == 1

    def test_group_by_week(self, store):
        # Week 10 (Mon Mar 9 - Sun Mar 15) and Week 11 (Mon Mar 16+)
        store.record(_make_record(timestamp="2026-03-10T10:00:00+00:00", cost=0.01))
        store.record(_make_record(timestamp="2026-03-17T10:00:00+00:00", cost=0.02))
        rows = store.aggregate(group_by="week")
        assert len(rows) == 2

    def test_group_by_month(self, store):
        store.record(_make_record(timestamp="2026-02-15T10:00:00+00:00", cost=0.01))
        store.record(_make_record(timestamp="2026-03-15T10:00:00+00:00", cost=0.02))
        rows = store.aggregate(group_by="month")
        assert len(rows) == 2
        assert rows[0]["period"] == "2026-02"
        assert rows[1]["period"] == "2026-03"

    def test_each_group_has_required_fields(self, store):
        store.record(_make_record(input_tokens=100, output_tokens=50, total_tokens=150, cost=0.01))
        rows = store.aggregate(group_by="day")
        assert len(rows) == 1
        row = rows[0]
        assert "period" in row
        assert "run_count" in row
        assert "total_tokens" in row
        assert "input_tokens" in row
        assert "output_tokens" in row
        assert "total_cost" in row

    def test_filtered_by_agent(self, store):
        store.record(_make_record(agent="chat", timestamp="2026-03-15T10:00:00+00:00"))
        store.record(_make_record(agent="digest", timestamp="2026-03-15T12:00:00+00:00"))
        rows = store.aggregate(agent="chat")
        assert len(rows) == 1
        assert rows[0]["run_count"] == 1

    def test_filtered_by_schedule_id(self, store):
        store.record(_make_record(schedule_id="morning", timestamp="2026-03-15T10:00:00+00:00"))
        store.record(_make_record(schedule_id=None, timestamp="2026-03-15T12:00:00+00:00"))
        rows = store.aggregate(schedule_id="morning")
        assert len(rows) == 1
        assert rows[0]["run_count"] == 1

    def test_empty_periods_not_included(self, store):
        store.record(_make_record(timestamp="2026-03-10T10:00:00+00:00"))
        store.record(_make_record(timestamp="2026-03-15T10:00:00+00:00"))
        rows = store.aggregate(group_by="day")
        assert len(rows) == 2  # Only days with data, not gap days


# ── Thread Safety ──


class TestThreadSafety:
    def test_concurrent_writes(self, store):
        """Concurrent record() calls should not corrupt data."""

        def write_record(i):
            store.record(
                _make_record(
                    agent=f"agent-{i}",
                    timestamp=f"2026-03-15T{i:02d}:00:00+00:00",
                    cost=0.01 * i,
                )
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(write_record, range(20)))

        records = store.query(limit=100)
        assert len(records) == 20

    def test_concurrent_read_write(self, store):
        """Reads during writes should not error."""
        store.record(_make_record())

        errors = []

        def writer():
            for i in range(10):
                try:
                    store.record(_make_record(agent=f"w-{i}", timestamp=f"2026-03-15T{i:02d}:00:00+00:00"))
                except Exception as e:
                    errors.append(e)

        def reader():
            for _ in range(10):
                try:
                    store.query()
                    store.summary()
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ── Edge Cases ──


class TestEdgeCases:
    def test_zero_token_records(self, store):
        store.record(_make_record(input_tokens=0, output_tokens=0, total_tokens=0, cost=0.0))
        records = store.query()
        assert len(records) == 1
        assert records[0].total_tokens == 0

    def test_very_large_token_counts(self, store):
        store.record(_make_record(input_tokens=10_000_000, output_tokens=5_000_000, total_tokens=15_000_000))
        records = store.query()
        assert records[0].total_tokens == 15_000_000

    def test_unicode_agent_names(self, store):
        store.record(_make_record(agent="日本語エージェント"))
        records = store.query(agent="日本語エージェント")
        assert len(records) == 1
        assert records[0].agent == "日本語エージェント"

    def test_cost_zero_vs_default(self, store):
        store.record(_make_record(cost=0.0))
        records = store.query()
        assert records[0].cost == 0.0
