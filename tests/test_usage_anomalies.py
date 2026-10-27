"""Tests for daily spend anomaly detection over usage.db."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from tsugite.usage.anomalies import detect_spend_anomalies
from tsugite.usage.store import UsageStore

TODAY = date(2026, 9, 20)


@pytest.fixture
def store(tmp_path):
    return UsageStore(db_path=tmp_path / "usage.db")


def _seed_history(store, model, costs):
    """Record one row per trailing day, most recent cost last, ending the day before TODAY."""
    start = TODAY - timedelta(days=len(costs))
    for i, cost in enumerate(costs):
        day = start + timedelta(days=i)
        store.record(
            agent="hollis",
            model=model,
            cost_usd=cost,
            timestamp=f"{day.isoformat()}T12:00:00+00:00",
        )


def _record_today(store, model, cost):
    store.record(agent="hollis", model=model, cost_usd=cost, timestamp=f"{TODAY.isoformat()}T12:00:00+00:00")


class TestThresholds:
    def test_flags_a_day_at_or_above_the_multiple(self, store):
        _seed_history(store, "anthropic:claude-opus-4-8", [50.0] * 14)
        _record_today(store, "anthropic:claude-opus-4-8", 200.0)

        results = detect_spend_anomalies(store, TODAY, multiple=3.0)

        assert len(results) == 1
        assert results[0].model == "anthropic:claude-opus-4-8"
        assert results[0].rule == "multiple"

    def test_does_not_flag_a_day_below_the_multiple(self, store):
        _seed_history(store, "anthropic:claude-opus-4-8", [50.0] * 14)
        _record_today(store, "anthropic:claude-opus-4-8", 100.0)

        results = detect_spend_anomalies(store, TODAY, multiple=3.0)

        assert results == []

    def test_flags_a_day_above_the_absolute_ceiling(self, store):
        _seed_history(store, "anthropic:claude-opus-4-8", [10.0] * 14)
        _record_today(store, "anthropic:claude-opus-4-8", 60.0)

        results = detect_spend_anomalies(store, TODAY, multiple=100.0, ceiling=50.0)

        assert len(results) == 1
        assert results[0].rule == "ceiling"

    def test_reports_both_rules_when_both_trip(self, store):
        _seed_history(store, "anthropic:claude-opus-4-8", [10.0] * 14)
        _record_today(store, "anthropic:claude-opus-4-8", 100.0)

        results = detect_spend_anomalies(store, TODAY, multiple=3.0, ceiling=50.0)

        assert results[0].rule == "multiple+ceiling"

    def test_no_ceiling_means_only_the_multiple_rule_applies(self, store):
        _seed_history(store, "anthropic:claude-opus-4-8", [10.0] * 14)
        _record_today(store, "anthropic:claude-opus-4-8", 15.0)

        results = detect_spend_anomalies(store, TODAY, multiple=3.0, ceiling=None)

        assert results == []


class TestThinHistory:
    def test_a_model_with_under_3_days_of_history_is_not_flagged_by_the_multiple_rule(self, store):
        _seed_history(store, "anthropic:claude-opus-4-8", [10.0, 10.0])
        _record_today(store, "anthropic:claude-opus-4-8", 1000.0)

        results = detect_spend_anomalies(store, TODAY, multiple=3.0)

        assert results == []

    def test_thin_history_can_still_trip_the_ceiling_rule(self, store):
        _seed_history(store, "anthropic:claude-opus-4-8", [10.0, 10.0])
        _record_today(store, "anthropic:claude-opus-4-8", 1000.0)

        results = detect_spend_anomalies(store, TODAY, multiple=3.0, ceiling=50.0)

        assert len(results) == 1
        assert results[0].rule == "ceiling"
        assert results[0].trailing_median is None
        assert results[0].ratio is None

    def test_exactly_3_days_of_history_is_usable(self, store):
        _seed_history(store, "anthropic:claude-opus-4-8", [10.0, 10.0, 10.0])
        _record_today(store, "anthropic:claude-opus-4-8", 100.0)

        results = detect_spend_anomalies(store, TODAY, multiple=3.0)

        assert len(results) == 1
        assert results[0].trailing_median == 10.0

    def test_no_history_at_all_and_no_ceiling_means_nothing_flags(self, store):
        _record_today(store, "anthropic:claude-opus-4-8", 1000.0)

        results = detect_spend_anomalies(store, TODAY, multiple=3.0, ceiling=None)

        assert results == []


class TestRatio:
    def test_ratio_is_day_total_over_trailing_median(self, store):
        _seed_history(store, "anthropic:claude-opus-4-8", [40.0, 50.0, 60.0])
        _record_today(store, "anthropic:claude-opus-4-8", 250.0)

        results = detect_spend_anomalies(store, TODAY, multiple=3.0)

        assert results[0].trailing_median == 50.0
        assert results[0].ratio == 5.0

    def test_no_spend_today_produces_no_record(self, store):
        _seed_history(store, "anthropic:claude-opus-4-8", [50.0] * 14)

        results = detect_spend_anomalies(store, TODAY, multiple=3.0)

        assert results == []


class TestZeroMedian:
    def test_spend_after_an_all_free_trailing_window_is_not_flagged_by_the_multiple_rule(self, store):
        _seed_history(store, "anthropic:claude-opus-4-8", [0.0] * 14)
        _record_today(store, "anthropic:claude-opus-4-8", 0.01)

        assert detect_spend_anomalies(store, TODAY, multiple=3.0) == []

    def test_spend_after_an_all_free_trailing_window_can_still_trip_the_ceiling(self, store):
        _seed_history(store, "anthropic:claude-opus-4-8", [0.0] * 14)
        _record_today(store, "anthropic:claude-opus-4-8", 5.0)

        results = detect_spend_anomalies(store, TODAY, multiple=3.0, ceiling=1.0)

        assert [r.rule for r in results] == ["ceiling"]
        assert results[0].trailing_median == 0.0
        assert results[0].ratio is None

    def test_no_spend_after_an_all_free_trailing_window_does_not_flag(self, store):
        _seed_history(store, "anthropic:claude-opus-4-8", [0.0] * 14)
        _record_today(store, "anthropic:claude-opus-4-8", 0.0)

        results = detect_spend_anomalies(store, TODAY, multiple=3.0)

        assert results == []


class TestUnreportedCost:
    def test_days_with_no_reported_cost_stay_out_of_the_history(self, store):
        _seed_history(store, "anthropic:claude-opus-4-8", [None] * 10 + [5.0] * 4)
        _record_today(store, "anthropic:claude-opus-4-8", 6.0)

        assert detect_spend_anomalies(store, TODAY, multiple=3.0) == []


class TestMultipleModels:
    def test_only_the_tripped_model_is_reported(self, store):
        _seed_history(store, "anthropic:claude-opus-4-8", [50.0] * 14)
        _record_today(store, "anthropic:claude-opus-4-8", 60.0)
        _seed_history(store, "openai:gpt-4o", [10.0] * 14)
        _record_today(store, "openai:gpt-4o", 100.0)

        results = detect_spend_anomalies(store, TODAY, multiple=3.0)

        assert [r.model for r in results] == ["openai:gpt-4o"]
