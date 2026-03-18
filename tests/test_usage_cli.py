"""Tests for the `tsu usage` CLI commands."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from tsugite.cli import app
from tsugite.usage import UsageRecord, UsageStore

runner = CliRunner()


@pytest.fixture
def populated_store(tmp_path):
    """Create a UsageStore with sample data."""
    store = UsageStore(tmp_path / "usage.db")
    store.record(UsageRecord(
        timestamp="2026-03-15T10:00:00+00:00", agent="chat", model="openai:gpt-4o",
        input_tokens=1000, output_tokens=500, total_tokens=1500, cost=0.03, duration_ms=2000,
    ))
    store.record(UsageRecord(
        timestamp="2026-03-14T08:00:00+00:00", agent="digest", model="anthropic:claude-3-sonnet",
        input_tokens=500, output_tokens=200, total_tokens=700, cost=0.01, duration_ms=1500,
        schedule_id="morning",
    ))
    store.record(UsageRecord(
        timestamp="2026-03-13T06:00:00+00:00", agent="chat", model="openai:gpt-4o",
        input_tokens=2000, output_tokens=800, total_tokens=2800, cost=0.05, duration_ms=3000,
    ))
    return store


@pytest.fixture
def empty_store(tmp_path):
    """Create an empty UsageStore."""
    return UsageStore(tmp_path / "empty_usage.db")


# ── tsu usage summary ──


class TestUsageSummary:
    def test_default_output(self, populated_store):
        with patch("tsugite.usage.get_usage_store", return_value=populated_store):
            result = runner.invoke(app, ["usage", "summary"])
        assert result.exit_code == 0
        assert "Usage Summary" in result.stdout
        assert "3" in result.stdout  # run_count
        assert "$0.09" in result.stdout  # total cost

    def test_agent_filter(self, populated_store):
        with patch("tsugite.usage.get_usage_store", return_value=populated_store):
            result = runner.invoke(app, ["usage", "summary", "--agent", "chat"])
        assert result.exit_code == 0
        assert "agent=chat" in result.stdout
        assert "2" in result.stdout  # 2 chat runs

    def test_schedule_filter(self, populated_store):
        with patch("tsugite.usage.get_usage_store", return_value=populated_store):
            result = runner.invoke(app, ["usage", "summary", "--schedule", "morning"])
        assert result.exit_code == 0
        assert "1" in result.stdout  # 1 scheduled run

    def test_since_relative(self, populated_store):
        with patch("tsugite.usage.get_usage_store", return_value=populated_store):
            result = runner.invoke(app, ["usage", "summary", "--since", "7d"])
        assert result.exit_code == 0

    def test_since_absolute(self, populated_store):
        with patch("tsugite.usage.get_usage_store", return_value=populated_store):
            result = runner.invoke(app, ["usage", "summary", "--since", "2026-03-14"])
        assert result.exit_code == 0

    def test_empty_store_message(self, empty_store):
        with patch("tsugite.usage.get_usage_store", return_value=empty_store):
            result = runner.invoke(app, ["usage", "summary"])
        assert result.exit_code == 0
        assert "No usage data" in result.stdout

    def test_cost_formatting(self, populated_store):
        with patch("tsugite.usage.get_usage_store", return_value=populated_store):
            result = runner.invoke(app, ["usage", "summary"])
        assert result.exit_code == 0
        # Should contain dollar sign and formatted cost
        assert "$" in result.stdout


# ── tsu usage history ──


class TestUsageHistory:
    def test_default_output(self, populated_store):
        with patch("tsugite.usage.get_usage_store", return_value=populated_store):
            result = runner.invoke(app, ["usage", "history"])
        assert result.exit_code == 0
        assert "Usage History" in result.stdout
        assert "chat" in result.stdout
        assert "digest" in result.stdout

    def test_limit(self, populated_store):
        with patch("tsugite.usage.get_usage_store", return_value=populated_store):
            result = runner.invoke(app, ["usage", "history", "--limit", "1"])
        assert result.exit_code == 0
        assert "showing 1" in result.stdout

    def test_agent_filter(self, populated_store):
        with patch("tsugite.usage.get_usage_store", return_value=populated_store):
            result = runner.invoke(app, ["usage", "history", "--agent", "digest"])
        assert result.exit_code == 0
        assert "digest" in result.stdout

    def test_since_filter(self, populated_store):
        with patch("tsugite.usage.get_usage_store", return_value=populated_store):
            result = runner.invoke(app, ["usage", "history", "--since", "2026-03-14"])
        assert result.exit_code == 0

    def test_empty_results(self, empty_store):
        with patch("tsugite.usage.get_usage_store", return_value=empty_store):
            result = runner.invoke(app, ["usage", "history"])
        assert result.exit_code == 0
        assert "No usage records" in result.stdout


# ── tsu usage breakdown ──


class TestUsageBreakdown:
    def test_period_day(self, populated_store):
        with patch("tsugite.usage.get_usage_store", return_value=populated_store):
            result = runner.invoke(app, ["usage", "breakdown", "--period", "day"])
        assert result.exit_code == 0
        assert "Usage by day" in result.stdout

    def test_period_week(self, populated_store):
        with patch("tsugite.usage.get_usage_store", return_value=populated_store):
            result = runner.invoke(app, ["usage", "breakdown", "--period", "week"])
        assert result.exit_code == 0
        assert "Usage by week" in result.stdout

    def test_period_month(self, populated_store):
        with patch("tsugite.usage.get_usage_store", return_value=populated_store):
            result = runner.invoke(app, ["usage", "breakdown", "--period", "month"])
        assert result.exit_code == 0
        assert "Usage by month" in result.stdout

    def test_invalid_period(self, populated_store):
        with patch("tsugite.usage.get_usage_store", return_value=populated_store):
            result = runner.invoke(app, ["usage", "breakdown", "--period", "year"])
        assert result.exit_code == 1

    def test_agent_filter(self, populated_store):
        with patch("tsugite.usage.get_usage_store", return_value=populated_store):
            result = runner.invoke(app, ["usage", "breakdown", "--agent", "chat"])
        assert result.exit_code == 0

    def test_empty_results(self, empty_store):
        with patch("tsugite.usage.get_usage_store", return_value=empty_store):
            result = runner.invoke(app, ["usage", "breakdown"])
        assert result.exit_code == 0
        assert "No usage data" in result.stdout
