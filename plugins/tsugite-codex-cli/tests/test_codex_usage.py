"""Tests for the Codex quota source."""

import json
import logging

import httpx
import pytest
from tsugite_codex_cli import usage
from tsugite_codex_cli.auth import CodexAuthStore

from tsugite.usage.quota import collect_quota_reports, register_quota_source, reset_quota_sources

FAKE_TOKEN = "example-access-token"
FAKE_ACCOUNT_ID = "example-account-id"

RATE_LIMIT = {
    "allowed": True,
    "limit_reached": False,
    "primary_window": {
        "used_percent": 11,
        "limit_window_seconds": 18000,
        "reset_after_seconds": 11471,
        "reset_at": 1767225600,
    },
    "secondary_window": {
        "used_percent": 7,
        "limit_window_seconds": 604800,
        "reset_after_seconds": 578114,
        "reset_at": 1767830400,
    },
}


@pytest.fixture(autouse=True)
def _no_live_result(monkeypatch):
    monkeypatch.setattr(usage, "_last_live", None)


@pytest.fixture
def signed_in(monkeypatch):
    monkeypatch.setattr(CodexAuthStore, "sync_get_or_refresh", lambda self: (FAKE_TOKEN, FAKE_ACCOUNT_ID))


def _serve(monkeypatch, rate_limit=RATE_LIMIT, status=200):
    calls = []

    def fake_get(url, headers, timeout):
        calls.append((url, headers))
        body = {"plan_type": "plus", "rate_limit": rate_limit, "credits": {"has_credits": False}}
        return httpx.Response(status, json=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    return calls


@pytest.fixture
def windows(signed_in, monkeypatch):
    _serve(monkeypatch)
    return {w.key: w for w in usage.fetch().windows}


def test_primary_window_is_labeled_by_its_duration(windows):
    assert windows["primary"].label == "5h"
    assert windows["primary"].used_pct == 11.0
    assert windows["primary"].resets_at == "2026-01-01T00:00:00+00:00"


def test_seven_day_window_is_labeled_week(windows):
    assert windows["secondary"].label == "week"
    assert windows["secondary"].used_pct == 7.0
    assert windows["secondary"].resets_at == "2026-01-08T00:00:00+00:00"


@pytest.mark.parametrize(("seconds", "label"), [(86400, "24h"), (2700, "45m")])
def test_other_durations_are_labeled_in_hours_or_minutes(signed_in, monkeypatch, seconds, label):
    window = {**RATE_LIMIT["primary_window"], "limit_window_seconds": seconds}
    _serve(monkeypatch, rate_limit={"primary_window": window, "secondary_window": None})

    assert [w.label for w in usage.fetch().windows] == [label]


def test_a_null_window_is_skipped(signed_in, monkeypatch):
    _serve(monkeypatch, rate_limit={**RATE_LIMIT, "secondary_window": None})

    report = usage.fetch()

    assert report.error is None
    assert [w.key for w in report.windows] == ["primary"]


def test_usage_endpoint_is_read_with_the_codex_token_and_account(signed_in, monkeypatch):
    calls = _serve(monkeypatch)

    usage.fetch()

    url, headers = calls[0]
    assert url == usage.USAGE_URL
    assert headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
    assert headers["ChatGPT-Account-ID"] == FAKE_ACCOUNT_ID


def test_live_result_is_reused_within_the_ttl(signed_in, monkeypatch):
    calls = _serve(monkeypatch)

    usage.fetch()
    usage.fetch()

    assert len(calls) == 1


def test_missing_codex_login_reports_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))

    report = usage.fetch()

    assert report.windows == ()
    assert report.error


def test_unexpected_shape_reports_an_error(signed_in, monkeypatch):
    _serve(monkeypatch, rate_limit={"primary_window": {"used_percent": 11}})

    report = usage.fetch()

    assert report.windows == ()
    assert report.error


def test_rejected_request_keeps_credentials_out_of_the_report_and_logs(signed_in, monkeypatch, caplog):
    _serve(monkeypatch, status=401)
    reset_quota_sources()
    register_quota_source(usage.PROVIDER, usage.LABEL, usage.fetch)

    with caplog.at_level(logging.DEBUG):
        row = next(r for r in collect_quota_reports() if r["provider"] == usage.PROVIDER)

    assert row["error"]
    for secret in (FAKE_TOKEN, FAKE_ACCOUNT_ID):
        assert secret not in json.dumps(row)
        assert secret not in caplog.text
