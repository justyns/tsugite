"""Tests for the Claude Code quota source."""

import json
import time

import httpx
import pytest
from tsugite_claude_code import usage

from tsugite.usage.quota import collect_quota_reports, register_quota_source, reset_quota_sources

FAKE_ACCOUNT_UUID = "11111111-2222-4333-8444-555555555555"

LIMITS = [
    {
        "kind": "session",
        "group": "session",
        "percent": 0,
        "severity": "normal",
        "resets_at": None,
        "scope": None,
        "is_active": False,
    },
    {
        "kind": "weekly_all",
        "group": "weekly",
        "percent": 42,
        "severity": "normal",
        "resets_at": "2026-01-08T12:00:00+00:00",
        "scope": None,
        "is_active": True,
    },
    {
        "kind": "weekly_scoped",
        "group": "weekly",
        "percent": 7,
        "severity": "normal",
        "resets_at": "2026-01-08T12:00:00+00:00",
        "is_active": False,
        "scope": {"model": {"id": None, "display_name": "Example Model"}, "surface": None},
    },
    {
        "kind": "monthly_experiment",
        "group": "monthly",
        "percent": 3,
        "severity": "normal",
        "resets_at": None,
        "scope": None,
        "is_active": False,
    },
]


def _write_config(tmp_path, monkeypatch, limits=None, cache=..., age_ms=0):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    if cache is ...:
        cache = {
            "fetchedAtMs": int(time.time() * 1000) - age_ms,
            "accountUuid": FAKE_ACCOUNT_UUID,
            "utilization": {"limits": LIMITS if limits is None else limits},
        }
    path = tmp_path / ".claude.json"
    path.write_text(json.dumps({"cachedUsageUtilization": cache} if cache else {}))
    return path


RATE_LIMIT_INFO = {
    "status": "allowed",
    "rateLimitType": "five_hour",
    "unifiedWindows": {
        "five_hour": {"utilization": 0.06, "resetsAt": 1767225600},
        "seven_day": {"utilization": 0.12, "resetsAt": 1767830400},
    },
}


@pytest.fixture(autouse=True)
def _no_live_result(monkeypatch):
    for name, value in [("_last_live", None), ("_failures", 0), ("_retry_at", 0.0), ("_turn", {})]:
        monkeypatch.setattr(usage, name, value)


@pytest.fixture
def clock(monkeypatch):
    now = [time.time()]
    monkeypatch.setattr(time, "time", lambda: now[0])
    return now


@pytest.fixture
def windows(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch)
    return {w.key: w for w in usage.fetch().windows}


def test_session_limit_is_labeled_session(windows):
    assert windows["session"].label == "session"
    assert windows["session"].used_pct == 0.0
    assert windows["session"].resets_at is None


def test_weekly_all_is_labeled_week(windows):
    assert windows["weekly_all"].label == "week"
    assert windows["weekly_all"].used_pct == 42.0
    assert windows["weekly_all"].resets_at == "2026-01-08T12:00:00+00:00"


def test_weekly_scoped_label_carries_the_model_display_name(windows):
    assert windows["weekly_scoped"].label == "week · Example Model"
    assert windows["weekly_scoped"].used_pct == 7.0


def test_a_limit_with_no_percent_is_skipped_not_fatal(tmp_path, monkeypatch):
    # One window the cache has not filled in must not discard the ones it has.
    _write_config(
        tmp_path,
        monkeypatch,
        limits=[
            {"kind": "session", "group": "session", "percent": None, "resets_at": None},
            {"kind": "weekly_all", "group": "weekly", "percent": 23, "resets_at": "2026-01-08T21:59:59+00:00"},
        ],
    )
    report = usage.fetch()
    assert report.error is None
    assert [w.key for w in report.windows] == ["weekly_all"]


def test_unknown_kind_still_produces_a_window_labeled_by_its_kind(windows):
    assert windows["monthly_experiment"].label == "monthly_experiment"
    assert windows["monthly_experiment"].used_pct == 3.0


def test_scoped_limit_without_a_display_name_falls_back_to_its_kind(tmp_path, monkeypatch):
    limits = [{"kind": "weekly_scoped", "percent": 5, "resets_at": None, "scope": {"model": None, "surface": None}}]
    _write_config(tmp_path, monkeypatch, limits=limits)

    assert usage.fetch().windows[0].label == "weekly_scoped"


def test_missing_config_reports_an_error_and_no_windows(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))

    report = usage.fetch()

    assert report.windows == ()
    assert report.error


def test_invalid_json_reports_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    (tmp_path / ".claude.json").write_text("{not json")

    report = usage.fetch()

    assert report.windows == ()
    assert report.error


def test_absent_usage_cache_reports_an_error(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch, cache=None)

    report = usage.fetch()

    assert report.windows == ()
    assert report.error


def test_account_uuid_never_reaches_the_collected_report(tmp_path, monkeypatch):
    config = _write_config(tmp_path, monkeypatch)
    assert FAKE_ACCOUNT_UUID in config.read_text()

    reset_quota_sources()
    register_quota_source(usage.PROVIDER, usage.LABEL, usage.fetch)
    payload = json.dumps(collect_quota_reports(), ensure_ascii=False)

    assert FAKE_ACCOUNT_UUID not in payload
    assert "accountUuid" not in payload
    assert "week · Example Model" in payload


def _write_credentials(tmp_path):
    (tmp_path / ".credentials.json").write_text(json.dumps({"claudeAiOauth": {"accessToken": "example-token"}}))


def _pcts(report):
    return {w.key: w.used_pct for w in report.windows}


def _serve_limits(monkeypatch):
    calls = []

    def fake_get(url, headers, timeout):
        calls.append((url, headers))
        return httpx.Response(200, json={"limits": LIMITS}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    return calls


def _unauthorized(url, headers, timeout):
    return httpx.Response(401, request=httpx.Request("GET", url))


def _offline(url, headers, timeout):
    raise httpx.ConnectError("offline")


def test_live_endpoint_is_read_with_the_oauth_token(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch, limits=[{"kind": "weekly_all", "percent": 99, "resets_at": None}])
    _write_credentials(tmp_path)
    calls = _serve_limits(monkeypatch)

    report = usage.fetch()

    assert report.error is None
    assert _pcts(report)["weekly_all"] == 42.0
    url, headers = calls[0]
    assert url == usage.USAGE_URL
    assert headers["Authorization"] == "Bearer example-token"
    assert headers["anthropic-beta"] == "oauth-2025-04-20"


def test_live_result_is_reused_within_the_ttl(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch)
    _write_credentials(tmp_path)
    calls = _serve_limits(monkeypatch)

    usage.fetch()
    usage.fetch()

    assert len(calls) == 1


def test_stale_cache_reports_an_error_instead_of_old_numbers(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch, age_ms=2 * 60 * 60 * 1000)

    report = usage.fetch()

    assert report.windows == ()
    assert report.error


@pytest.mark.parametrize("fake_get", [_unauthorized, _offline])
def test_fresh_cache_is_used_when_the_live_fetch_fails(tmp_path, monkeypatch, fake_get):
    _write_config(tmp_path, monkeypatch)
    _write_credentials(tmp_path)
    monkeypatch.setattr(httpx, "get", fake_get)

    report = usage.fetch()

    assert report.error is None
    assert _pcts(report)["weekly_all"] == 42.0


def test_turn_windows_are_reported_with_the_endpoint_keys_and_labels(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))

    usage.record_rate_limit(RATE_LIMIT_INFO)
    report = usage.fetch()

    assert report.error is None
    windows = {w.key: w for w in report.windows}
    assert windows["session"].label == "session"
    assert windows["session"].used_pct == 6.0
    assert windows["session"].resets_at == "2026-01-01T00:00:00+00:00"
    assert windows["weekly_all"].label == "week"
    assert windows["weekly_all"].used_pct == 12.0
    assert windows["weekly_all"].resets_at == "2026-01-08T00:00:00+00:00"


def test_rate_limit_event_without_windows_records_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))

    usage.record_rate_limit({"status": "allowed", "rateLimitType": "five_hour", "resetsAt": 1767225600})

    assert usage.fetch().error


def test_newer_turn_reading_overrides_endpoint_windows_and_keeps_scoped_ones(tmp_path, monkeypatch, clock):
    _write_config(tmp_path, monkeypatch, cache=None)
    _write_credentials(tmp_path)
    _serve_limits(monkeypatch)
    usage.fetch()

    clock[0] += 60
    usage.record_rate_limit(RATE_LIMIT_INFO)

    pcts = _pcts(usage.fetch())
    assert pcts["session"] == 6.0
    assert pcts["weekly_all"] == 12.0
    assert pcts["weekly_scoped"] == 7.0


def test_endpoint_reading_newer_than_the_turn_is_kept(tmp_path, monkeypatch, clock):
    _write_config(tmp_path, monkeypatch, cache=None)
    _write_credentials(tmp_path)
    _serve_limits(monkeypatch)
    usage.record_rate_limit(RATE_LIMIT_INFO)

    clock[0] += 60
    assert _pcts(usage.fetch())["weekly_all"] == 42.0


def test_endpoint_is_called_again_only_after_an_hour(tmp_path, monkeypatch, clock):
    _write_config(tmp_path, monkeypatch, cache=None)
    _write_credentials(tmp_path)
    calls = _serve_limits(monkeypatch)

    usage.fetch()
    clock[0] += 59 * 60
    usage.fetch()
    assert len(calls) == 1

    clock[0] += 2 * 60
    usage.fetch()
    assert len(calls) == 2


def test_failed_refresh_serves_the_last_good_reading_and_backs_off(tmp_path, monkeypatch, clock):
    _write_config(tmp_path, monkeypatch, cache=None)
    _write_credentials(tmp_path)
    _serve_limits(monkeypatch)
    usage.fetch()

    calls = []

    def rate_limited(url, headers, timeout):
        calls.append(url)
        return httpx.Response(429, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", rate_limited)
    clock[0] += 61 * 60
    report = usage.fetch()
    assert report.error is None
    assert _pcts(report)["weekly_all"] == 42.0

    for backoff_min in (5, 10, 20):
        attempts = len(calls)
        clock[0] += (backoff_min - 1) * 60
        usage.fetch()
        assert len(calls) == attempts
        clock[0] += 60
        usage.fetch()
        assert len(calls) == attempts + 1


def test_backoff_caps_at_an_hour(tmp_path, monkeypatch, clock):
    _write_config(tmp_path, monkeypatch, cache=None)
    _write_credentials(tmp_path)
    calls = []

    def not_json(url, headers, timeout):
        calls.append(url)
        return httpx.Response(200, text="<html>", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", not_json)
    for _ in range(6):
        usage.fetch()
        clock[0] += 60 * 60
    assert len(calls) == 6

    clock[0] -= 1
    usage.fetch()
    assert len(calls) == 6
