"""Tests for the provider quota-source registry."""

from tsugite.usage import (
    QuotaReport,
    QuotaWindow,
    collect_quota_reports,
    register_quota_source,
)


def _rows_by_provider() -> dict[str, dict]:
    return {row["provider"]: row for row in collect_quota_reports()}


def test_registered_source_round_trips_as_a_dict():
    register_quota_source(
        "demo",
        "Demo",
        lambda: QuotaReport(
            provider="demo",
            label="Demo",
            windows=(
                QuotaWindow(key="session", label="session", used_pct=0.0),
                QuotaWindow(key="weekly_all", label="week", used_pct=73.0, resets_at="2026-01-08T00:00:00+00:00"),
            ),
        ),
    )

    assert _rows_by_provider()["demo"] == {
        "provider": "demo",
        "label": "Demo",
        "error": None,
        "windows": [
            {"key": "session", "label": "session", "used_pct": 0.0, "resets_at": None},
            {
                "key": "weekly_all",
                "label": "week",
                "used_pct": 73.0,
                "resets_at": "2026-01-08T00:00:00+00:00",
            },
        ],
    }


def test_a_raising_source_becomes_an_error_row_without_losing_its_sibling():
    def boom() -> QuotaReport:
        raise RuntimeError("local state unreadable")

    register_quota_source("broken", "Broken", boom)
    register_quota_source(
        "healthy",
        "Healthy",
        lambda: QuotaReport(provider="healthy", label="Healthy", windows=(QuotaWindow("week", "week", 12.0),)),
    )

    rows = _rows_by_provider()

    assert rows["broken"]["windows"] == []
    assert "local state unreadable" in rows["broken"]["error"]
    assert rows["healthy"]["error"] is None
    assert rows["healthy"]["windows"] == [{"key": "week", "label": "week", "used_pct": 12.0, "resets_at": None}]


def test_reports_are_sorted_by_provider():
    register_quota_source("zebra", "Zebra", lambda: QuotaReport(provider="zebra", label="Zebra"))
    register_quota_source("alpha", "Alpha", lambda: QuotaReport(provider="alpha", label="Alpha"))

    providers = [row["provider"] for row in collect_quota_reports()]

    assert {"alpha", "zebra"} <= set(providers)
    assert providers == sorted(providers)


def test_re_registering_a_provider_replaces_the_previous_source():
    register_quota_source("demo", "Old", lambda: QuotaReport(provider="demo", label="Old"))
    register_quota_source(
        "demo",
        "New",
        lambda: QuotaReport(provider="demo", label="New", windows=(QuotaWindow("week", "week", 5.0),)),
    )

    rows = [row for row in collect_quota_reports() if row["provider"] == "demo"]

    assert len(rows) == 1
    assert rows[0]["label"] == "New"
    assert rows[0]["windows"] == [{"key": "week", "label": "week", "used_pct": 5.0, "resets_at": None}]


def test_a_source_registered_by_one_test_does_not_leak_into_the_next():
    assert "demo" not in _rows_by_provider()


def test_a_concurrent_first_read_waits_for_the_sources_to_load(monkeypatch):
    """The first collect imports the plugin sources; a second collect racing it
    must see them rather than an empty registry."""
    import threading

    from tsugite.usage import quota

    started, release = threading.Event(), threading.Event()

    def slow_load(_group):
        started.set()
        release.wait(5)
        register_quota_source("demo", "Demo", lambda: QuotaReport(provider="demo", label="Demo"))

    monkeypatch.setattr(quota, "_loaded", False)
    monkeypatch.setattr("tsugite.plugins.load_module_only_plugins", slow_load)
    first = threading.Thread(target=collect_quota_reports)
    first.start()
    assert started.wait(5)
    seen = []
    second = threading.Thread(target=lambda: seen.append(collect_quota_reports()))
    second.start()
    release.set()
    first.join(5)
    second.join(5)
    assert [r["provider"] for r in seen[0]] == ["demo"]
