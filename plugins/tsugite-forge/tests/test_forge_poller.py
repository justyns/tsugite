"""The adapter's per-forge cache, poller and backoff."""

import asyncio
import logging
import threading

import pytest
from forge_helpers import TOKEN, FakeForgejo, make_adapter

INTERVAL = 300  # poll_minutes: 5


class FakeProvider:
    def __init__(self, prs=(), error=None):
        self.prs = list(prs)
        self.error = error
        self.calls = 0
        self.viewer_calls = 0
        self.viewer_error = None

    def list_repos(self):
        self.calls += 1
        if self.error:
            raise self.error
        return ["octo-org/demo"]

    def list_prs(self, repos):
        return self.prs

    def viewer(self):
        self.viewer_calls += 1
        if self.viewer_error:
            raise self.viewer_error
        return "octocat"


@pytest.fixture
def make(clock, secrets_backend):
    return lambda forges=("home", "gh"), **kw: make_adapter(clock, forges=forges, **kw)


def _status(adapter, name):
    return next(entry for entry in adapter.status() if entry["name"] == name)


@pytest.mark.asyncio
async def test_refresh_fills_the_cache_and_status(make, clock):
    adapter = make()
    adapter.forges["home"].provider = FakeProvider(prs=["pr-7"])
    adapter.forges["gh"].provider = FakeProvider(prs=["pr-12"])
    await adapter.refresh()
    assert adapter.forges["home"].prs == ["pr-7"]
    assert adapter.forges["home"].repos == ["octo-org/demo"]
    assert adapter.forges["home"].viewer == "octocat"
    clock.now += 42
    assert _status(adapter, "home") == {
        "name": "home",
        "kind": "forgejo",
        "ok": True,
        "last_refresh": "1970-01-01T00:16:40+00:00",
        "cache_age_seconds": 42,
        "error": None,
        "next_attempt": "1970-01-01T00:21:40+00:00",
        "viewer": "octocat",
    }


def test_status_before_the_first_refresh(make):
    entry = _status(make(), "gh")
    assert (entry["ok"], entry["last_refresh"], entry["cache_age_seconds"], entry["error"]) == (False, None, None, None)


@pytest.mark.asyncio
async def test_viewer_is_read_once(make):
    adapter = make(["gh"])
    provider = adapter.forges["gh"].provider = FakeProvider()
    await adapter.refresh()
    await adapter.refresh()
    assert (provider.calls, provider.viewer_calls) == (2, 1)


@pytest.mark.asyncio
async def test_a_failed_viewer_lookup_keeps_the_pull_requests_and_is_retried(make, clock):
    adapter = make(["home"])
    provider = adapter.forges["home"].provider = FakeProvider(prs=["pr-7"])
    provider.viewer_error = RuntimeError("403 needs read:user")
    await adapter.refresh()
    assert adapter.forges["home"].prs == ["pr-7"]
    status = _status(adapter, "home")
    assert (status["ok"], status["error"], status["viewer"]) == (
        False,
        "viewer lookup failed: RuntimeError: 403 needs read:user",
        None,
    )
    assert status["next_attempt"] == "1970-01-01T00:21:40+00:00"

    provider.viewer_error = None
    await adapter.refresh()
    assert (_status(adapter, "home")["ok"], adapter.forges["home"].viewer, provider.viewer_calls) == (
        True,
        "octocat",
        2,
    )


async def _poll(adapter, clock, monkeypatch, advance, sleeps=3):
    """Run the gh poller until it has slept `sleeps` times. Each sleep moves the clock by `advance(seconds)`."""
    import tsugite_forge.adapter as adapter_module

    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)
        if len(slept) == sleeps:
            raise asyncio.CancelledError
        clock.now += advance(seconds)

    monkeypatch.setattr(adapter_module.asyncio, "sleep", fake_sleep)
    with pytest.raises(asyncio.CancelledError):
        await adapter._poll(adapter.forges["gh"])
    return slept


@pytest.mark.asyncio
async def test_poll_waits_until_the_forge_is_due(make, clock, monkeypatch):
    adapter = make(["gh"])
    provider = adapter.forges["gh"].provider = FakeProvider()
    slept = await _poll(adapter, clock, monkeypatch, advance=lambda seconds: 100)
    assert provider.calls == 1
    assert slept == [INTERVAL, INTERVAL - 100, INTERVAL - 200]


@pytest.mark.asyncio
async def test_poll_refreshes_each_time_the_forge_is_due(make, clock, monkeypatch):
    adapter = make(["gh"])
    provider = adapter.forges["gh"].provider = FakeProvider()
    await _poll(adapter, clock, monkeypatch, advance=lambda seconds: seconds)
    assert provider.calls == 3


@pytest.mark.asyncio
async def test_failures_back_off_exponentially_up_to_an_hour(make, clock):
    from tsugite_forge.providers import ForgeError

    adapter = make(["gh"])
    provider = adapter.forges["gh"].provider = FakeProvider(error=ForgeError("gh: boom"))
    delays = []
    for _ in range(5):
        await adapter.refresh()
        delays.append(adapter.forges["gh"].next_attempt - clock.now)
        clock.now = adapter.forges["gh"].next_attempt
    assert delays == [600, 1200, 2400, 3600, 3600]
    assert provider.calls == 5
    assert adapter.forges["gh"].failures == 5
    assert _status(adapter, "gh")["error"] == "gh: boom"
    assert _status(adapter, "gh")["ok"] is False


@pytest.mark.asyncio
async def test_backoff_never_drops_below_the_poll_interval(make, clock):
    from tsugite_forge.providers import ForgeError

    adapter = make(["gh"], poll_minutes=90)
    adapter.forges["gh"].provider = FakeProvider(error=ForgeError("gh: boom"))
    for _ in range(3):
        await adapter.refresh()
        assert adapter.forges["gh"].next_attempt - clock.now == 5400
        clock.now = adapter.forges["gh"].next_attempt


@pytest.mark.asyncio
async def test_success_resets_the_backoff_and_keeps_cache_through_a_failure(make, clock):
    from tsugite_forge.providers import ForgeError

    adapter = make(["gh"])
    provider = adapter.forges["gh"].provider = FakeProvider(prs=["pr-12"])
    await adapter.refresh()
    provider.error = ForgeError("gh: down")
    await adapter.refresh()
    assert adapter.forges["gh"].prs == ["pr-12"]
    assert _status(adapter, "gh")["error"] == "gh: down"
    provider.error = None
    await adapter.refresh()
    cache = adapter.forges["gh"]
    assert (cache.failures, cache.error, cache.next_attempt - clock.now) == (0, None, INTERVAL)


@pytest.mark.asyncio
async def test_one_forge_down_does_not_stop_the_others(make):
    from tsugite_forge.providers import ForgeError

    adapter = make()
    adapter.forges["home"].provider = FakeProvider(prs=["pr-7"])
    adapter.forges["gh"].provider = FakeProvider(error=ForgeError("gh: run `gh auth login`"))
    await adapter.refresh()
    assert adapter.forges["home"].prs == ["pr-7"]
    assert _status(adapter, "home")["ok"] is True
    assert _status(adapter, "gh")["error"] == "gh: run `gh auth login`"


@pytest.mark.asyncio
async def test_an_unexpected_exception_is_recorded_not_raised(make):
    adapter = make(["gh"])
    adapter.forges["gh"].provider = FakeProvider(error=KeyError("draft"))
    await adapter.refresh()
    assert _status(adapter, "gh")["error"] == "KeyError: 'draft'"


@pytest.mark.asyncio
async def test_a_slow_forge_does_not_hold_back_the_others(make):
    adapter = make()
    release = threading.Event()

    class Slow(FakeProvider):
        def list_repos(self):
            release.wait(10)
            return []

    adapter.forges["home"].provider = Slow()
    adapter.forges["gh"].provider = FakeProvider(prs=["pr-12"])
    await adapter.start()
    try:
        for _ in range(200):
            if adapter.forges["gh"].last_refresh is not None:
                break
            await asyncio.sleep(0.01)
        assert adapter.forges["gh"].prs == ["pr-12"]
        assert adapter.forges["home"].last_refresh is None
    finally:
        release.set()
        await adapter.stop()


@pytest.mark.asyncio
async def test_status_and_logs_never_contain_the_token(make, caplog):
    from tsugite_forge.providers.forgejo import ForgejoProvider

    caplog.set_level(logging.DEBUG)
    echo = {"__status": 401, "message": f"token {TOKEN} is not valid"}
    adapter = make()
    server = FakeForgejo({"/api/v1/repos/octo-org/demo/pulls?state=open&limit=50&page=1": echo})
    adapter.forges["home"].provider = ForgejoProvider(adapter.forges["home"].config, transport=server.transport)
    gh = adapter.forges["gh"].provider = FakeProvider()
    await adapter.refresh()
    # A token can only leak into an error once it has been read, which registers it.
    gh.error = RuntimeError(f"leaked {TOKEN}")
    await adapter.refresh()
    assert "401" in _status(adapter, "home")["error"]
    assert _status(adapter, "gh")["error"] == "RuntimeError: leaked ***"
    assert TOKEN not in str(adapter.status())
    assert TOKEN not in caplog.text


@pytest.mark.asyncio
async def test_start_and_stop_set_and_clear_the_tools_handle(make):
    from tsugite_forge.tools import forge_repos

    adapter = make(["gh"])
    adapter.forges["gh"].provider = FakeProvider()
    await adapter.start()
    assert forge_repos().startswith("gh (github)")
    await asyncio.wait_for(adapter.stop(), 1)
    with pytest.raises(RuntimeError, match="daemon"):
        forge_repos()
