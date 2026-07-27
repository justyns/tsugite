"""Behavioral tests for the push HTTP endpoints (PushMixin).

The route table (tests/daemon/test_http_route_table.py) only pins that these
routes *exist*; nothing exercised what they return. This pins the contract:
the vapid-key 404 when unconfigured, auth gating on subscribe/unsubscribe, the
'not configured' 404, the missing-endpoint 400, and the store side effects on
success. The handlers read only self.vapid_public_key / self.push_store /
self._check_auth / request.json(), so a light harness stands in for HTTPServer
without the full app fixture.
"""

import json

import pytest
from starlette.responses import JSONResponse
from tsugite_daemon.adapters.http.push import PushMixin
from tsugite_daemon.push import PushSubscriptionStore

SAMPLE = {"endpoint": "https://push.example.com/sub/abc", "keys": {"p256dh": "x", "auth": "y"}}


class _FakeRequest:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


class _PushHarness(PushMixin):
    """Minimal stand-in exposing only what the mixin handlers read."""

    def __init__(self, *, push_store=None, vapid_public_key=None, authed=True):
        self.push_store = push_store
        self.vapid_public_key = vapid_public_key
        self._authed = authed

    def _check_auth(self, request):
        # None means "authorized"; a response means "rejected" (mirrors HTTPServer).
        return None if self._authed else JSONResponse({"error": "unauthorized"}, status_code=401)


def _payload(resp: JSONResponse) -> dict:
    return json.loads(resp.body)


@pytest.fixture
def store(tmp_path):
    return PushSubscriptionStore(tmp_path / "subs.json")


class TestVapidKey:
    @pytest.mark.asyncio
    async def test_404_when_web_push_unconfigured(self):
        resp = await _PushHarness(vapid_public_key=None)._push_vapid_key(_FakeRequest({}))
        assert resp.status_code == 404
        assert _payload(resp) == {"error": "web push not configured"}

    @pytest.mark.asyncio
    async def test_returns_public_key_when_configured(self):
        resp = await _PushHarness(vapid_public_key="BPUBKEY")._push_vapid_key(_FakeRequest({}))
        assert resp.status_code == 200
        assert _payload(resp) == {"public_key": "BPUBKEY"}


class TestSubscribe:
    @pytest.mark.asyncio
    async def test_auth_gate_precedes_config_check(self, store):
        # Unauthorized must 401 even though a store is present (auth is checked first).
        h = _PushHarness(push_store=store, authed=False)
        resp = await h._push_subscribe(_FakeRequest(SAMPLE))
        assert resp.status_code == 401
        assert store.all() == []  # not applied

    @pytest.mark.asyncio
    async def test_404_when_no_store(self):
        resp = await _PushHarness(push_store=None)._push_subscribe(_FakeRequest(SAMPLE))
        assert resp.status_code == 404
        assert _payload(resp) == {"error": "web push not configured"}

    @pytest.mark.asyncio
    async def test_400_on_missing_endpoint(self, store):
        h = _PushHarness(push_store=store)
        resp = await h._push_subscribe(_FakeRequest({"keys": {"p256dh": "x"}}))
        assert resp.status_code == 400
        assert _payload(resp) == {"error": "missing endpoint"}
        assert store.all() == []

    @pytest.mark.asyncio
    async def test_success_persists_subscription(self, store):
        h = _PushHarness(push_store=store)
        resp = await h._push_subscribe(_FakeRequest(SAMPLE))
        assert resp.status_code == 200
        assert _payload(resp) == {"status": "subscribed"}
        assert [s["endpoint"] for s in store.all()] == [SAMPLE["endpoint"]]


class TestUnsubscribe:
    @pytest.mark.asyncio
    async def test_auth_gate_precedes_config_check(self, store):
        store.subscribe(SAMPLE)
        h = _PushHarness(push_store=store, authed=False)
        resp = await h._push_unsubscribe(_FakeRequest({"endpoint": SAMPLE["endpoint"]}))
        assert resp.status_code == 401
        assert len(store.all()) == 1  # not removed

    @pytest.mark.asyncio
    async def test_404_when_no_store(self):
        resp = await _PushHarness(push_store=None)._push_unsubscribe(_FakeRequest({"endpoint": SAMPLE["endpoint"]}))
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_400_on_missing_endpoint(self, store):
        h = _PushHarness(push_store=store)
        resp = await h._push_unsubscribe(_FakeRequest({}))
        assert resp.status_code == 400
        assert _payload(resp) == {"error": "missing endpoint"}

    @pytest.mark.asyncio
    async def test_success_removes_subscription(self, store):
        store.subscribe(SAMPLE)
        h = _PushHarness(push_store=store)
        resp = await h._push_unsubscribe(_FakeRequest({"endpoint": SAMPLE["endpoint"]}))
        assert resp.status_code == 200
        assert _payload(resp) == {"status": "unsubscribed"}
        assert store.all() == []
