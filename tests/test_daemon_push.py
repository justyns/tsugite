"""Tests for web push notification support."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tsugite.daemon.config import NotificationChannelConfig
from tsugite.daemon.push import PushSubscriptionStore


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "push_subscriptions.json"


@pytest.fixture
def store(store_path):
    return PushSubscriptionStore(store_path)


@pytest.fixture
def sample_sub():
    return {
        "endpoint": "https://push.example.com/sub/abc123",
        "keys": {"p256dh": "BNcRd...", "auth": "tBHI..."},
    }


class TestPushSubscriptionStore:
    def test_empty_store(self, store):
        assert store.all() == []

    def test_subscribe(self, store, sample_sub):
        store.subscribe(sample_sub)
        subs = store.all()
        assert len(subs) == 1
        assert subs[0]["endpoint"] == sample_sub["endpoint"]

    def test_unsubscribe(self, store, sample_sub):
        store.subscribe(sample_sub)
        store.unsubscribe(sample_sub["endpoint"])
        assert store.all() == []

    def test_unsubscribe_nonexistent(self, store):
        store.unsubscribe("https://nonexistent.example.com")
        assert store.all() == []

    def test_subscribe_replaces_same_endpoint(self, store, sample_sub):
        store.subscribe(sample_sub)
        updated = {**sample_sub, "keys": {"p256dh": "new_key", "auth": "new_auth"}}
        store.subscribe(updated)
        subs = store.all()
        assert len(subs) == 1
        assert subs[0]["keys"]["p256dh"] == "new_key"

    def test_multiple_subscriptions(self, store, sample_sub):
        sub2 = {"endpoint": "https://push.example.com/sub/def456", "keys": {"p256dh": "x", "auth": "y"}}
        store.subscribe(sample_sub)
        store.subscribe(sub2)
        assert len(store.all()) == 2

    def test_persistence(self, store_path, sample_sub):
        store1 = PushSubscriptionStore(store_path)
        store1.subscribe(sample_sub)

        store2 = PushSubscriptionStore(store_path)
        assert len(store2.all()) == 1
        assert store2.all()[0]["endpoint"] == sample_sub["endpoint"]

    def test_atomic_save(self, store_path, sample_sub):
        store = PushSubscriptionStore(store_path)
        store.subscribe(sample_sub)
        assert store_path.exists()
        assert not store_path.with_suffix(".tmp").exists()

    def test_corrupt_file_recovery(self, store_path):
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.write_text("not valid json{{{")
        store = PushSubscriptionStore(store_path)
        assert store.all() == []

    def test_creates_parent_dirs(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "subs.json"
        store = PushSubscriptionStore(deep_path)
        store.subscribe({"endpoint": "https://example.com", "keys": {}})
        assert deep_path.exists()


class TestGetOrCreateVapidKeys:
    def test_generates_keys(self, tmp_path):
        from tsugite.daemon.push import get_or_create_vapid_keys

        private_key_path, public_key_b64url = get_or_create_vapid_keys(tmp_path)
        assert Path(private_key_path).exists()
        assert len(public_key_b64url) > 0

    def test_reuses_existing_keys(self, tmp_path):
        from tsugite.daemon.push import get_or_create_vapid_keys

        path1, pub1 = get_or_create_vapid_keys(tmp_path)
        path2, pub2 = get_or_create_vapid_keys(tmp_path)
        assert path1 == path2
        assert pub1 == pub2


class TestSendWebPush:
    @pytest.mark.asyncio
    async def test_send_success(self):
        from tsugite.daemon.push import send_web_push

        sub = {"endpoint": "https://push.example.com/sub/abc", "keys": {"p256dh": "x", "auth": "y"}}
        with patch("pywebpush.webpush") as mock_wp:
            result = await send_web_push(sub, {"title": "Test"}, "/fake/key.pem", {"sub": "mailto:test@test"})
            assert result == {"status": "sent"}
            mock_wp.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_expired_410(self):
        from tsugite.daemon.push import send_web_push

        sub = {"endpoint": "https://push.example.com/sub/expired", "keys": {"p256dh": "x", "auth": "y"}}
        mock_response = MagicMock()
        mock_response.status_code = 410

        from pywebpush import WebPushException

        exc = WebPushException("Gone", response=mock_response)

        with patch("pywebpush.webpush", side_effect=exc):
            result = await send_web_push(sub, {"title": "Test"}, "/fake/key.pem", {"sub": "mailto:test@test"})
            assert result["status"] == "expired"
            assert result["endpoint"] == sub["endpoint"]

    @pytest.mark.asyncio
    async def test_send_expired_404(self):
        from tsugite.daemon.push import send_web_push

        sub = {"endpoint": "https://push.example.com/sub/gone", "keys": {"p256dh": "x", "auth": "y"}}
        mock_response = MagicMock()
        mock_response.status_code = 404

        from pywebpush import WebPushException

        exc = WebPushException("Not Found", response=mock_response)

        with patch("pywebpush.webpush", side_effect=exc):
            result = await send_web_push(sub, {"title": "Test"}, "/fake/key.pem", {"sub": "mailto:test@test"})
            assert result["status"] == "expired"

    @pytest.mark.asyncio
    async def test_send_other_error(self):
        from tsugite.daemon.push import send_web_push

        sub = {"endpoint": "https://push.example.com/sub/err", "keys": {"p256dh": "x", "auth": "y"}}
        mock_response = MagicMock()
        mock_response.status_code = 500

        from pywebpush import WebPushException

        exc = WebPushException("Server Error", response=mock_response)

        with patch("pywebpush.webpush", side_effect=exc):
            result = await send_web_push(sub, {"title": "Test"}, "/fake/key.pem", {"sub": "mailto:test@test"})
            assert "error" in result


class TestNotificationChannelConfig:
    def test_web_push_type_valid(self):
        config = NotificationChannelConfig(type="web-push")
        assert config.type == "web-push"

    def test_web_push_no_required_fields(self):
        config = NotificationChannelConfig(type="web-push")
        assert config.user_id is None
        assert config.bot is None
        assert config.url is None

    def test_discord_still_requires_fields(self):
        with pytest.raises(ValueError, match="user_id"):
            NotificationChannelConfig(type="discord")

    def test_webhook_still_requires_url(self):
        with pytest.raises(ValueError, match="url"):
            NotificationChannelConfig(type="webhook")


class TestSendWebPushAll:
    @pytest.mark.asyncio
    async def test_no_store(self):
        from tsugite.daemon.gateway import _send_web_push_all

        result = await _send_web_push_all(None, "test", "/key.pem", {"sub": "mailto:t@t"})
        assert result == {"error": "push store not initialized"}

    @pytest.mark.asyncio
    async def test_no_subscribers(self, store):
        from tsugite.daemon.gateway import _send_web_push_all

        result = await _send_web_push_all(store, "test", "/key.pem", {"sub": "mailto:t@t"})
        assert result == {"status": "no_subscribers"}

    @pytest.mark.asyncio
    async def test_sends_to_all(self, store, sample_sub):
        from tsugite.daemon.gateway import _send_web_push_all

        sub2 = {"endpoint": "https://push.example.com/sub/def456", "keys": {"p256dh": "x", "auth": "y"}}
        store.subscribe(sample_sub)
        store.subscribe(sub2)

        with patch("tsugite.daemon.push.send_web_push", new_callable=AsyncMock, return_value={"status": "sent"}):
            result = await _send_web_push_all(store, "hello", "/key.pem", {"sub": "mailto:t@t"})
            assert result["sent"] == 2
            assert result["expired"] == 0

    @pytest.mark.asyncio
    async def test_prunes_expired(self, store, sample_sub):
        from tsugite.daemon.gateway import _send_web_push_all

        store.subscribe(sample_sub)

        with patch(
            "tsugite.daemon.push.send_web_push",
            new_callable=AsyncMock,
            return_value={"status": "expired", "endpoint": sample_sub["endpoint"]},
        ):
            result = await _send_web_push_all(store, "hello", "/key.pem", {"sub": "mailto:t@t"})
            assert result["expired"] == 1
            assert store.all() == []

    @pytest.mark.asyncio
    async def test_truncates_long_message(self, store, sample_sub):
        from tsugite.daemon.gateway import _send_web_push_all

        store.subscribe(sample_sub)
        long_msg = "x" * 500

        with patch(
            "tsugite.daemon.push.send_web_push", new_callable=AsyncMock, return_value={"status": "sent"}
        ) as mock_send:
            await _send_web_push_all(store, long_msg, "/key.pem", {"sub": "mailto:t@t"})
            payload = mock_send.call_args[0][1]
            assert len(payload["body"]) == 200


class TestBuildNotifier:
    @pytest.mark.asyncio
    async def test_web_push_channel(self, store, sample_sub):
        from tsugite.daemon.gateway import _build_notifier

        store.subscribe(sample_sub)
        notifier = _build_notifier(
            {}, push_store=store, vapid_private_key="/key.pem", vapid_claims={"sub": "mailto:t@t"}
        )

        config = NotificationChannelConfig(type="web-push")
        with patch("tsugite.daemon.push.send_web_push", new_callable=AsyncMock, return_value={"status": "sent"}):
            results = await notifier("test message", [("push", config)])
            assert "push" in results
            assert results["push"]["sent"] == 1
