"""Web push notification support for the daemon."""

import asyncio
import json
import logging
import threading
from pathlib import Path

from tsugite.core.record_store import SqliteCollectionStorage

logger = logging.getLogger(__name__)


class PushSubscriptionStore:
    """Web push subscriptions, write-through to daemon.db (keyed by endpoint URL).

    The legacy JSON path stays as the constructor argument purely as a one-time
    migration source (imported when the db collection is empty, then left
    untouched as a backup).
    """

    def __init__(self, path: Path):
        self._path = path
        self._storage = SqliteCollectionStorage.for_state_file(path, "push_subscriptions")
        self._subs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._load()

    def _read_legacy(self) -> list[dict]:
        # The legacy file was a bare JSON array (not `{key: [...]}`), so it
        # needs its own read rather than load_legacy_json_entries.
        if not self._path.exists():
            return []
        try:
            return [s for s in json.loads(self._path.read_text()) if isinstance(s, dict) and "endpoint" in s]
        except (json.JSONDecodeError, TypeError):
            return []

    def _load(self):
        entries, migrating = self._storage.load_or_migrate(
            self._path, "push_subscriptions", legacy_reader=self._read_legacy
        )
        self._subs = {s["endpoint"]: s for s in entries if isinstance(s, dict) and "endpoint" in s}
        if migrating:
            self._storage.replace_all(dict(self._subs))

    def subscribe(self, subscription_info: dict) -> None:
        with self._lock:
            self._subs[subscription_info["endpoint"]] = subscription_info
            self._storage.upsert(subscription_info["endpoint"], subscription_info)

    def unsubscribe(self, endpoint: str) -> None:
        with self._lock:
            self._subs.pop(endpoint, None)
            self._storage.delete(endpoint)

    def all(self) -> list[dict]:
        with self._lock:
            return list(self._subs.values())


def get_or_create_vapid_keys(state_dir: Path) -> tuple[str, str]:
    """Get or generate VAPID keys. Returns (private_key_pem_path, public_key_b64url)."""
    import base64

    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    from py_vapid import Vapid

    private_key_path = state_dir / "vapid_private.pem"
    public_key_path = state_dir / "vapid_public.pem"

    state_dir.mkdir(parents=True, exist_ok=True)

    if private_key_path.exists():
        vapid = Vapid.from_file(str(private_key_path))
    else:
        vapid = Vapid()
        vapid.generate_keys()
        vapid.save_key(str(private_key_path))
        vapid.save_public_key(str(public_key_path))
        logger.info("Generated new VAPID keys at %s", state_dir)

    raw = vapid.public_key.public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
    public_key_b64url = base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
    return str(private_key_path), public_key_b64url


async def send_web_push(
    subscription_info: dict,
    message: dict,
    vapid_private_key: str,
    vapid_claims: dict,
) -> dict:
    """Send a web push notification. Returns status dict. Auto-detects expired subs."""
    from pywebpush import WebPushException, webpush

    def _send():
        try:
            webpush(
                subscription_info=subscription_info,
                data=json.dumps(message),
                vapid_private_key=vapid_private_key,
                # pywebpush mutates the claims dict (pins `aud` to the first
                # endpoint's origin) - pass a copy so one subscriber's push
                # service can't poison the JWT audience for all the others.
                vapid_claims=dict(vapid_claims),
            )
            return {"status": "sent"}
        except WebPushException as e:
            if hasattr(e, "response") and e.response is not None and e.response.status_code in (404, 410):
                return {"status": "expired", "endpoint": subscription_info.get("endpoint")}
            return {"error": str(e)}

    return await asyncio.to_thread(_send)
