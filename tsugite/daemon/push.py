"""Web push notification support for the daemon."""

import asyncio
import json
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class PushSubscriptionStore:
    """JSON file backed store for web push subscriptions, keyed by endpoint URL."""

    def __init__(self, path: Path):
        self._path = path
        self._subs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                self._subs = {s["endpoint"]: s for s in json.loads(self._path.read_text())}
            except (json.JSONDecodeError, KeyError):
                self._subs = {}

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(list(self._subs.values()), indent=2))
        os.replace(str(tmp), str(self._path))

    def subscribe(self, subscription_info: dict) -> None:
        with self._lock:
            self._subs[subscription_info["endpoint"]] = subscription_info
            self._save()

    def unsubscribe(self, endpoint: str) -> None:
        with self._lock:
            self._subs.pop(endpoint, None)
            self._save()

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
                vapid_claims=vapid_claims,
            )
            return {"status": "sent"}
        except WebPushException as e:
            if hasattr(e, "response") and e.response is not None and e.response.status_code in (404, 410):
                return {"status": "expired", "endpoint": subscription_info.get("endpoint")}
            return {"error": str(e)}

    return await asyncio.to_thread(_send)
