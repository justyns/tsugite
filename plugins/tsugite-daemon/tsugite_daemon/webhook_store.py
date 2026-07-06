"""Persistent webhook storage (daemon.db `webhooks` collection)."""

import logging
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from tsugite.core.record_store import SqliteCollectionStorage

logger = logging.getLogger(__name__)


@dataclass
class WebhookEntry:
    token: str
    agent: str
    source: str
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class WebhookStore:
    """Persistent webhook store, write-through to daemon.db.

    The legacy ``webhooks.json`` path is the constructor argument purely as a
    one-time migration source (imported when the db collection is empty, then
    left untouched as a backup).
    """

    def __init__(self, path: Path):
        self._path = path
        self._storage = SqliteCollectionStorage.for_state_file(path, "webhooks")
        self._webhooks: dict[str, WebhookEntry] = {}
        self._load()

    def add(self, agent: str, source: str, token: str | None = None) -> WebhookEntry:
        if token is None:
            token = secrets.token_urlsafe(32)
        if token in self._webhooks:
            raise ValueError(f"Webhook token already exists: {token[:8]}...")
        entry = WebhookEntry(token=token, agent=agent, source=source)
        self._webhooks[token] = entry
        self._storage.upsert(token, asdict(entry))
        logger.info("Added webhook for agent '%s' source '%s'", agent, source)
        return entry

    def remove(self, token: str) -> None:
        if token not in self._webhooks:
            raise ValueError("Webhook not found")
        del self._webhooks[token]
        self._storage.delete(token)
        logger.info("Removed webhook %s...", token[:8])

    def get(self, token: str) -> WebhookEntry | None:
        return self._webhooks.get(token)

    def list(self) -> list[WebhookEntry]:
        return list(self._webhooks.values())

    def _load(self):
        entries, migrating = self._storage.load_or_migrate(self._path, "webhooks")
        for entry_data in entries:
            try:
                entry = WebhookEntry(**entry_data)
            except TypeError as e:
                logger.error("Skipping malformed webhook entry: %s", e)
                continue
            self._webhooks[entry.token] = entry
        if migrating:
            self._storage.replace_all({t: asdict(e) for t, e in self._webhooks.items()})
