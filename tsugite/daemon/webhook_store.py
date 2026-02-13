"""Persistent webhook storage backed by a JSON file."""

import json
import logging
import os
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

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
    """Persistent webhook store with atomic saves."""

    def __init__(self, path: Path):
        self._path = path
        self._webhooks: dict[str, WebhookEntry] = {}
        self._load()

    def add(self, agent: str, source: str, token: str | None = None) -> WebhookEntry:
        if token is None:
            token = secrets.token_urlsafe(32)
        if token in self._webhooks:
            raise ValueError(f"Webhook token already exists: {token[:8]}...")
        entry = WebhookEntry(token=token, agent=agent, source=source)
        self._webhooks[token] = entry
        self._save()
        logger.info("Added webhook for agent '%s' source '%s'", agent, source)
        return entry

    def remove(self, token: str) -> None:
        if token not in self._webhooks:
            raise ValueError("Webhook not found")
        del self._webhooks[token]
        self._save()
        logger.info("Removed webhook %s...", token[:8])

    def get(self, token: str) -> WebhookEntry | None:
        return self._webhooks.get(token)

    def list(self) -> list[WebhookEntry]:
        return list(self._webhooks.values())

    def _load(self):
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            for entry_data in data.get("webhooks", []):
                entry = WebhookEntry(**entry_data)
                self._webhooks[entry.token] = entry
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.error("Failed to load webhooks from %s: %s", self._path, e)

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"webhooks": [asdict(e) for e in self._webhooks.values()]}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(str(tmp), str(self._path))
