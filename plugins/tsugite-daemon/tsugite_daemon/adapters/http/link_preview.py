"""LinkPreviewMixin: GET /api/link-preview, answered by the plugin link resolvers.

A resolver timeout gets a 504, so the web UI can ask again.
"""

import asyncio
import inspect
import re
from urllib.parse import urlsplit

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from tsugite.link_previews import resolver_for
from tsugite.secrets.registry import get_registry
from tsugite_daemon.adapters.http.helpers import logger

RESOLVE_TIMEOUT = 2.0
TONES = ("ok", "warn", "err", "info", "mute")
MAX_ITEMS = 8
LABEL_MAX = 80
SHORT_MAX = 40
TEXT_MAX = 200


def _is_http(url) -> bool:
    return isinstance(url, str) and urlsplit(url).scheme in ("http", "https")


def _text(value, limit: int) -> str:
    # Masked before clipping, which could cut a secret short of what the mask recognizes.
    text = get_registry().mask(" ".join(str(value).split()))
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _badge(raw: dict) -> dict:
    return {"text": _text(raw["text"], SHORT_MAX), "tone": raw.get("tone") if raw.get("tone") in TONES else "mute"}


def _icon(icon, plugin: str) -> str | None:
    # Segments cannot start with a dot, which rules out `.` and `..`.
    pattern = rf"/api/plugins/{re.escape(plugin)}/ui(/[A-Za-z0-9_-][A-Za-z0-9_.-]*)+"
    return icon if isinstance(icon, str) and re.fullmatch(pattern, icon) else None


def normalize_preview(raw: dict, url: str, plugin: str) -> dict:
    """The fields the web UI renders, as bounded masked strings. Raises on a preview with no label."""
    label = _text(raw.get("label") or "", LABEL_MAX)
    if not label:
        raise ValueError("preview has no label")
    own_url = get_registry().mask(raw["url"]) if _is_http(raw.get("url")) else None
    return {
        "label": label,
        "title": _text(raw["title"], TEXT_MAX) if raw.get("title") else label,
        "url": own_url or url,
        "status": _badge(raw["status"]) if raw.get("status") else None,
        "badges": [_badge(b) for b in (raw.get("badges") or [])[:MAX_ITEMS]],
        "fields": [
            {"label": _text(f["label"], SHORT_MAX), "value": _text(f["value"], TEXT_MAX)}
            for f in (raw.get("fields") or [])[:MAX_ITEMS]
        ],
        "icon": _icon(raw.get("icon"), plugin),
    }


class LinkPreviewMixin:
    def _link_preview_routes(self) -> list:
        return [Route("/api/link-preview", self._api_link_preview, methods=["GET"])]

    async def _api_link_preview(self, request: Request) -> JSONResponse:
        if err := self._check_auth(request):
            return err
        url = request.query_params.get("url")
        if not _is_http(url):
            return JSONResponse({"error": "url must be an http(s) URL"}, status_code=400)
        try:
            preview = await self._resolve_link(url)
        except TimeoutError:
            return JSONResponse({"error": "preview timed out"}, status_code=504)
        if preview is None:
            return JSONResponse({"error": "no preview"}, status_code=404)
        return JSONResponse(preview)

    async def _resolve_link(self, url: str) -> dict | None:
        resolver = resolver_for(url)
        if resolver is None:
            return None
        name = f"{resolver['plugin']}:{resolver['name']}"
        resolve = resolver["resolve"]
        try:
            pending = resolve(url) if inspect.iscoroutinefunction(resolve) else asyncio.to_thread(resolve, url)
            raw = await asyncio.wait_for(pending, RESOLVE_TIMEOUT)
            if raw is None:
                logger.debug("link resolver %s has no preview for %s", name, url)
                return None
            return normalize_preview(raw, url, resolver["plugin"])
        except TimeoutError:
            logger.info("link resolver %s timed out for %s", name, url)
            raise
        except Exception:
            logger.warning("link resolver %s failed for %s", name, url, exc_info=True)
            return None
