"""Context detector: auto-attach fetched page text for URLs in a message.

Registered under the ``tsugite.context_providers`` entry point (``web``). When an
outgoing message contains http(s) URLs, this detector fetches each page's
readable article text server-side and attaches it as a ``<client_context>``
item, so the agent sees the page contents without a manual fetch tool call.

Each fetch is gated on a human approval unless the URL's host is on the
permissions allowlist; "Always allow" adds the host so it fetches silently
thereafter (see ``_may_fetch``).
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlsplit

from tsugite.approval import request_approval
from tsugite.attachments import get_specific_handler
from tsugite.attachments.base import Attachment
from tsugite.context import ContextProvider, register_context_provider
from tsugite.permissions import get_permissions
from tsugite.tools.http import fetch_text

logger = logging.getLogger(__name__)

# http(s) URLs. `\S+` grabs the whole run; trailing sentence punctuation and
# wrapping brackets/quotes are stripped afterwards so "(see https://x.com)." and
# a markdown "[x](https://x.com)" both yield a clean URL.
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_TRAILING = ".,;:!?)]}>\"'"

# One link-heavy message shouldn't fan out into a pile of blocking fetches, and a
# single page's worth of text is plenty of context per link.
_MAX_URLS = 3
_MAX_VALUE_CHARS = 4000


def _find_urls(message: str) -> list[str]:
    """The http(s) URLs in ``message``: de-duplicated, order preserved, capped."""
    seen: set[str] = set()
    urls: list[str] = []
    for match in _URL_RE.findall(message or ""):
        url = match.rstrip(_TRAILING)
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
            if len(urls) >= _MAX_URLS:
                break
    return urls


def _host_of(url: str) -> str | None:
    """The lowercased hostname of ``url``, or ``None`` when it has none."""
    host = urlsplit(url).hostname
    return host.lower() if host else None


def _may_fetch(url: str) -> bool:
    """Whether the page at ``url`` may be fetched.

    A host on the active permissions store's allowlist fetches silently; any
    other host - including when there is no store at all - is gated on a human
    approval, and answering "Always allow" persists the host to the allowlist.
    A URL with no hostname is never fetched.

    The detector runs on a worker thread, so the blocking approval prompt here
    pauses the send until the user answers.
    """
    host = _host_of(url)
    if not host:
        return False
    perms = get_permissions()
    if perms and perms.web_fetch_allowed(host):
        return True
    decision = request_approval(f"Fetch content from {host}?", allow_always=True)
    if decision == "always":
        if perms:
            perms.web_fetch_allow(host)
        return True
    return decision == "approve"


def _fetch_url_text(url: str) -> str:
    """The text to attach for ``url``.

    A specific attachment handler (a plugin like the YouTube transcript handler)
    is preferred when one claims the URL, so a pasted video link yields its
    transcript rather than the scraped watch page - the same result ``tsu -f`` has
    always given. Everything else, and a handler that fails or returns nothing,
    falls back to the generic readable-article scrape.
    """
    handler = get_specific_handler(url)
    if handler is not None:
        try:
            att = handler.fetch(url)
            if isinstance(att.content, str) and att.content.strip():
                return att.content
        except Exception as e:
            logger.info("Attachment handler %s failed for %s, scraping instead: %s", type(handler).__name__, url, e)
    return fetch_text(url, extract_article=True)


def detect_urls(message: str, context: dict) -> list[Attachment]:
    """Fetch each URL's readable text and attach it as one item per reachable URL.

    Each URL is gated on approval unless its host is allowlisted (see
    ``_may_fetch``); a denied or hostless URL attaches nothing. A URL that fails
    to fetch also attaches nothing: a dead or slow link must never break the send
    (detectors are run best-effort at send time).
    """
    items: list[Attachment] = []
    for url in _find_urls(message):
        if not _may_fetch(url):
            continue
        try:
            text = _fetch_url_text(url)
        except Exception as e:
            logger.info("Context detector 'webpage' skipping %s: %s", url, e)
            continue
        value = (text or "").strip()
        if not value:
            continue
        items.append(
            Attachment.context(key=f"webpage:{url}", label=url, value=value[:_MAX_VALUE_CHARS], untrusted=True)
        )
    return items


register_context_provider(ContextProvider(key="webpage", label="Web page", icon="link", detect=detect_urls))
