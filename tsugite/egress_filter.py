"""Egress filter: blocks or redacts configured strings in agent-reachable HTTP requests."""

import logging
import re
from functools import cache
from urllib.parse import quote, quote_plus

import httpx

from tsugite.config import EgressFilterConfig, load_config

logger = logging.getLogger(__name__)

REDACTED = "[REDACTED]"


class EgressBlockedError(RuntimeError):
    """Raised before sending when a request contains a configured string."""


@cache
def get_egress_filter() -> EgressFilterConfig:
    return load_config().egress_filter or EgressFilterConfig()


def apply_egress_filter(request: httpx.Request) -> None:
    """httpx request hook. Scans the URL, header values and body; raises or redacts in place.

    Errors and log records name the matching rule (`strings[i]`), never the matched text.
    """
    config = get_egress_filter()
    if config.mode == "off" or not config.strings:
        return

    patterns = [
        re.compile("|".join(re.escape(v) for v in {s, quote(s, safe=""), quote_plus(s)}), re.IGNORECASE)
        for s in config.strings
    ]
    hits: set[int] = set()

    def scrub(text: str) -> str:
        for i, pattern in enumerate(patterns):
            text, count = pattern.subn(REDACTED, text)
            if count:
                hits.add(i)
        return text

    url = scrub(str(request.url))
    headers = [(k, scrub(v)) for k, v in request.headers.multi_items()]
    body_text = request.content.decode(errors="replace")
    body = scrub(body_text)
    if not hits:
        return

    rules = ", ".join(f"strings[{i}]" for i in sorted(hits))
    if config.mode == "reject":
        raise EgressBlockedError(f"Request blocked by egress filter: matched {rules}")

    logger.warning("Egress filter redacted a request: matched %s", rules)
    request.url = httpx.URL(url)
    request.headers = httpx.Headers(headers)
    if body != body_text:
        data = body.encode()
        request.stream = httpx.ByteStream(data)
        request._content = data  # httpx.Request.read() returns this, not the stream
        request.headers["Content-Length"] = str(len(data))
