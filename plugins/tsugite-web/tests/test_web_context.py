"""Tests for the tsugite-web URL context detector."""

from unittest.mock import patch

import pytest
from tsugite_web import context as web_ctx

from tsugite.context import get_context_provider, reset_context_providers


class _FakePerms:
    """Stand-in for tsugite.permissions.Permissions: an in-memory allowlist that
    records every host persisted via ``web_fetch_allow``."""

    def __init__(self, allowed: set[str] | None = None):
        self.allowed = set(allowed or ())
        self.allow_calls: list[str] = []

    def web_fetch_allowed(self, host: str) -> bool:
        return host in self.allowed

    def web_fetch_allow(self, host: str) -> None:
        self.allow_calls.append(host)
        self.allowed.add(host)


@pytest.fixture(autouse=True)
def _default_approval():
    """Default the approval gate open so the fetch-behavior tests below stay
    focused on fetching; the approval-flow tests override this patch."""
    with patch.object(web_ctx, "request_approval", return_value="approve"):
        yield


def test_detect_attaches_item_per_reachable_url():
    msg = "compare https://a.example and https://b.example please"
    with patch.object(web_ctx, "fetch_text", side_effect=lambda url, **kw: f"body of {url}"):
        items = web_ctx.detect_urls(msg, {})

    assert [i.key for i in items] == ["webpage:https://a.example", "webpage:https://b.example"]
    assert items[0].label == "https://a.example"
    assert items[0].value == "body of https://a.example"


def test_detect_requests_article_extraction():
    with patch.object(web_ctx, "fetch_text", return_value="x") as m:
        web_ctx.detect_urls("https://a.example", {})
    m.assert_called_once_with("https://a.example", extract_article=True)


def test_detect_uses_specific_handler_when_one_claims_the_url():
    """A URL a specific attachment handler claims (a YouTube transcript, etc.) is
    fetched through that handler, not the generic article scraper - matching what
    ``tsu -f <url>`` has always done on the CLI."""

    class _FakeHandler:
        def fetch(self, source):
            from tsugite.attachments.base import Attachment, AttachmentContentType

            return Attachment(
                name="youtube:x",
                content="TRANSCRIPT",
                content_type=AttachmentContentType.TEXT,
                mime_type="text/plain",
                untrusted=True,
            )

    with (
        patch.object(web_ctx, "get_specific_handler", return_value=_FakeHandler()),
        patch.object(web_ctx, "fetch_text") as scraper,
    ):
        items = web_ctx.detect_urls("https://youtu.be/x", {})

    assert items[0].value == "TRANSCRIPT"
    assert items[0].untrusted is True
    scraper.assert_not_called()


def test_detect_scrapes_when_no_specific_handler():
    with (
        patch.object(web_ctx, "get_specific_handler", return_value=None),
        patch.object(web_ctx, "fetch_text", side_effect=lambda url, **kw: "ARTICLE"),
    ):
        items = web_ctx.detect_urls("https://plain.example", {})

    assert items[0].value == "ARTICLE"


def test_detect_scrapes_when_specific_handler_fails():
    """A handler that raises (e.g. a video with no transcript) must not swallow the
    URL: fall back to the generic scrape instead of attaching nothing."""

    class _Boom:
        def fetch(self, source):
            raise ValueError("no transcript")

    with (
        patch.object(web_ctx, "get_specific_handler", return_value=_Boom()),
        patch.object(web_ctx, "fetch_text", side_effect=lambda url, **kw: "ARTICLE"),
    ):
        items = web_ctx.detect_urls("https://youtu.be/x", {})

    assert items[0].value == "ARTICLE"


def test_detect_skips_failed_fetch():
    def fake(url, **kw):
        if "bad" in url:
            raise RuntimeError("boom")
        return f"ok {url}"

    with patch.object(web_ctx, "fetch_text", side_effect=fake):
        items = web_ctx.detect_urls("https://bad.example then https://good.example", {})

    assert [i.key for i in items] == ["webpage:https://good.example"]


def test_detect_no_urls_returns_empty():
    with patch.object(web_ctx, "fetch_text") as m:
        assert web_ctx.detect_urls("no links here at all", {}) == []
    m.assert_not_called()


def test_detect_caps_at_three_urls():
    urls = " ".join(f"https://e{i}.example" for i in range(6))
    calls: list[str] = []

    def fake(url, **kw):
        calls.append(url)
        return "x"

    with patch.object(web_ctx, "fetch_text", side_effect=fake):
        items = web_ctx.detect_urls(urls, {})

    assert len(items) == 3
    assert calls == ["https://e0.example", "https://e1.example", "https://e2.example"]


def test_detect_dedupes_keeping_order():
    with patch.object(web_ctx, "fetch_text", side_effect=lambda url, **kw: "x") as m:
        items = web_ctx.detect_urls("https://a.example https://a.example", {})

    assert len(items) == 1
    assert m.call_count == 1


def test_detect_truncates_long_value():
    with patch.object(web_ctx, "fetch_text", side_effect=lambda url, **kw: "y" * 5000):
        items = web_ctx.detect_urls("https://a.example", {})

    assert len(items[0].value) == 4000


def test_detect_strips_trailing_punctuation():
    with patch.object(web_ctx, "fetch_text", side_effect=lambda url, **kw: "x") as m:
        web_ctx.detect_urls("see (https://a.example/path), thanks.", {})
    m.assert_called_once_with("https://a.example/path", extract_article=True)


def test_registers_webpage_detector():
    import importlib

    reset_context_providers()
    importlib.reload(web_ctx)

    provider = get_context_provider("webpage")
    assert provider is not None
    assert provider.detect is not None
    assert not provider.in_menu
    assert (provider.label, provider.icon) == ("Web page", "link")


def test_host_of_lowercases_and_none_when_absent():
    assert web_ctx._host_of("https://A.EXAMPLE/x") == "a.example"
    assert web_ctx._host_of("https:///no-host") is None


def test_allowlisted_host_fetches_without_prompting():
    perms = _FakePerms(allowed={"a.example"})
    with (
        patch.object(web_ctx, "get_permissions", return_value=perms),
        patch.object(web_ctx, "request_approval") as ask,
        patch.object(web_ctx, "fetch_text", side_effect=lambda url, **kw: f"body {url}"),
    ):
        items = web_ctx.detect_urls("https://a.example/x", {})

    assert [i.key for i in items] == ["webpage:https://a.example/x"]
    ask.assert_not_called()


def test_allowlist_matches_uppercase_url_host():
    perms = _FakePerms(allowed={"a.example"})
    with (
        patch.object(web_ctx, "get_permissions", return_value=perms),
        patch.object(web_ctx, "request_approval") as ask,
        patch.object(web_ctx, "fetch_text", side_effect=lambda url, **kw: "body"),
    ):
        items = web_ctx.detect_urls("https://A.EXAMPLE/x", {})

    assert len(items) == 1
    ask.assert_not_called()


def test_unknown_host_prompts_and_fetches_on_approve():
    perms = _FakePerms()
    with (
        patch.object(web_ctx, "get_permissions", return_value=perms),
        patch.object(web_ctx, "request_approval", return_value="approve") as ask,
        patch.object(web_ctx, "fetch_text", side_effect=lambda url, **kw: "body"),
    ):
        items = web_ctx.detect_urls("https://a.example/x", {})

    assert [i.key for i in items] == ["webpage:https://a.example/x"]
    ask.assert_called_once()
    prompt, kwargs = ask.call_args
    assert "a.example" in prompt[0]
    assert kwargs.get("allow_always") is True
    assert perms.allow_calls == []


def test_always_persists_host_then_fetches():
    perms = _FakePerms()
    with (
        patch.object(web_ctx, "get_permissions", return_value=perms),
        patch.object(web_ctx, "request_approval", return_value="always"),
        patch.object(web_ctx, "fetch_text", side_effect=lambda url, **kw: "body"),
    ):
        items = web_ctx.detect_urls("https://a.example/x", {})

    assert len(items) == 1
    assert perms.allow_calls == ["a.example"]


def test_deny_attaches_nothing_and_skips_fetch():
    perms = _FakePerms()
    with (
        patch.object(web_ctx, "get_permissions", return_value=perms),
        patch.object(web_ctx, "request_approval", return_value="deny"),
        patch.object(web_ctx, "fetch_text") as fetch,
    ):
        items = web_ctx.detect_urls("https://a.example/x", {})

    assert items == []
    fetch.assert_not_called()


def test_no_perms_still_gates_and_denies():
    """A bare context (no store) must never silently fetch an un-vetted host: it
    still routes through approval, and a deny attaches nothing."""
    with (
        patch.object(web_ctx, "get_permissions", return_value=None),
        patch.object(web_ctx, "request_approval", return_value="deny") as ask,
        patch.object(web_ctx, "fetch_text") as fetch,
    ):
        items = web_ctx.detect_urls("https://a.example/x", {})

    assert items == []
    ask.assert_called_once()
    fetch.assert_not_called()


def test_no_perms_fetches_on_approve():
    with (
        patch.object(web_ctx, "get_permissions", return_value=None),
        patch.object(web_ctx, "request_approval", return_value="approve"),
        patch.object(web_ctx, "fetch_text", side_effect=lambda url, **kw: "body"),
    ):
        items = web_ctx.detect_urls("https://a.example/x", {})

    assert len(items) == 1


def test_always_without_perms_fetches_without_crashing():
    """ "Always allow" with no store can't persist, but must still fetch."""
    with (
        patch.object(web_ctx, "get_permissions", return_value=None),
        patch.object(web_ctx, "request_approval", return_value="always"),
        patch.object(web_ctx, "fetch_text", side_effect=lambda url, **kw: "body"),
    ):
        items = web_ctx.detect_urls("https://a.example/x", {})

    assert len(items) == 1


def test_hostless_url_is_skipped_without_prompting():
    with (
        patch.object(web_ctx, "get_permissions", return_value=None),
        patch.object(web_ctx, "request_approval") as ask,
        patch.object(web_ctx, "fetch_text") as fetch,
    ):
        items = web_ctx.detect_urls("https:///no-host", {})

    assert items == []
    ask.assert_not_called()
    fetch.assert_not_called()
