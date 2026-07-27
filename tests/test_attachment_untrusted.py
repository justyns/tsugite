"""Untrusted-attachment metadata: the `untrusted` flag, its `<attachment>` tag
rendering, the remote handlers that default it, and `get_specific_handler`
routing (which lets the web detector prefer a smart handler over a raw scrape)."""

from unittest.mock import patch

from tsugite.attachments import get_specific_handler
from tsugite.attachments.base import (
    Attachment,
    AttachmentContentType,
    AttachmentHandler,
    format_attachment_open_tag,
)
from tsugite.attachments.url import GenericURLHandler


def _att(**kw) -> Attachment:
    base = dict(name="x", content="c", content_type=AttachmentContentType.TEXT, mime_type="text/plain")
    base.update(kw)
    return Attachment(**base)


def test_attachment_defaults_trusted():
    assert _att().untrusted is False


def test_open_tag_marks_untrusted():
    assert format_attachment_open_tag(_att(name="p", untrusted=True)) == '<attachment name="p" untrusted="true">'


def test_open_tag_omits_flag_when_trusted():
    assert format_attachment_open_tag(_att(name="p")) == '<attachment name="p">'


def test_open_tag_composes_mode_and_untrusted():
    tag = format_attachment_open_tag(_att(name="p", mode="index", untrusted=True))
    assert tag == '<attachment name="p" mode="index" untrusted="true">'


class _FakeResp:
    """Minimal urlopen context-manager stand-in."""

    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_url_handler_marks_fetched_text_untrusted():
    """The CLI trust fix: `tsu -f <url>` used to yield a trusted attachment. A
    fetched page is external content, so the handler now marks it untrusted."""
    h = GenericURLHandler()
    with (
        patch.object(h, "_get_content_type", return_value="text/plain"),
        patch("tsugite.attachments.url.urllib.request.urlopen", return_value=_FakeResp(b"page body")),
    ):
        att = h.fetch("https://example.com/p")
    assert att.content == "page body"
    assert att.untrusted is True


def test_url_handler_marks_image_reference_untrusted():
    h = GenericURLHandler()
    with patch.object(h, "_get_content_type", return_value="image/png"):
        att = h.fetch("https://example.com/pic.png")
    assert att.content_type is AttachmentContentType.IMAGE
    assert att.untrusted is True


def test_get_specific_handler_skips_the_generic_url_fallback():
    # A plain URL is claimed only by the generic fallback, which this excludes, so
    # the caller keeps its own generic path instead.
    assert get_specific_handler("https://example.com/page") is None


def test_get_specific_handler_returns_a_builtin_specific_handler():
    from tsugite.attachments.auto_context import AutoContextHandler

    assert isinstance(get_specific_handler("auto-context"), AutoContextHandler)


def test_get_specific_handler_includes_plugin_handlers(monkeypatch):
    class _FakeYouTube(AttachmentHandler):
        def can_handle(self, source: str) -> bool:
            return "youtu" in source

        def fetch(self, source: str) -> Attachment:
            return _att(name="yt")

    monkeypatch.setattr("tsugite.plugins.get_attachment_handlers", lambda: [_FakeYouTube()])
    assert isinstance(get_specific_handler("https://youtu.be/abc"), _FakeYouTube)
    assert get_specific_handler("https://example.com/p") is None
