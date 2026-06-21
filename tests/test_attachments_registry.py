"""Phase 1e: attachment handlers are discoverable via the tsugite.attachments group."""

from tsugite.attachments import get_handler
from tsugite.attachments.base import Attachment, AttachmentContentType, AttachmentHandler


class _DummyHandler(AttachmentHandler):
    def can_handle(self, source):
        return source.startswith("dummy:")

    def fetch(self, source):
        return Attachment(name="dummy", content="x", content_type=AttachmentContentType.TEXT, mime_type="text/plain")


def _patch_entry_points(monkeypatch, handler):
    import tsugite.plugins as plugins

    class _EP:
        name = "dummy"
        value = "test:dummy"

        def load(self):
            return lambda config: handler

    real = plugins.importlib.metadata.entry_points
    monkeypatch.setattr(
        plugins.importlib.metadata,
        "entry_points",
        lambda **kw: [_EP()] if kw.get("group") == "tsugite.attachments" else real(**kw),
    )
    plugins.reset_attachment_handlers()


def test_plugin_attachment_handler_resolves(monkeypatch):
    _patch_entry_points(monkeypatch, _DummyHandler())
    assert isinstance(get_handler("dummy:abc"), _DummyHandler)


def test_plugin_handlers_rank_before_url_fallback(monkeypatch):
    """Plugin handlers must win over the generic URL/file fallbacks for overlapping sources."""
    from tsugite.attachments.url import GenericURLHandler

    class _URLishHandler(AttachmentHandler):
        def can_handle(self, source):
            return source.startswith("https://example.test/")

        def fetch(self, source):
            return Attachment(name="x", content="x", content_type=AttachmentContentType.TEXT, mime_type="text/plain")

    _patch_entry_points(monkeypatch, _URLishHandler())

    handler = get_handler("https://example.test/page")
    assert isinstance(handler, _URLishHandler)
    assert not isinstance(handler, GenericURLHandler)


def test_builtin_handlers_still_resolve(monkeypatch):
    """With no plugins, built-in handlers (inline) still resolve."""
    import tsugite.plugins as plugins

    plugins.reset_attachment_handlers()
    from tsugite.attachments.inline import InlineHandler

    assert isinstance(get_handler("inline"), InlineHandler)
