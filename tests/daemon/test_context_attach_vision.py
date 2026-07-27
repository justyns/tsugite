"""An image uploaded to a non-vision model must not be inlined-and-dropped.

_should_context_attach gates image inlining on the target model's vision
support: a non-vision model routes the image to the workspace-only path (saved
under uploads/ with a path hint) instead of silently vanishing into an image
block the model can't read.
"""

from unittest.mock import MagicMock, patch

from tsugite_daemon.adapters.http import HTTPServer
from tsugite_daemon.adapters.http.helpers import _should_context_attach
from tsugite_daemon.config import HTTPConfig
from tsugite_daemon.webhook_store import WebhookStore


def _image(tmp_path):
    p = tmp_path / "photo.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"0" * 100)
    return p


def test_image_inlines_for_vision_model(tmp_path):
    assert _should_context_attach(_image(tmp_path), 104, supports_vision=True) is True


def test_image_falls_to_workspace_only_for_non_vision_model(tmp_path):
    assert _should_context_attach(_image(tmp_path), 104, supports_vision=False) is False


def test_text_attachment_unaffected_by_vision_flag(tmp_path):
    p = tmp_path / "notes.md"
    p.write_text("hello")
    assert _should_context_attach(p, 5, supports_vision=False) is True


def test_unsupported_image_type_falls_to_workspace_only_even_for_vision_model(tmp_path):
    # svg/bmp/tiff aren't inline-able by any mainstream vision API, so they must
    # route to the workspace-only path (saved + hint), not get inlined-and-dropped.
    for name in ("diagram.svg", "old.bmp", "scan.tiff"):
        p = tmp_path / name
        p.write_bytes(b"data" * 10)
        assert _should_context_attach(p, 40, supports_vision=True) is False, name


def test_default_is_vision_capable(tmp_path):
    # Callers that don't resolve a model (advisory upload flag) stay optimistic.
    assert _should_context_attach(_image(tmp_path), 104) is True


# ── _chat's model→vision resolution feeds the gate above ──


def _server(tmp_path):
    return HTTPServer(
        config=HTTPConfig(enabled=True),
        adapters={},
        webhook_store=WebhookStore(tmp_path / "webhooks.json"),
        agent_configs={},
    )


class _FakeAdapter:
    def __init__(self, model):
        self._model = model
        self.session_store = MagicMock()
        self.session_store.get_model_override.return_value = None

    def resolve_model(self):
        return self._model

    def resolve_session_model(self, session_id):
        override = self.session_store.get_model_override(session_id) if session_id else None
        return override or self.resolve_model()


def _provider_with_vision(vision: bool):
    provider = MagicMock()
    provider.get_model_info.return_value = MagicMock(supports_vision=vision)
    return provider


def test_session_supports_vision_true_when_model_has_vision(tmp_path):
    server = _server(tmp_path)
    with patch("tsugite.providers.get_provider", return_value=_provider_with_vision(True)):
        assert server._session_supports_vision(_FakeAdapter("openai:gpt-5.6-sol"), None) is True


def test_session_supports_vision_false_when_model_lacks_vision(tmp_path):
    server = _server(tmp_path)
    with patch("tsugite.providers.get_provider", return_value=_provider_with_vision(False)):
        assert server._session_supports_vision(_FakeAdapter("openai:gpt-5.6-sol"), None) is False


def test_session_supports_vision_defaults_true_on_resolution_failure(tmp_path):
    server = _server(tmp_path)
    with patch("tsugite.providers.get_provider", side_effect=RuntimeError("no key")):
        assert server._session_supports_vision(_FakeAdapter("openai:gpt-5.6-sol"), None) is True
