"""An image uploaded to a non-vision model must not be inlined-and-dropped.

can_inline_file gates image inlining on the target model's vision
support: a non-vision model routes the image to the workspace-only path (saved
under uploads/ with a path hint) instead of silently vanishing into an image
block the model can't read.
"""

from unittest.mock import MagicMock, patch

from tsugite.attachments.delegation import can_inline_file
from tsugite.models import model_supports_vision


def _image(tmp_path):
    p = tmp_path / "photo.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"0" * 100)
    return p


def test_image_inlines_for_vision_model(tmp_path):
    assert can_inline_file(_image(tmp_path), 104, supports_vision=True) is True


def test_image_falls_to_workspace_only_for_non_vision_model(tmp_path):
    assert can_inline_file(_image(tmp_path), 104, supports_vision=False) is False


def test_text_attachment_unaffected_by_vision_flag(tmp_path):
    p = tmp_path / "notes.md"
    p.write_text("hello")
    assert can_inline_file(p, 5, supports_vision=False) is True


def test_unsupported_image_type_falls_to_workspace_only_even_for_vision_model(tmp_path):
    # svg/bmp/tiff aren't inline-able by any mainstream vision API, so they must
    # route to the workspace-only path (saved + hint), not get inlined-and-dropped.
    for name in ("diagram.svg", "old.bmp", "scan.tiff"):
        p = tmp_path / name
        p.write_bytes(b"data" * 10)
        assert can_inline_file(p, 40, supports_vision=True) is False, name


def test_default_is_vision_capable(tmp_path):
    # Callers that don't resolve a model (advisory upload flag) stay optimistic.
    assert can_inline_file(_image(tmp_path), 104) is True


# ── _chat's model→vision resolution feeds the gate above ──


def _provider_with_vision(vision: bool):
    provider = MagicMock()
    provider.get_model_info.return_value = MagicMock(supports_vision=vision)
    return provider


def test_session_supports_vision_true_when_model_has_vision():
    with patch("tsugite.providers.get_provider", return_value=_provider_with_vision(True)):
        assert model_supports_vision("openai:gpt-5.6-sol") is True


def test_session_supports_vision_false_when_model_lacks_vision():
    with patch("tsugite.providers.get_provider", return_value=_provider_with_vision(False)):
        assert model_supports_vision("openai:gpt-5.6-sol") is False


def test_session_supports_vision_defaults_true_on_resolution_failure():
    with patch("tsugite.providers.get_provider", side_effect=RuntimeError("no key")):
        assert model_supports_vision("openai:gpt-5.6-sol") is True
