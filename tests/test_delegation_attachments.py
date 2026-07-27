"""Shared delegation-attachment helpers backing spawn_agent(files=) / spawn_job(files=).

A delegating agent hands real files (images especially) to the child it delegates
to: the child's model sees the pixels. These cover the gate (inline vs on-disk),
the workspace traversal guard, materialization (text + image both work), and the
path hint that keeps non-inlinable files from silently vanishing.
"""

from unittest.mock import MagicMock, patch

import pytest

from tsugite.attachments.base import AttachmentContentType
from tsugite.attachments.delegation import (
    can_inline_file,
    format_delegation_hint,
    materialize_delegation_attachments,
    partition_delegation_files,
    resolve_delegation_files,
)
from tsugite.models import model_supports_vision

JPEG = b"\xff\xd8\xff\xe0" + b"0" * 100


def _img(directory, name="photo.jpg"):
    p = directory / name
    p.write_bytes(JPEG)
    return p


# -- resolve / traversal guard --


def test_resolve_accepts_relative_workspace_path(tmp_path):
    _img(tmp_path)
    assert resolve_delegation_files(["photo.jpg"], tmp_path) == [(tmp_path / "photo.jpg").resolve()]


def test_resolve_rejects_traversal(tmp_path):
    (tmp_path / "secret.txt").write_text("x")
    ws = tmp_path / "ws"
    ws.mkdir()
    with pytest.raises(ValueError, match="escapes"):
        resolve_delegation_files(["../secret.txt"], ws)


def test_resolve_rejects_absolute_outside_workspace(tmp_path):
    outside = tmp_path / "abs_secret.txt"
    outside.write_text("x")
    ws = tmp_path / "ws"
    ws.mkdir()
    with pytest.raises(ValueError, match="escapes"):
        resolve_delegation_files([str(outside)], ws)


def test_resolve_rejects_missing(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        resolve_delegation_files(["nope.jpg"], tmp_path)


# -- inline gate --


def test_image_inlines_for_vision_model(tmp_path):
    assert can_inline_file(_img(tmp_path), 104, supports_vision=True) is True


def test_image_hint_only_for_non_vision_model(tmp_path):
    assert can_inline_file(_img(tmp_path), 104, supports_vision=False) is False


def test_text_inlines_regardless_of_vision(tmp_path):
    p = tmp_path / "n.md"
    p.write_text("hi")
    assert can_inline_file(p, 5, supports_vision=False) is True


def test_oversize_text_not_inlined(tmp_path):
    p = tmp_path / "big.md"
    p.write_text("x")
    assert can_inline_file(p, 60 * 1024, supports_vision=True) is False


def test_svg_never_inlines(tmp_path):
    p = tmp_path / "d.svg"
    p.write_bytes(b"<svg/>")
    assert can_inline_file(p, 6, supports_vision=True) is False


# -- partition --


def test_partition_splits_inline_and_hint(tmp_path):
    img = _img(tmp_path)
    svg = tmp_path / "d.svg"
    svg.write_bytes(b"<svg/>")
    inline, hint = partition_delegation_files([img, svg], supports_vision=True)
    assert inline == [img]
    assert hint == [svg]


# -- materialize (text + image both work) --


def test_materialize_text_and_image(tmp_path):
    img = _img(tmp_path)
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    atts = materialize_delegation_attachments([img, txt])
    kinds = {a.content_type for a in atts}
    assert AttachmentContentType.IMAGE in kinds
    assert AttachmentContentType.TEXT in kinds
    img_att = next(a for a in atts if a.content_type == AttachmentContentType.IMAGE)
    assert img_att.mime_type == "image/jpeg"
    assert img_att.content  # base64 payload crossed over, not a bare path


# -- hint --


def test_hint_empty_when_no_paths():
    assert format_delegation_hint([]) == ""


def test_hint_lists_paths(tmp_path):
    p = tmp_path / "d.svg"
    hint = format_delegation_hint([p])
    assert str(p) in hint
    assert "open" in hint.lower()


# -- model_supports_vision (feeds the gate) --


def _provider(vision):
    prov = MagicMock()
    prov.get_model_info.return_value = MagicMock(supports_vision=vision)
    return prov


def test_model_supports_vision_true():
    with patch("tsugite.providers.get_provider", return_value=_provider(True)):
        assert model_supports_vision("openai:gpt-4o") is True


def test_model_supports_vision_false():
    with patch("tsugite.providers.get_provider", return_value=_provider(False)):
        assert model_supports_vision("openai:gpt-4o") is False


def test_model_supports_vision_defaults_true_on_resolution_failure():
    with patch("tsugite.providers.get_provider", side_effect=RuntimeError("no key")):
        assert model_supports_vision("openai:gpt-4o") is True
