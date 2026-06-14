"""The hint for non-inlined uploads must give the agent a tool-ready `uploads/<name>` path.

Non-inlined files are saved to <workspace>/uploads/ and the agent must open them itself,
so a bare filename leaves it guessing (usually the workspace root) and failing to read them.
"""

from tsugite.daemon.adapters.http import _format_upload_message_suffix


def test_workspace_only_hint_includes_uploads_path():
    suffix = _format_upload_message_suffix(["data.csv"], [])
    assert "uploads/data.csv" in suffix, f"hint must point at uploads/<name>, got: {suffix!r}"


def test_multiple_workspace_only_files_each_get_uploads_path():
    suffix = _format_upload_message_suffix(["a.log", "b.bin"], [])
    assert "uploads/a.log" in suffix
    assert "uploads/b.bin" in suffix


def test_attachments_hint_still_mentions_uploads_dir():
    suffix = _format_upload_message_suffix([], ["img.png"])
    assert "uploads/" in suffix
    assert "img.png" in suffix


def test_no_files_yields_empty_suffix():
    assert _format_upload_message_suffix([], []) == ""
