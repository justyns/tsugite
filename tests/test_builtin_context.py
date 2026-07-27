"""Built-in Workspace-file context provider.

Pure-core provider (stdlib + the ``ctx`` dict, no daemon imports): ``choices``
lists a session workspace's text files and ``capture`` reads a picked file's
text as a single context item. The security-critical contract is the ``..`` /
absolute-path traversal guard on ``capture`` and the binary / oversize / missing
filtering that keeps a bad pick from poisoning a send.
"""

from __future__ import annotations

import pytest

from tsugite import context as ctx_module
from tsugite.builtin_context import WORKSPACE_FILE_PROVIDER, register_builtin_providers
from tsugite.context import get_context_provider, reset_context_providers


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_context_providers()
    yield
    reset_context_providers()


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("alpha")
    (ws / "sub").mkdir()
    (ws / "sub" / "b.py").write_text("print('b')")
    # Everything below must be skipped by choices():
    (ws / ".git").mkdir()
    (ws / ".git" / "config").write_text("[core]")
    (ws / ".secret").write_text("shh")  # hidden file
    (ws / "node_modules").mkdir()
    (ws / "node_modules" / "pkg.js").write_text("module.exports = {}")
    return ws


def _ctx(ws):
    return {"workspace_dir": ws}


def test_provider_shape():
    p = WORKSPACE_FILE_PROVIDER
    assert p.key == "file"
    assert p.label == "Workspace file"
    assert p.icon == "file"
    assert p.picker is True
    assert p.capture is not None
    assert p.choices is not None
    assert p.in_menu is True


def test_choices_lists_files_sorted(workspace):
    choices = WORKSPACE_FILE_PROVIDER.choices(_ctx(workspace))
    assert [c.value for c in choices] == ["a.txt", "sub/b.py"]
    assert [c.label for c in choices] == [c.value for c in choices]


def test_choices_skips_git_hidden_and_node_modules(workspace):
    values = [c.value for c in WORKSPACE_FILE_PROVIDER.choices(_ctx(workspace))]
    assert not any(v.startswith(".git") for v in values)
    assert ".secret" not in values
    assert not any("node_modules" in v for v in values)


def test_choices_skips_binaries(workspace):
    (workspace / "img.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    values = [c.value for c in WORKSPACE_FILE_PROVIDER.choices(_ctx(workspace))]
    assert "img.png" not in values
    assert values == ["a.txt", "sub/b.py"]


def test_choices_caps_result(workspace, monkeypatch):
    for i in range(60):
        (workspace / f"f{i:03d}.txt").write_text("x")
    monkeypatch.setattr("tsugite.builtin_context._MAX_CHOICES", 50)
    assert len(WORKSPACE_FILE_PROVIDER.choices(_ctx(workspace))) == 50


def test_choices_missing_or_absent_workspace_returns_empty(tmp_path):
    assert WORKSPACE_FILE_PROVIDER.choices({"workspace_dir": tmp_path / "nope"}) == []
    assert WORKSPACE_FILE_PROVIDER.choices({}) == []
    assert WORKSPACE_FILE_PROVIDER.choices({"workspace_dir": None}) == []


def test_capture_reads_file(workspace):
    items = WORKSPACE_FILE_PROVIDER.capture("a.txt", _ctx(workspace))
    assert len(items) == 1
    item = items[0]
    assert item.key == "file:a.txt"
    assert item.label == "a.txt"
    assert item.value == "alpha"


def test_capture_reads_nested_file(workspace):
    items = WORKSPACE_FILE_PROVIDER.capture("sub/b.py", _ctx(workspace))
    assert [(i.key, i.value) for i in items] == [("file:sub/b.py", "print('b')")]


def test_capture_rejects_parent_traversal(workspace, tmp_path):
    (tmp_path / "outside.txt").write_text("top secret")
    assert WORKSPACE_FILE_PROVIDER.capture("../outside.txt", _ctx(workspace)) == []


def test_capture_rejects_absolute_path(workspace):
    assert WORKSPACE_FILE_PROVIDER.capture("/etc/hostname", _ctx(workspace)) == []


def test_capture_missing_file_returns_empty(workspace):
    assert WORKSPACE_FILE_PROVIDER.capture("nope.txt", _ctx(workspace)) == []


def test_capture_none_or_empty_arg_returns_empty(workspace):
    assert WORKSPACE_FILE_PROVIDER.capture(None, _ctx(workspace)) == []
    assert WORKSPACE_FILE_PROVIDER.capture("", _ctx(workspace)) == []


def test_capture_binary_returns_empty(workspace):
    (workspace / "bin.dat").write_bytes(b"\x00\x01\x02\x03rest")
    assert WORKSPACE_FILE_PROVIDER.capture("bin.dat", _ctx(workspace)) == []


def test_capture_head_caps_value(workspace, monkeypatch):
    monkeypatch.setattr("tsugite.builtin_context._MAX_VALUE_CHARS", 10)
    (workspace / "big.txt").write_text("0123456789ABCDEFGHIJ")
    items = WORKSPACE_FILE_PROVIDER.capture("big.txt", _ctx(workspace))
    assert len(items) == 1
    assert items[0].value == "0123456789"


def test_capture_oversized_file_returns_empty(workspace, monkeypatch):
    monkeypatch.setattr("tsugite.builtin_context._MAX_FILE_BYTES", 8)
    (workspace / "toolong.txt").write_text("way more than eight bytes")
    assert WORKSPACE_FILE_PROVIDER.capture("toolong.txt", _ctx(workspace)) == []


def test_capture_no_workspace_returns_empty(tmp_path):
    assert WORKSPACE_FILE_PROVIDER.capture("a.txt", {}) == []
    assert WORKSPACE_FILE_PROVIDER.capture("a.txt", {"workspace_dir": tmp_path / "nope"}) == []


def test_register_builtin_providers_registers_file(monkeypatch):
    monkeypatch.setattr(ctx_module, "ensure_loaded", lambda: None)
    reset_context_providers()
    register_builtin_providers()
    assert get_context_provider("file") is WORKSPACE_FILE_PROVIDER


def test_ensure_loaded_reregisters_builtin_after_reset():
    """The builtin must survive ``reset_context_providers()`` because it is
    registered by a *called* function in ``ensure_loaded``, not an import side
    effect (a cached module import would not re-run after a reset)."""
    reset_context_providers()
    assert get_context_provider("file") is WORKSPACE_FILE_PROVIDER
    reset_context_providers()
    assert get_context_provider("file") is WORKSPACE_FILE_PROVIDER
