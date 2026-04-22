"""Tests for per-session state load/save helpers."""

import json
import stat

import pytest

from tsugite.core.state import load_state, save_state
from tsugite.exceptions import StateSerializationError


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "session1" / "state.json"
    save_state({"greeting": "hello", "count": 7}, path)

    loaded = load_state(path)
    assert loaded == {"greeting": "hello", "count": 7}


def test_save_creates_file_mode_0600(tmp_path):
    path = tmp_path / "s" / "state.json"
    save_state({"x": 1}, path)

    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600


def test_save_raises_for_non_json_value(tmp_path):
    path = tmp_path / "state.json"

    with pytest.raises(StateSerializationError) as exc:
        save_state({"s": {1, 2, 3}}, path, session_id="sess-1")
    assert exc.value.key == "s"
    assert exc.value.session_id == "sess-1"
    assert "s" in str(exc.value)


def test_per_key_size_cap(tmp_path):
    path = tmp_path / "state.json"

    with pytest.raises(StateSerializationError) as exc:
        save_state({"big": "x" * 1024}, path, max_bytes_per_key=128)
    assert exc.value.key == "big"
    assert exc.value.reason == "size-cap"


def test_total_size_cap(tmp_path):
    path = tmp_path / "state.json"
    data = {f"k{i}": "x" * 64 for i in range(10)}

    with pytest.raises(StateSerializationError) as exc:
        save_state(data, path, max_bytes_per_key=1024, max_bytes_total=256)
    assert exc.value.reason == "size-cap"


def test_load_missing_file_is_empty(tmp_path):
    path = tmp_path / "does_not_exist.json"
    assert load_state(path) == {}


def test_save_atomic_no_corruption_on_failure(tmp_path):
    """If save raises mid-way, the existing file must not be corrupted."""
    path = tmp_path / "state.json"
    save_state({"ok": "first"}, path)

    with pytest.raises(StateSerializationError):
        save_state({"ok": "first", "bad": {1, 2}}, path)

    assert load_state(path) == {"ok": "first"}


def test_load_from_existing_file(tmp_path):
    path = tmp_path / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pre": "existing"}))

    assert load_state(path) == {"pre": "existing"}


def test_save_serializes_nested_structures(tmp_path):
    path = tmp_path / "state.json"
    data = {"list": [1, 2, 3], "nested": {"a": True, "b": None}}
    save_state(data, path)

    assert load_state(path) == data
