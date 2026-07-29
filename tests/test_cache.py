"""Content-cache behavior: round-trip, atomic writes, metadata shape."""

import pytest

from tsugite import cache


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Isolate the attachments cache under tmp_path and return its directory."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path / "cache" / "tsugite" / "attachments"


def test_save_and_get_roundtrip(cache_dir):
    cache.save_to_cache("src://a", "hello world")
    assert cache.get_cached_content("src://a") == "hello world"
    assert cache.get_cached_content("src://missing") is None


def test_save_overwrites_existing(cache_dir):
    cache.save_to_cache("src://a", "first")
    cache.save_to_cache("src://a", "second")
    assert cache.get_cached_content("src://a") == "second"


def test_write_is_atomic_failed_replace_keeps_old_content(cache_dir, monkeypatch):
    """A write that dies at the final replace must not destroy the prior content.

    Load-bearing for the temp-then-replace fix: with a plain in-place write the
    second save would simply succeed (no replace to fail), so the pytest.raises
    below would fail; the atomic write funnels the failure through os.replace and
    leaves the original file untouched.
    """
    cache.save_to_cache("src://a", "GOOD")

    def boom(*_args, **_kwargs):
        raise OSError("replace failed")

    monkeypatch.setattr("tsugite.utils.os.replace", boom)
    with pytest.raises(RuntimeError):
        cache.save_to_cache("src://a", "TRUNCATED")

    assert cache.get_cached_content("src://a") == "GOOD"
    assert list(cache_dir.glob("*.tmp")) == []  # no stray temp left behind


def test_no_temp_residue_after_successful_save(cache_dir):
    cache.save_to_cache("src://a", "content")
    assert list(cache_dir.glob("*.tmp")) == []


def test_metadata_and_listing(cache_dir):
    cache.save_to_cache("src://a", "abc")
    entries = cache.list_cache()
    key = cache.get_cache_key("src://a")
    assert key in entries
    assert entries[key]["source"] == "src://a"
    assert entries[key]["size"] == len("abc")
    assert cache.get_cache_info("src://a") == entries[key]
    assert cache.get_cache_info("src://missing") is None


def test_clear_cache(cache_dir):
    cache.save_to_cache("src://a", "one")
    cache.save_to_cache("src://b", "two")
    assert cache.clear_cache("src://a") == 1
    assert cache.get_cached_content("src://a") is None
    assert cache.get_cached_content("src://b") == "two"
    assert cache.get_cache_info("src://a") is None
    assert cache.clear_cache() == 1  # the remaining entry
    assert cache.get_cached_content("src://b") is None
