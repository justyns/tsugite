"""Tests for the tsugite-web web_search tool."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from tsugite_web import web_search


@pytest.fixture
def mock_ddgs():
    """Provide a mocked DDGS search instance."""
    with patch("ddgs.DDGS") as mock_cls:
        instance = MagicMock()
        instance.__enter__ = MagicMock(return_value=instance)
        instance.__exit__ = MagicMock(return_value=None)
        mock_cls.return_value = instance
        yield instance


def test_registers_via_entry_point():
    """web_search is registered into the tool registry when the plugin loads."""
    import tsugite.tools as tools

    tools._ensure_tools_loaded()
    assert "web_search" in tools._tools


def test_web_search_basic(mock_ddgs):
    mock_ddgs.text.return_value = [
        {"title": "Test Result 1", "href": "https://example.com/1", "body": "First result snippet"},
        {"title": "Test Result 2", "href": "https://example.com/2", "body": "Second result snippet"},
    ]

    results = web_search("test query", max_results=2)

    assert len(results) == 2
    assert results[0]["title"] == "Test Result 1"
    assert results[0]["url"] == "https://example.com/1"
    assert results[0]["snippet"] == "First result snippet"


def test_web_search_max_results(mock_ddgs):
    mock_ddgs.text.return_value = [
        {"title": f"Result {i}", "href": f"https://example.com/{i}", "body": f"Snippet {i}"} for i in range(3)
    ]

    results = web_search("test query", max_results=3)

    assert len(results) == 3
    mock_ddgs.text.assert_called_once_with("test query", max_results=3)


def test_web_search_empty_results(mock_ddgs):
    mock_ddgs.text.return_value = []
    assert web_search("nonexistent query") == []


def test_web_search_missing_fields(mock_ddgs):
    mock_ddgs.text.return_value = [
        {"title": "Test Result", "href": "https://example.com"},
        {"title": "Test Result 2", "body": "Snippet only"},
        {},
    ]

    results = web_search("test query", max_results=3)

    assert len(results) == 3
    assert results[0]["snippet"] == ""
    assert results[1]["url"] == ""
    assert results[2]["title"] == ""


def test_web_search_error_handling(mock_ddgs):
    mock_ddgs.text.side_effect = Exception("Network error")

    with pytest.raises(RuntimeError) as exc_info:
        web_search("test query")

    assert "Web search failed" in str(exc_info.value)
    assert "Network error" in str(exc_info.value)


def test_web_search_without_ddgs(monkeypatch):
    """web_search raises a clear install hint when ddgs is unavailable."""
    monkeypatch.setitem(sys.modules, "ddgs", None)
    with pytest.raises(RuntimeError, match=r"tsugite-cli\[web\]"):
        web_search("anything")
