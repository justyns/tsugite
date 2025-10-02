"""Tests for HTTP tools."""

from unittest.mock import MagicMock, patch

import pytest

from tsugite.tools.http import web_search


def test_web_search_basic():
    """Test basic web search functionality."""
    mock_results = [
        {"title": "Test Result 1", "href": "https://example.com/1", "body": "First result snippet"},
        {"title": "Test Result 2", "href": "https://example.com/2", "body": "Second result snippet"},
    ]

    with patch("tsugite.tools.http.DDGS") as mock_ddgs:
        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=None)
        mock_instance.text = MagicMock(return_value=mock_results)
        mock_ddgs.return_value = mock_instance

        results = web_search("test query", max_results=2)

        assert len(results) == 2
        assert results[0]["title"] == "Test Result 1"
        assert results[0]["url"] == "https://example.com/1"
        assert results[0]["snippet"] == "First result snippet"
        assert results[1]["title"] == "Test Result 2"
        assert results[1]["url"] == "https://example.com/2"


def test_web_search_max_results():
    """Test that max_results parameter is respected."""
    mock_results = [
        {"title": f"Result {i}", "href": f"https://example.com/{i}", "body": f"Snippet {i}"} for i in range(10)
    ]

    with patch("tsugite.tools.http.DDGS") as mock_ddgs:
        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=None)
        mock_instance.text = MagicMock(return_value=mock_results[:3])
        mock_ddgs.return_value = mock_instance

        results = web_search("test query", max_results=3)

        assert len(results) == 3
        mock_instance.text.assert_called_once_with("test query", max_results=3)


def test_web_search_empty_results():
    """Test web search with empty results."""
    with patch("tsugite.tools.http.DDGS") as mock_ddgs:
        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=None)
        mock_instance.text = MagicMock(return_value=[])
        mock_ddgs.return_value = mock_instance

        results = web_search("nonexistent query")

        assert results == []


def test_web_search_missing_fields():
    """Test web search handles missing fields gracefully."""
    mock_results = [
        {"title": "Test Result", "href": "https://example.com"},  # Missing body
        {"title": "Test Result 2", "body": "Snippet only"},  # Missing href
        {},  # All fields missing
    ]

    with patch("tsugite.tools.http.DDGS") as mock_ddgs:
        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=None)
        mock_instance.text = MagicMock(return_value=mock_results)
        mock_ddgs.return_value = mock_instance

        results = web_search("test query", max_results=3)

        assert len(results) == 3
        assert results[0]["title"] == "Test Result"
        assert results[0]["url"] == "https://example.com"
        assert results[0]["snippet"] == ""
        assert results[1]["title"] == "Test Result 2"
        assert results[1]["url"] == ""
        assert results[1]["snippet"] == "Snippet only"
        assert results[2]["title"] == ""
        assert results[2]["url"] == ""
        assert results[2]["snippet"] == ""


def test_web_search_error_handling():
    """Test web search error handling."""
    with patch("tsugite.tools.http.DDGS") as mock_ddgs:
        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=None)
        mock_instance.text = MagicMock(side_effect=Exception("Network error"))
        mock_ddgs.return_value = mock_instance

        with pytest.raises(RuntimeError) as exc_info:
            web_search("test query")

        assert "Web search failed" in str(exc_info.value)
        assert "Network error" in str(exc_info.value)
