"""Tests for HTTP tools."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from tsugite.tools.http import HttpResponse, http_request, web_search


@pytest.fixture
def mock_ddgs():
    """Provide a mocked DDGS search instance."""
    with patch("tsugite.tools.http.DDGS") as mock_cls:
        instance = MagicMock()
        instance.__enter__ = MagicMock(return_value=instance)
        instance.__exit__ = MagicMock(return_value=None)
        mock_cls.return_value = instance
        yield instance


# --- web_search tests ---


def test_web_search_basic(mock_ddgs):
    """Test basic web search functionality."""
    mock_ddgs.text.return_value = [
        {"title": "Test Result 1", "href": "https://example.com/1", "body": "First result snippet"},
        {"title": "Test Result 2", "href": "https://example.com/2", "body": "Second result snippet"},
    ]

    results = web_search("test query", max_results=2)

    assert len(results) == 2
    assert results[0]["title"] == "Test Result 1"
    assert results[0]["url"] == "https://example.com/1"
    assert results[0]["snippet"] == "First result snippet"
    assert results[1]["title"] == "Test Result 2"
    assert results[1]["url"] == "https://example.com/2"


def test_web_search_max_results(mock_ddgs):
    """Test that max_results parameter is respected."""
    mock_ddgs.text.return_value = [
        {"title": f"Result {i}", "href": f"https://example.com/{i}", "body": f"Snippet {i}"} for i in range(3)
    ]

    results = web_search("test query", max_results=3)

    assert len(results) == 3
    mock_ddgs.text.assert_called_once_with("test query", max_results=3)


def test_web_search_empty_results(mock_ddgs):
    """Test web search with empty results."""
    mock_ddgs.text.return_value = []

    results = web_search("nonexistent query")

    assert results == []


def test_web_search_missing_fields(mock_ddgs):
    """Test web search handles missing fields gracefully."""
    mock_ddgs.text.return_value = [
        {"title": "Test Result", "href": "https://example.com"},
        {"title": "Test Result 2", "body": "Snippet only"},
        {},
    ]

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


def test_web_search_error_handling(mock_ddgs):
    """Test web search error handling."""
    mock_ddgs.text.side_effect = Exception("Network error")

    with pytest.raises(RuntimeError) as exc_info:
        web_search("test query")

    assert "Web search failed" in str(exc_info.value)
    assert "Network error" in str(exc_info.value)


# --- http_request tests ---


def _mock_response(status_code=200, json_data=None, text="OK", headers=None):
    """Create a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.headers = httpx.Headers(headers or {"content-type": "application/json"})
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = json.JSONDecodeError("", "", 0)
    resp.raise_for_status.return_value = None
    return resp


@pytest.fixture
def mock_httpx_client():
    """Provide a mocked httpx.Client with context manager support."""
    with patch("tsugite.tools.http.httpx.Client") as mock_client_cls:
        client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=None)
        yield client


def test_http_request_post_dict_body(mock_httpx_client):
    """POST with dict body auto-serializes to JSON."""
    mock_httpx_client.request.return_value = _mock_response(200, {"id": 1})

    result = http_request("https://api.example.com/items", body={"name": "test"})

    assert isinstance(result, HttpResponse)
    assert result.status_code == 200
    assert result.body == {"id": 1}
    call_kwargs = mock_httpx_client.request.call_args
    assert call_kwargs.kwargs["json"] == {"name": "test"}


def test_http_request_put_string_body(mock_httpx_client):
    """PUT with string body uses content kwarg."""
    mock_httpx_client.request.return_value = _mock_response(200, text="updated")

    result = http_request("https://api.example.com/items/1", method="PUT", body="raw data")

    call_kwargs = mock_httpx_client.request.call_args
    assert call_kwargs.kwargs["method"] == "PUT"
    assert call_kwargs.kwargs["content"] == "raw data"
    assert "json" not in call_kwargs.kwargs


def test_http_request_patch_dict_body(mock_httpx_client):
    """PATCH with dict body works like POST."""
    mock_httpx_client.request.return_value = _mock_response(200, {"updated": True})

    result = http_request("https://api.example.com/items/1", method="PATCH", body={"name": "new"})

    assert result.body == {"updated": True}
    call_kwargs = mock_httpx_client.request.call_args
    assert call_kwargs.kwargs["method"] == "PATCH"
    assert call_kwargs.kwargs["json"] == {"name": "new"}


def test_http_request_delete_no_body(mock_httpx_client):
    """DELETE with no body."""
    mock_httpx_client.request.return_value = _mock_response(204, text="")

    result = http_request("https://api.example.com/items/1", method="DELETE")

    assert result.status_code == 204
    call_kwargs = mock_httpx_client.request.call_args
    assert call_kwargs.kwargs["method"] == "DELETE"
    assert "json" not in call_kwargs.kwargs
    assert "content" not in call_kwargs.kwargs


def test_http_request_get_no_body(mock_httpx_client):
    """GET works as a verbose fetch returning HttpResponse."""
    mock_httpx_client.request.return_value = _mock_response(200, {"data": [1, 2, 3]})

    result = http_request("https://api.example.com/items", method="GET")

    assert isinstance(result, HttpResponse)
    assert result.body == {"data": [1, 2, 3]}


def test_http_request_timeout_error(mock_httpx_client):
    """Timeout raises RuntimeError."""
    mock_httpx_client.request.side_effect = httpx.TimeoutException("timed out")

    with pytest.raises(RuntimeError, match="Request timed out"):
        http_request("https://api.example.com/slow", timeout=5)


def test_http_request_http_error(mock_httpx_client):
    """HTTP error status raises RuntimeError."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 404
    resp.text = "Not Found"
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404", request=MagicMock(), response=resp
    )
    mock_httpx_client.request.return_value = resp

    with pytest.raises(RuntimeError, match="HTTP error 404"):
        http_request("https://api.example.com/missing")


def test_http_request_non_json_response(mock_httpx_client):
    """Non-JSON response body falls back to text string."""
    mock_httpx_client.request.return_value = _mock_response(200, json_data=None, text="plain text response")

    result = http_request("https://api.example.com/text", method="GET")

    assert result.body == "plain text response"
    assert result.status_code == 200


def test_http_request_returns_headers(mock_httpx_client):
    """Response includes headers as dict."""
    mock_httpx_client.request.return_value = _mock_response(
        200, {"ok": True}, headers={"x-request-id": "abc123", "content-type": "application/json"}
    )

    result = http_request("https://api.example.com/items", body={"a": 1})

    assert "x-request-id" in result.headers
    assert result.headers["x-request-id"] == "abc123"
