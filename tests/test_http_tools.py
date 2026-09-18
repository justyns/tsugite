"""Tests for HTTP tools."""

import json
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

from tsugite.exceptions import ToolUnavailableError
from tsugite.tools.http import HttpResponse, _default_headers, download_file, fetch_json, fetch_text, http_request

# --- http_request tests ---


_DEFAULT_TEXT = object()


def _mock_response(status_code=200, json_data=None, text=_DEFAULT_TEXT, headers=None):
    """Create a mock httpx.Response. If json_data is given and text isn't, text is the JSON string."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.headers = httpx.Headers(headers or {"content-type": "application/json"})
    if text is _DEFAULT_TEXT:
        resp.text = json.dumps(json_data) if json_data is not None else "OK"
    else:
        resp.text = text
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

    result = http_request("https://api.example.com/items", method="POST", body={"name": "test"})

    assert isinstance(result, HttpResponse)
    assert result.status_code == 200
    assert result.text == '{"id": 1}'
    assert result.json() == {"id": 1}
    call_kwargs = mock_httpx_client.request.call_args
    assert call_kwargs.kwargs["method"] == "POST"
    assert call_kwargs.kwargs["json"] == {"name": "test"}


def test_http_request_default_method_is_get(mock_httpx_client):
    """Omitting `method=` must default to GET, not POST.

    Background: a POST default caused real data loss when an LLM omitted the
    method on a Vikunja read-then-update flow - the empty-body POST cleared
    fields. GET is safe and idempotent and matches the convention of
    `requests`, `urllib`, and the rest of the http tools in this module.
    """
    mock_httpx_client.request.return_value = _mock_response(200, {"data": [1, 2, 3]})

    http_request("https://api.example.com/items")

    call_kwargs = mock_httpx_client.request.call_args
    assert call_kwargs.kwargs["method"] == "GET"


def test_http_request_put_string_body(mock_httpx_client):
    """PUT with string body uses content kwarg."""
    mock_httpx_client.request.return_value = _mock_response(200, text="updated")

    http_request("https://api.example.com/items/1", method="PUT", body="raw data")

    call_kwargs = mock_httpx_client.request.call_args
    assert call_kwargs.kwargs["method"] == "PUT"
    assert call_kwargs.kwargs["content"] == "raw data"
    assert "json" not in call_kwargs.kwargs


def test_http_request_patch_dict_body(mock_httpx_client):
    """PATCH with dict body works like POST."""
    mock_httpx_client.request.return_value = _mock_response(200, {"updated": True})

    result = http_request("https://api.example.com/items/1", method="PATCH", body={"name": "new"})

    assert result.json() == {"updated": True}
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
    assert result.json() == {"data": [1, 2, 3]}


def test_http_request_timeout_error(mock_httpx_client):
    mock_httpx_client.request.side_effect = httpx.TimeoutException("timed out")

    with pytest.raises(TimeoutError, match="Request timed out after 5s"):
        http_request("https://api.example.com/slow", timeout=5)


@pytest.mark.parametrize("status_code", [401, 404, 500])
def test_http_request_returns_non_2xx_responses(mock_httpx_client, status_code):
    body = f'{{"error":{status_code}}}'
    resp = _mock_response(status_code=status_code, text=body)
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(str(status_code), request=MagicMock(), response=resp)
    mock_httpx_client.request.return_value = resp

    result = http_request("https://api.example.com/protected")

    assert isinstance(result, HttpResponse)
    assert result.status_code == status_code
    assert result.text == body


def test_http_request_non_json_response(mock_httpx_client):
    """Non-JSON response exposes raw text; .json() raises a clear ValueError."""
    mock_httpx_client.request.return_value = _mock_response(200, json_data=None, text="plain text response")

    result = http_request("https://api.example.com/text", method="GET")

    assert result.text == "plain text response"
    assert result.status_code == 200
    with pytest.raises(ValueError, match="not valid JSON"):
        result.json()


def test_http_request_json_response_exposes_both_text_and_parsed(mock_httpx_client):
    """A JSON response keeps .text as the raw string and parses via .json()."""
    mock_httpx_client.request.return_value = _mock_response(200, {"a": 1, "b": [2, 3]})

    result = http_request("https://api.example.com/items", method="GET")

    assert isinstance(result.text, str)
    assert json.loads(result.text) == {"a": 1, "b": [2, 3]}
    assert result.json() == {"a": 1, "b": [2, 3]}


def test_http_response_body_aliases_text(mock_httpx_client):
    """`.body` is a back-compat alias for `.text`."""
    mock_httpx_client.request.return_value = _mock_response(200, json_data=None, text="hello")

    result = http_request("https://api.example.com/text", method="GET")

    assert result.body == result.text == "hello"


# --- fetch_text tests ---

SAMPLE_HTML = "<html><head><title>Test</title></head><body><h1>Hello</h1><p>World</p></body></html>"

SAMPLE_ARTICLE_HTML = """<html><head><title>My Article</title></head><body>
<nav>Menu items</nav>
<article><h1>Article Title</h1><p>This is the main article content with enough text to be recognized.</p>
<p>Another paragraph of substantial content for the readability algorithm to detect.</p></article>
<footer>Footer stuff</footer>
</body></html>"""


def test_fetch_text_default_strips_html(mock_httpx_client):
    """Default fetch_text strips HTML (strip_html=True by default)."""
    mock_httpx_client.request.return_value = _mock_response(
        200, text=SAMPLE_HTML, headers={"content-type": "text/html"}
    )
    result = fetch_text("https://example.com")
    assert "<h1>" not in result
    assert "Hello" in result


def test_fetch_text_raw_html_when_strip_disabled(mock_httpx_client):
    """strip_html=False returns raw HTML."""
    mock_httpx_client.request.return_value = _mock_response(
        200, text=SAMPLE_HTML, headers={"content-type": "text/html"}
    )
    result = fetch_text("https://example.com", strip_html=False)
    assert "<h1>Hello</h1>" in result


def test_fetch_text_strip_html(mock_httpx_client):
    """strip_html=True converts HTML to markdown."""
    mock_httpx_client.request.return_value = _mock_response(
        200, text=SAMPLE_HTML, headers={"content-type": "text/html"}
    )
    result = fetch_text("https://example.com", strip_html=True)
    assert "<h1>" not in result
    assert "Hello" in result
    assert "World" in result


def test_fetch_text_strip_html_non_html_passthrough(mock_httpx_client):
    """strip_html=True with non-HTML content-type returns raw text."""
    mock_httpx_client.request.return_value = _mock_response(
        200, text='{"key": "value"}', headers={"content-type": "application/json"}
    )
    result = fetch_text("https://example.com", strip_html=True)
    assert result == '{"key": "value"}'


def test_fetch_text_extract_article(mock_httpx_client):
    """extract_article=True extracts article content."""
    mock_httpx_client.request.return_value = _mock_response(
        200, text=SAMPLE_ARTICLE_HTML, headers={"content-type": "text/html; charset=utf-8"}
    )
    result = fetch_text("https://example.com", extract_article=True)
    assert "main article content" in result
    assert "<nav>" not in result
    assert "<footer>" not in result


def test_fetch_text_extract_article_non_html_passthrough(mock_httpx_client):
    """extract_article=True with non-HTML content-type returns raw text."""
    plain = "Just plain text"
    mock_httpx_client.request.return_value = _mock_response(200, text=plain, headers={"content-type": "text/plain"})
    result = fetch_text("https://example.com", extract_article=True)
    assert result == plain


def test_fetch_text_extract_article_falls_back_to_page(mock_httpx_client):
    """A sliver of a much larger page means readability found no article, so the page is used."""
    page = "<html><body><div id='app'>" + "<div>Body text the extractor discarded. </div>" * 40 + "</div></body></html>"
    mock_httpx_client.request.return_value = _mock_response(200, text=page, headers={"content-type": "text/html"})

    with patch("readability.Document") as doc:
        doc.return_value.summary.return_value = '<body id="readabilityBody"><p>Generated with AI assistance.</p></body>'
        result = fetch_text("https://example.com", extract_article=True)

    assert "Body text the extractor discarded" in result


def test_fetch_text_extract_article_keeps_substantial_extraction(mock_httpx_client):
    """A substantial extraction is kept even when the page is much bigger."""
    article = "<p>" + "Real article sentence that readability kept. " * 20 + "</p>"
    page = f"<html><body><nav>{'Menu link ' * 200}</nav><article>{article}</article></body></html>"
    mock_httpx_client.request.return_value = _mock_response(200, text=page, headers={"content-type": "text/html"})

    with patch("readability.Document") as doc:
        doc.return_value.summary.return_value = f'<body id="readabilityBody">{article}</body>'
        result = fetch_text("https://example.com", extract_article=True)

    assert "Real article sentence" in result
    assert "Menu link" not in result


def test_fetch_text_extract_article_raises_on_empty_page(mock_httpx_client):
    """A client-rendered shell has no text to fall back to."""
    shell = (
        "<html><head><title>App</title></head><body><div id='root'></div><script src='/a.js'></script></body></html>"
    )
    mock_httpx_client.request.return_value = _mock_response(200, text=shell, headers={"content-type": "text/html"})

    with pytest.raises(RuntimeError, match="no readable text"):
        fetch_text("https://example.com", extract_article=True)


def test_fetch_text_extract_article_takes_precedence(mock_httpx_client):
    """When both flags set, extract_article takes precedence."""
    mock_httpx_client.request.return_value = _mock_response(
        200, text=SAMPLE_ARTICLE_HTML, headers={"content-type": "text/html"}
    )
    result = fetch_text("https://example.com", strip_html=True, extract_article=True)
    assert "main article content" in result


def test_fetch_text_timeout_error(mock_httpx_client):
    mock_httpx_client.request.side_effect = httpx.TimeoutException("timed out")
    with pytest.raises(TimeoutError, match="Request timed out after 30s"):
        fetch_text("https://example.com")


def test_fetch_json_timeout_error(mock_httpx_client):
    mock_httpx_client.request.side_effect = httpx.TimeoutException("timed out")
    with pytest.raises(TimeoutError, match="Request timed out after 5s"):
        fetch_json("https://api.example.com/slow", timeout=5)


def test_fetch_json_non_json_body_is_not_double_wrapped(mock_httpx_client):
    resp = _mock_response(200, text="not json")
    resp.json.side_effect = json.JSONDecodeError("Expecting value", "not json", 0)
    mock_httpx_client.request.return_value = resp

    with pytest.raises(RuntimeError, match=r"\AInvalid JSON response"):
        fetch_json("https://api.example.com/text")


def test_download_file_timeout_error(mock_httpx_client, tmp_path):

    mock_httpx_client.stream.side_effect = httpx.TimeoutException("timed out")
    with pytest.raises(TimeoutError, match="Request timed out after 60s"):
        download_file("https://example.com/big", str(tmp_path / "out.bin"))


def test_http_request_returns_headers(mock_httpx_client):
    """Response includes headers as dict."""
    mock_httpx_client.request.return_value = _mock_response(
        200, {"ok": True}, headers={"x-request-id": "abc123", "content-type": "application/json"}
    )

    result = http_request("https://api.example.com/items", method="POST", body={"a": 1})

    assert "x-request-id" in result.headers
    assert result.headers["x-request-id"] == "abc123"


# --- User-Agent tests ---


def test_default_headers_sets_user_agent():
    """Default headers include User-Agent."""
    headers = _default_headers()
    assert "User-Agent" in headers
    assert headers["User-Agent"].startswith("Tsugite/")


def test_default_headers_overwrites_custom_user_agent():
    """Caller-provided User-Agent is dropped; framework UA is enforced.

    Agent code that hand-rolls a User-Agent has been observed leaking PII
    (e.g. user emails copied from system context). The framework value wins.
    """
    headers = _default_headers({"User-Agent": "Custom/1.0"})
    assert headers["User-Agent"].startswith("Tsugite/")


def test_default_headers_disabled_user_agent():
    """User-Agent is omitted when config sets user_agent to empty string."""
    with patch("tsugite.user_agent.get_user_agent", return_value=None):
        headers = _default_headers()
        assert "User-Agent" not in headers


def test_default_headers_custom_config_user_agent():
    """User-Agent uses config value when set."""
    with patch("tsugite.user_agent.get_user_agent", return_value="MyBot/2.0"):
        headers = _default_headers()
        assert headers["User-Agent"] == "MyBot/2.0"


def test_request_sends_user_agent(mock_httpx_client):
    """HTTP requests include User-Agent header."""
    mock_httpx_client.request.return_value = _mock_response(200, {"ok": True})

    http_request("https://api.example.com/test", method="GET")

    call_kwargs = mock_httpx_client.request.call_args
    sent_headers = call_kwargs.kwargs["headers"]
    assert "User-Agent" in sent_headers
    assert sent_headers["User-Agent"].startswith("Tsugite/")


# --- redirect following (3xx was the top tool-failure class) ---


@pytest.fixture
def redirect_transport(monkeypatch):
    """Route tool requests through a real httpx.Client backed by MockTransport
    so genuine httpx redirect semantics (hop following, 307 method+body
    preservation, loop bounding) are exercised, not mocked away."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/old":
            return httpx.Response(301, headers={"location": "/new"})
        if path == "/307post":
            return httpx.Response(307, headers={"location": "/posted"})
        if path == "/posted":
            return httpx.Response(
                200,
                json={"method": request.method, "body": request.content.decode()},
            )
        if path == "/loop":
            return httpx.Response(302, headers={"location": "/loop"})
        return httpx.Response(200, text="final", headers={"content-type": "text/plain"})

    real_client = httpx.Client

    def patched_client(**kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(**kwargs)

    monkeypatch.setattr("tsugite.tools.http.httpx.Client", patched_client)


def test_fetch_text_follows_redirects_by_default(redirect_transport):
    assert fetch_text("https://example.com/old") == "final"


def test_http_request_follows_redirects_and_exposes_final_url(redirect_transport):
    result = http_request("https://example.com/old")
    assert result.status_code == 200
    assert result.text == "final"
    assert result.url.endswith("/new"), f"final URL after redirect must be exposed; got {result.url!r}"


def test_http_request_opt_out_returns_inspectable_redirect(redirect_transport):
    """follow_redirects=False exists to inspect the 3xx - it must return the
    redirect response, not raise."""
    result = http_request("https://example.com/old", follow_redirects=False)
    assert result.status_code == 301
    assert result.headers.get("location") == "/new"


def test_http_request_307_preserves_method_and_body(redirect_transport):
    result = http_request("https://example.com/307post", method="POST", body="payload")
    parsed = result.json()
    assert parsed["method"] == "POST"
    assert parsed["body"] == "payload"


def test_fetch_text_redirect_loop_is_bounded(redirect_transport):
    with pytest.raises(RuntimeError):
        fetch_text("https://example.com/loop")


def test_download_file_follows_redirects(redirect_transport, tmp_path):

    target = tmp_path / "out.txt"
    download_file("https://example.com/old", str(target))
    assert target.read_text() == "final"


def test_fetch_text_extract_article_without_readability_is_unavailable(mock_httpx_client):
    mock_httpx_client.request.return_value = _mock_response(
        200, text="<html><body><p>hi</p></body></html>", headers={"content-type": "text/html"}
    )
    with patch.dict(sys.modules, {"readability": None}):
        with pytest.raises(ToolUnavailableError, match="Article extraction requires readability-lxml"):
            fetch_text("https://example.com", extract_article=True)
