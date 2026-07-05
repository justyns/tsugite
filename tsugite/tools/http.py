"""HTTP client tools for Tsugite agents."""

import json
from typing import Any, Dict, Optional, Union

import httpx

from tsugite.tools import tool
from tsugite.user_agent import set_user_agent_header
from tsugite.utils import convert_html_to_markdown

_WEB_EXTRA_HINT = "Install it with: pip install tsugite-cli[web]"

# Redirect hop bound for follow_redirects=True. Sandbox network policy applies
# at the process level (bwrap network namespace), so followed redirects can't
# reach anything the original request couldn't.
_MAX_REDIRECTS = 5


def _extract_article(html: str) -> str:
    """Extract the main article body from HTML using readability (optional dep)."""
    try:
        from readability import Document
    except ImportError as e:
        raise RuntimeError(f"Article extraction requires readability-lxml. {_WEB_EXTRA_HINT}") from e
    return Document(html).summary()


def _default_headers(headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return headers with User-Agent set if not already provided."""
    merged = dict(headers) if headers else {}
    set_user_agent_header(merged)
    return merged


class HttpResponse:
    """Structured HTTP response. Use `.text` for the raw body; call `.json()` to parse as JSON.
    `.url` is the final URL after any followed redirects."""

    def __init__(self, status_code: int, headers: Dict[str, str], text: str, url: Optional[str] = None):
        self.status_code = status_code
        self.headers = headers
        self.text = text
        self.url = url

    @property
    def body(self) -> str:
        """Alias for `.text` (raw response body)."""
        return self.text

    def json(self) -> Union[Dict[str, Any], list]:
        """Parse the response body as JSON. Raises ValueError if the body is not valid JSON."""
        try:
            return json.loads(self.text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Response body is not valid JSON: {e}") from e

    def __repr__(self) -> str:
        return f"HttpResponse(status_code={self.status_code}, headers={self.headers!r}, text={self.text!r})"


def _simple_request(
    url: str,
    method: str,
    headers: Optional[Dict[str, str]],
    timeout: int,
    body: Optional[Union[str, Dict[str, Any]]] = None,
    follow_redirects: bool = True,
) -> httpx.Response:
    """Make an HTTP request and return the raw response.

    Redirects are followed by default (bounded by _MAX_REDIRECTS; httpx keeps
    method+body on 307/308). With follow_redirects=False a 3xx response is
    returned for inspection instead of raising - that's the point of opting out.
    """
    kwargs: Dict[str, Any] = {}
    if isinstance(body, dict):
        kwargs["json"] = body
    elif isinstance(body, str):
        kwargs["content"] = body

    with httpx.Client(timeout=timeout, follow_redirects=follow_redirects, max_redirects=_MAX_REDIRECTS) as client:
        response = client.request(method=method.upper(), url=url, headers=_default_headers(headers), **kwargs)
        if not follow_redirects and response.is_redirect:
            return response
        response.raise_for_status()
        return response


@tool
def fetch_json(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
    follow_redirects: bool = True,
) -> Union[Dict[str, Any], list]:
    """Fetch JSON data from a URL.

    Args:
        url: URL to fetch from
        method: HTTP method (GET, POST, PUT, DELETE)
        headers: Optional HTTP headers
        timeout: Request timeout in seconds
        follow_redirects: Follow 3xx redirects (default True, max 5 hops)
    """
    try:
        response = _simple_request(url, method, headers, timeout, follow_redirects=follow_redirects)
        try:
            return response.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response: {e}") from e
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"Request timed out after {timeout} seconds") from exc
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"HTTP error {e.response.status_code}: {e.response.text}") from e
    except Exception as e:
        raise RuntimeError(f"Request failed: {e}") from e


@tool
def fetch_text(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
    strip_html: bool = True,
    extract_article: bool = False,
    follow_redirects: bool = True,
) -> str:
    """Fetch text content from a URL.

    Args:
        url: URL to fetch from
        method: HTTP method
        headers: Optional HTTP headers
        timeout: Request timeout in seconds
        strip_html: Convert HTML to markdown (preserves headings, lists, links)
        extract_article: Extract article content only (strips nav/ads/boilerplate), implies strip_html
        follow_redirects: Follow 3xx redirects (default True, max 5 hops)
    """
    try:
        response = _simple_request(url, method, headers, timeout, follow_redirects=follow_redirects)
        text = response.text
        content_type = response.headers.get("content-type", "")

        if "text/html" not in content_type or (not strip_html and not extract_article):
            return text

        if extract_article:
            text = _extract_article(text)
        return convert_html_to_markdown(text)
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"Request timed out after {timeout} seconds") from exc
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"HTTP error {e.response.status_code}: {e.response.text}") from e
    except Exception as e:
        raise RuntimeError(f"Request failed: {e}") from e


@tool
def http_request(
    url: str,
    method: str = "GET",
    body: Optional[Union[str, Dict[str, Any]]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
    follow_redirects: bool = True,
) -> HttpResponse:
    """Make an HTTP request. Returns HttpResponse with .status_code, .headers, .text (raw body); call .json() to parse JSON.

    Args:
        url: URL to send the request to
        method: HTTP method (GET, POST, PUT, PATCH, DELETE) - defaults to GET (safe, idempotent)
        body: Request body - dict for JSON (auto-serialized), string for raw body
        headers: Optional HTTP headers
        timeout: Request timeout in seconds
        follow_redirects: Follow 3xx redirects (default True, max 5 hops); False returns the 3xx response for inspection
    """
    try:
        request_headers = dict(headers) if headers else {}
        response = _simple_request(url, method, request_headers, timeout, body, follow_redirects=follow_redirects)

        return HttpResponse(
            status_code=response.status_code,
            headers=dict(response.headers),
            text=response.text,
            url=str(response.url),
        )
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"Request timed out after {timeout} seconds") from exc
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"HTTP error {e.response.status_code}: {e.response.text}") from e
    except Exception as e:
        raise RuntimeError(f"Request failed: {e}") from e


@tool
def download_file(url: str, local_path: str, timeout: int = 60, follow_redirects: bool = True) -> str:
    """Download a file from URL to local path.

    Args:
        url: URL to download from
        local_path: Local file path to save to
        timeout: Request timeout in seconds
        follow_redirects: Follow 3xx redirects (default True, max 5 hops)
    """
    try:
        with httpx.Client(
            timeout=timeout,
            headers=_default_headers(),
            follow_redirects=follow_redirects,
            max_redirects=_MAX_REDIRECTS,
        ) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()

                total_size = 0
                with open(local_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)
                        total_size += len(chunk)

                return f"Downloaded {total_size} bytes to {local_path}"
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"Request timed out after {timeout} seconds") from exc
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"HTTP error {e.response.status_code}") from e
    except OSError as e:
        raise RuntimeError(f"File write error: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}") from e


@tool
def check_url(url: str, timeout: int = 10) -> Dict[str, Any]:
    """Check if a URL is accessible and return basic info.

    Args:
        url: URL to check
        timeout: Request timeout in seconds
    """
    try:
        with httpx.Client(timeout=timeout, headers=_default_headers()) as client:
            response = client.head(url, follow_redirects=True)

            return {
                "url": str(response.url),
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "accessible": response.status_code < 400,
                "content_type": response.headers.get("content-type", "unknown"),
                "content_length": response.headers.get("content-length", "unknown"),
            }

    except Exception as e:
        return {
            "url": url,
            "accessible": False,
            "error": str(e),
        }
