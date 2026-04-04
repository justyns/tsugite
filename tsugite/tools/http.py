"""HTTP client tools for Tsugite agents."""

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import httpx
from ddgs import DDGS

from tsugite.user_agent import set_user_agent_header
from tsugite.tools import tool
from tsugite.utils import convert_html_to_markdown


def _default_headers(headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return headers with User-Agent set if not already provided."""
    merged = dict(headers) if headers else {}
    set_user_agent_header(merged)
    return merged


@dataclass
class HttpResponse:
    """Structured HTTP response."""

    status_code: int
    headers: Dict[str, str]
    body: Union[Dict[str, Any], list, str]


def _simple_request(
    url: str,
    method: str,
    headers: Optional[Dict[str, str]],
    timeout: int,
    body: Optional[Union[str, Dict[str, Any]]] = None,
) -> httpx.Response:
    """Make an HTTP request and return the raw response."""
    kwargs: Dict[str, Any] = {}
    if isinstance(body, dict):
        kwargs["json"] = body
    elif isinstance(body, str):
        kwargs["content"] = body

    with httpx.Client(timeout=timeout) as client:
        response = client.request(method=method.upper(), url=url, headers=_default_headers(headers), **kwargs)
        response.raise_for_status()
        return response


@tool
def fetch_json(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> Union[Dict[str, Any], list]:
    """Fetch JSON data from a URL.

    Args:
        url: URL to fetch from
        method: HTTP method (GET, POST, PUT, DELETE)
        headers: Optional HTTP headers
        timeout: Request timeout in seconds
    """
    try:
        response = _simple_request(url, method, headers, timeout)
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
) -> str:
    """Fetch text content from a URL.

    Args:
        url: URL to fetch from
        method: HTTP method
        headers: Optional HTTP headers
        timeout: Request timeout in seconds
        strip_html: Convert HTML to markdown (preserves headings, lists, links)
        extract_article: Extract article content only (strips nav/ads/boilerplate), implies strip_html
    """
    try:
        response = _simple_request(url, method, headers, timeout)
        text = response.text
        content_type = response.headers.get("content-type", "")

        if "text/html" not in content_type or (not strip_html and not extract_article):
            return text

        if extract_article:
            from readability import Document

            text = Document(text).summary()
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
    method: str = "POST",
    body: Optional[Union[str, Dict[str, Any]]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> HttpResponse:
    """Make an HTTP request with optional body. Use for POST/PUT/PATCH/DELETE or verbose GET.

    Args:
        url: URL to send the request to
        method: HTTP method (POST, PUT, PATCH, DELETE, GET)
        body: Request body — dict for JSON (auto-serialized), string for raw body
        headers: Optional HTTP headers
        timeout: Request timeout in seconds
    """
    try:
        request_headers = dict(headers) if headers else {}
        response = _simple_request(url, method, request_headers, timeout, body)

        try:
            parsed_body = response.json()
        except (json.JSONDecodeError, ValueError):
            parsed_body = response.text

        return HttpResponse(
            status_code=response.status_code,
            headers=dict(response.headers),
            body=parsed_body,
        )
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"Request timed out after {timeout} seconds") from exc
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"HTTP error {e.response.status_code}: {e.response.text}") from e
    except Exception as e:
        raise RuntimeError(f"Request failed: {e}") from e


@tool
def download_file(url: str, local_path: str, timeout: int = 60) -> str:
    """Download a file from URL to local path.

    Args:
        url: URL to download from
        local_path: Local file path to save to
        timeout: Request timeout in seconds
    """
    try:
        with httpx.Client(timeout=timeout, headers=_default_headers()) as client:
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


@tool
def web_search(query: str, max_results: int = 5) -> list[Dict[str, str]]:
    """Search the web using DuckDuckGo and return results.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)

    Returns:
        List of search result dictionaries with title, url, and snippet

    Raises:
        RuntimeError: If search fails
    """
    try:
        results = []
        with DDGS() as ddgs:
            search_results = ddgs.text(query, max_results=max_results)
            for result in search_results:
                results.append(
                    {
                        "title": result.get("title", ""),
                        "url": result.get("href", ""),
                        "snippet": result.get("body", ""),
                    }
                )

        return results

    except Exception as e:
        raise RuntimeError(f"Web search failed: {e}") from e
