"""HTTP client tools for Tsugite agents."""

import json
from typing import Dict, Any, Optional, Union
import httpx
from ddgs import DDGS
from tsugite.tools import tool


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
        with httpx.Client(timeout=timeout) as client:
            response = client.request(
                method=method.upper(),
                url=url,
                headers=headers or {},
            )
            response.raise_for_status()

            try:
                return response.json()
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON response: {e}")

    except httpx.TimeoutException:
        raise RuntimeError(f"Request timed out after {timeout} seconds")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"HTTP error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        raise RuntimeError(f"Request failed: {e}")


@tool
def fetch_text(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> str:
    """Fetch text content from a URL.

    Args:
        url: URL to fetch from
        method: HTTP method
        headers: Optional HTTP headers
        timeout: Request timeout in seconds

    Returns:
        Response text content

    Raises:
        RuntimeError: If request fails
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.request(
                method=method.upper(),
                url=url,
                headers=headers or {},
            )
            response.raise_for_status()
            return response.text

    except httpx.TimeoutException:
        raise RuntimeError(f"Request timed out after {timeout} seconds")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"HTTP error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        raise RuntimeError(f"Request failed: {e}")


@tool
def post_json(
    url: str,
    data: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> Union[Dict[str, Any], str]:
    """Send JSON data via POST request.

    Args:
        url: URL to post to
        data: JSON data to send
        headers: Optional HTTP headers
        timeout: Request timeout in seconds

    Returns:
        Response as JSON dict/list or text if not JSON

    Raises:
        RuntimeError: If request fails
    """
    try:
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)

        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                url=url,
                json=data,
                headers=request_headers,
            )
            response.raise_for_status()

            # Try to return JSON, fall back to text
            try:
                return response.json()
            except json.JSONDecodeError:
                return response.text

    except httpx.TimeoutException:
        raise RuntimeError(f"Request timed out after {timeout} seconds")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"HTTP error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        raise RuntimeError(f"Request failed: {e}")


@tool
def download_file(url: str, local_path: str, timeout: int = 60) -> str:
    """Download a file from URL to local path.

    Args:
        url: URL to download from
        local_path: Local file path to save to
        timeout: Request timeout in seconds

    Returns:
        Success message with file size

    Raises:
        RuntimeError: If download fails
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()

                total_size = 0
                with open(local_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)
                        total_size += len(chunk)

                return f"Downloaded {total_size} bytes to {local_path}"

    except httpx.TimeoutException:
        raise RuntimeError(f"Download timed out after {timeout} seconds")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"HTTP error {e.response.status_code}")
    except OSError as e:
        raise RuntimeError(f"File write error: {e}")
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")


@tool
def check_url(url: str, timeout: int = 10) -> Dict[str, Any]:
    """Check if a URL is accessible and return basic info.

    Args:
        url: URL to check
        timeout: Request timeout in seconds

    Returns:
        Dictionary with status info

    Raises:
        RuntimeError: If check fails
    """
    try:
        with httpx.Client(timeout=timeout) as client:
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
        raise RuntimeError(f"Web search failed: {e}")
