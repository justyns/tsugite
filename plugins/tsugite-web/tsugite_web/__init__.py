"""Tsugite plugin: web search via DuckDuckGo.

Registers the ``web_search`` tool. The readability dependency it ships also enables
core ``fetch_text(extract_article=True)``.
"""

from typing import Dict

from tsugite.tools import tool

_WEB_EXTRA_HINT = "Install it with: pip install tsugite-cli[web]"


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
        from ddgs import DDGS
    except ImportError as e:
        raise RuntimeError(f"Web search requires the ddgs package. {_WEB_EXTRA_HINT}") from e

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
