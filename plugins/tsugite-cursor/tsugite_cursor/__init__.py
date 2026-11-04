"""Tsugite plugin: Cursor provider (routes LLM calls through the Cursor Python SDK)."""

from tsugite_cursor.provider import CursorProvider, create_provider

__all__ = ["CursorProvider", "create_provider"]
