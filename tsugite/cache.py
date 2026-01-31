"""Cache management for attachment content.

This module provides content-addressable storage for attachments.
Content is stored by SHA256 hash, allowing deduplication and
efficient storage of session context.
"""

import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tsugite.config import get_xdg_cache_path


def get_cache_key(source: str) -> str:
    """Generate cache key from source URL or path.

    Args:
        source: URL or file path to cache

    Returns:
        16-character hex string (first 16 chars of SHA256)
    """
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def get_cache_file_path(source: str) -> Path:
    """Get cache file path for a source.

    Args:
        source: URL or file path

    Returns:
        Path to cache file
    """
    cache_dir = get_xdg_cache_path("attachments")
    cache_key = get_cache_key(source)
    return cache_dir / f"{cache_key}.txt"


def get_cached_content(source: str) -> Optional[str]:
    """Get cached content if exists.

    Args:
        source: URL or file path

    Returns:
        Cached content if exists, None otherwise
    """
    cache_file = get_cache_file_path(source)
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    return None


def save_to_cache(source: str, content: str) -> None:
    """Save content to cache.

    Args:
        source: URL or file path
        content: Content to cache

    Raises:
        RuntimeError: If cache save fails
    """
    cache_file = get_cache_file_path(source)

    # Ensure cache directory exists
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        cache_file.write_text(content, encoding="utf-8")

        # Update metadata
        _update_cache_metadata(source, cache_file)
    except IOError as e:
        raise RuntimeError(f"Failed to save cache for {source}: {e}") from e


def _update_cache_metadata(source: str, cache_file: Path) -> None:
    """Update cache metadata file.

    Args:
        source: Original source URL/path
        cache_file: Path to cache file
    """
    metadata_file = get_xdg_cache_path("attachments") / "metadata.json"

    # Load existing metadata
    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, IOError):
            metadata = {}
    else:
        metadata = {}

    # Update entry
    cache_key = get_cache_key(source)
    metadata[cache_key] = {
        "source": source,
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "size": cache_file.stat().st_size,
    }

    # Save metadata
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except IOError:
        # Metadata update failure is not critical
        pass


def clear_cache(source: Optional[str] = None) -> int:
    """Clear cache for source, or entire cache if source is None.

    Args:
        source: URL or file path to clear, or None to clear all

    Returns:
        Number of cache files deleted
    """
    cache_dir = get_xdg_cache_path("attachments")

    if not cache_dir.exists():
        return 0

    count = 0

    if source:
        # Clear specific cache entry
        cache_file = get_cache_file_path(source)
        if cache_file.exists():
            cache_file.unlink()
            count = 1

        # Update metadata
        metadata_file = cache_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                cache_key = get_cache_key(source)
                if cache_key in metadata:
                    del metadata[cache_key]

                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
            except (json.JSONDecodeError, IOError):
                pass
    else:
        # Clear all cache files
        for cache_file in cache_dir.glob("*.txt"):
            cache_file.unlink()
            count += 1

        # Clear metadata
        metadata_file = cache_dir / "metadata.json"
        if metadata_file.exists():
            metadata_file.unlink()

    return count


def list_cache() -> Dict[str, Dict[str, any]]:
    """List all cached entries with metadata.

    Returns:
        Dictionary mapping cache keys to metadata
    """
    metadata_file = get_xdg_cache_path("attachments") / "metadata.json"

    if not metadata_file.exists():
        return {}

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def get_cache_info(source: str) -> Optional[Dict[str, any]]:
    """Get cache metadata for a specific source.

    Args:
        source: URL or file path

    Returns:
        Metadata dict if cached, None otherwise
    """
    metadata = list_cache()
    cache_key = get_cache_key(source)
    return metadata.get(cache_key)


# Session Storage V2 functions


def get_content_hash(content: Union[str, bytes]) -> str:
    """Compute SHA256 hash of content.

    Args:
        content: String or bytes content

    Returns:
        Full SHA256 hex digest
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def get_binary_cache_path(content_hash: str) -> Path:
    """Get cache file path for binary content by hash.

    Args:
        content_hash: SHA256 hash of content

    Returns:
        Path to cache file
    """
    cache_dir = get_xdg_cache_path("attachments")
    return cache_dir / f"{content_hash}.bin"


def store_content(content: Union[str, bytes], is_binary: bool = False) -> str:
    """Store content by hash, returns hash key.

    Args:
        content: Content to store (string or bytes)
        is_binary: If True, treat as binary and base64 encode for storage

    Returns:
        SHA256 hash of content (use as cache key)
    """
    content_hash = get_content_hash(content)

    if is_binary:
        cache_file = get_binary_cache_path(content_hash)
        if cache_file.exists():
            return content_hash

        cache_file.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(content, str):
            content = content.encode("utf-8")

        cache_file.write_bytes(content)
    else:
        cache_file = get_xdg_cache_path("attachments") / f"{content_hash}.txt"
        if cache_file.exists():
            return content_hash

        cache_file.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(content, bytes):
            content = content.decode("utf-8")

        cache_file.write_text(content, encoding="utf-8")

    _update_hash_metadata(content_hash, cache_file)
    return content_hash


def get_content_by_hash(content_hash: str, is_binary: bool = False) -> Optional[Union[str, bytes]]:
    """Retrieve content by hash.

    Args:
        content_hash: SHA256 hash of content
        is_binary: If True, return bytes

    Returns:
        Content if found, None otherwise
    """
    if is_binary:
        cache_file = get_binary_cache_path(content_hash)
        if cache_file.exists():
            return cache_file.read_bytes()
    else:
        cache_file = get_xdg_cache_path("attachments") / f"{content_hash}.txt"
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")

    return None


def get_content_by_hash_as_base64(content_hash: str) -> Optional[str]:
    """Retrieve binary content by hash as base64 string.

    Useful for reconstructing image data URIs.

    Args:
        content_hash: SHA256 hash of content

    Returns:
        Base64-encoded content if found, None otherwise
    """
    content = get_content_by_hash(content_hash, is_binary=True)
    if content is None:
        return None
    return base64.b64encode(content).decode("ascii")


def _update_hash_metadata(content_hash: str, cache_file: Path) -> None:
    """Update cache metadata for hash-based storage."""
    metadata_file = get_xdg_cache_path("attachments") / "metadata.json"

    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, IOError):
            metadata = {}
    else:
        metadata = {}

    metadata[content_hash] = {
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "size": cache_file.stat().st_size,
    }

    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except IOError:
        pass


def compute_context_hash(attachments: Dict[str, Any], skills: List[str]) -> str:
    """Compute hash of context state for change detection.

    Creates a deterministic hash from attachment references and skill list.

    Args:
        attachments: Dict of attachment name -> AttachmentRef (or dict representation)
        skills: List of skill names

    Returns:
        SHA256 hash of context state
    """
    # Build deterministic representation
    att_items = []
    for name in sorted(attachments.keys()):
        ref = attachments[name]
        if hasattr(ref, "model_dump"):
            ref = ref.model_dump(exclude_none=True)
        att_items.append((name, json.dumps(ref, sort_keys=True)))

    skills_sorted = sorted(skills)

    context_str = json.dumps({"attachments": att_items, "skills": skills_sorted}, sort_keys=True)
    return hashlib.sha256(context_str.encode()).hexdigest()[:16]
