"""Key-value store tools for Tsugite agents."""

from typing import Optional

from ..tools import tool


def _resolve_namespace(namespace: Optional[str]) -> str:
    from tsugite.agent_runner.helpers import resolve_current_agent
    return resolve_current_agent(namespace)


@tool
def kv_get(key: str, namespace: Optional[str] = None) -> str:
    """Get a value from the key-value store.

    Args:
        key: The key to look up
        namespace: Optional namespace (defaults to current agent name)

    Returns:
        The stored value, or a message if not found
    """
    from tsugite.kvstore import get_backend
    ns = _resolve_namespace(namespace)
    value = get_backend().get(ns, key)
    if value is None:
        return f"No value found for key '{key}' in namespace '{ns}'"
    return value


@tool
def kv_set(key: str, value: str, namespace: Optional[str] = None, ttl_seconds: Optional[int] = None) -> str:
    """Set a value in the key-value store.

    Args:
        key: The key to set
        value: The value to store (string)
        namespace: Optional namespace (defaults to current agent name)
        ttl_seconds: Optional time-to-live in seconds

    Returns:
        Confirmation message
    """
    from tsugite.kvstore import get_backend
    ns = _resolve_namespace(namespace)
    get_backend().set(ns, key, value, ttl_seconds)
    return f"Set '{key}' in namespace '{ns}'"


@tool
def kv_delete(key: str, namespace: Optional[str] = None) -> str:
    """Delete a key from the key-value store.

    Args:
        key: The key to delete
        namespace: Optional namespace (defaults to current agent name)

    Returns:
        Confirmation message
    """
    from tsugite.kvstore import get_backend
    ns = _resolve_namespace(namespace)
    deleted = get_backend().delete(ns, key)
    if deleted:
        return f"Deleted '{key}' from namespace '{ns}'"
    return f"Key '{key}' not found in namespace '{ns}'"


@tool
def kv_list(namespace: Optional[str] = None, prefix: str = "") -> str:
    """List keys in the key-value store.

    Args:
        namespace: Optional namespace (defaults to current agent name)
        prefix: Optional key prefix filter

    Returns:
        Newline-separated list of keys, or a message if empty
    """
    from tsugite.kvstore import get_backend
    ns = _resolve_namespace(namespace)
    keys = get_backend().list_keys(ns, prefix)
    if not keys:
        return f"No keys found in namespace '{ns}'" + (f" with prefix '{prefix}'" if prefix else "")
    return "\n".join(keys)
