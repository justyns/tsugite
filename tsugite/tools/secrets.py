"""Secret management tools for Tsugite agents."""

from tsugite.tools import tool


@tool(parent_only=True)
def get_secret(name: str) -> str:
    """Get a secret by name. The value is automatically masked in all output.

    Args:
        name: Secret name (e.g., "forgejo-token", "OPENAI_API_KEY")

    Returns:
        The secret value (automatically scrubbed from LLM-visible output)
    """
    from tsugite.agent_runner.helpers import get_allowed_secrets, resolve_current_agent
    from tsugite.secrets import get_backend
    from tsugite.secrets.registry import get_registry

    agent_name = resolve_current_agent()

    # Check per-agent allowlist
    allowed = get_allowed_secrets()
    if allowed and name not in allowed:
        raise PermissionError(f"Agent '{agent_name}' is not allowed to access secret '{name}'. Allowed: {allowed}")

    backend = get_backend()
    value = backend.get(name)
    if value is None:
        raise RuntimeError(f"Secret '{name}' not found")

    registry = get_registry()
    return registry.register(name, value, agent=agent_name)


@tool(parent_only=True)
def list_secrets() -> str:
    """List available secret names (not values).

    Returns:
        Newline-separated list of secret names, or a message if none found
    """
    from tsugite.secrets import get_backend

    names = get_backend().list_names()
    if not names:
        return "No secrets found (or backend does not support listing)"
    return "\n".join(sorted(names))
