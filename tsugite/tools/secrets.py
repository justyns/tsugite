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
    result = registry.register(name, value, agent=agent_name)

    from tsugite.events.helpers import emit_secret_access_event

    emit_secret_access_event(name)
    return result


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


def register_jinja_globals(env) -> None:
    """Register get_secret/list_secrets on a Jinja environment.

    The functions enforce the per-agent allowlist and route values through the
    masking registry, so exposing them as Jinja globals is safe.
    """
    env.globals["get_secret"] = get_secret
    env.globals["list_secrets"] = list_secrets
