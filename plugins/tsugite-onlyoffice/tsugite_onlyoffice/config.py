"""Configuration model for the `plugins.onlyoffice` block of daemon.yaml."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class OnlyOfficeConfig(BaseModel):
    """The `plugins.onlyoffice` block.

    Attributes:
        enabled: Opt-in switch. The adapter factory returns None while this is false.
        server_url: Base URL of the ONLYOFFICE Docs server.
        jwt_secret_name: Name of the shared JWT secret in the secrets backend.
        public_base_url: Base URL the document server uses to reach this daemon. It
            fetches `document.url` and POSTs `callbackUrl` itself, so a bind address
            is not a usable substitute.
        documents_dir: Directory holding the editable documents. Every path arriving
            over HTTP or from a tool must resolve inside it.
        agent_name: The author on every comment the tools write. A browser session
            opens as the human viewing it, not as this.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    server_url: str
    jwt_secret_name: str
    public_base_url: str
    documents_dir: Path
    agent_name: str = "Tsugite"

    def resolve_jwt_secret(self) -> str:
        """Resolve the shared JWT secret from the secrets backend.

        Called after `configure_from_daemon()` has set the backend up.

        Returns:
            The secret value.

        Raises:
            RuntimeError: The secret is not present in the backend.
        """
        from tsugite.secrets import get_backend

        value = get_backend().get(self.jwt_secret_name)
        if value is None:
            raise RuntimeError(f"onlyoffice plugin: secret {self.jwt_secret_name!r} not found in secrets backend")
        return value
