"""Environment variable secret backend."""

import os


class EnvSecretBackend:
    """Reads secrets from environment variables."""

    def __init__(self, prefix: str = ""):
        self._prefix = prefix

    def _normalize(self, name: str) -> str:
        return self._prefix + name.upper().replace("-", "_")

    def get(self, name: str) -> str | None:
        return os.environ.get(self._normalize(name)) or os.environ.get(name)

    def list_names(self) -> list[str]:
        return []

    def set(self, name: str, value: str) -> None:
        raise NotImplementedError("Cannot set secrets via environment variables")

    def delete(self, name: str) -> bool:
        raise NotImplementedError("Cannot delete secrets via environment variables")
