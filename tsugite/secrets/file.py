"""File-based secret backend. One file per secret in a directory."""

import os
from pathlib import Path


class FileSecretBackend:
    """Reads/writes secrets as plaintext files in a directory."""

    def __init__(self, config: dict):
        path = config.get("path", "secrets")
        self._dir = Path(path).resolve()

    def _safe_path(self, name: str) -> Path:
        path = (self._dir / name).resolve()
        if not path.is_relative_to(self._dir):
            raise ValueError(f"Invalid secret name: {name}")
        return path

    def get(self, name: str) -> str | None:
        path = self._safe_path(name)
        if not path.is_file():
            return None
        return path.read_text().strip()

    def list_names(self) -> list[str]:
        if not self._dir.is_dir():
            return []
        return sorted(f.name for f in self._dir.iterdir() if f.is_file() and not f.name.startswith("."))

    def set(self, name: str, value: str) -> None:
        path = self._safe_path(name)
        self._dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(value + "\n")

    def delete(self, name: str) -> bool:
        path = self._safe_path(name)
        if not path.is_file():
            return False
        path.unlink()
        return True
