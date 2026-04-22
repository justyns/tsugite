"""Per-session `state` dict persisted as JSON between agent turns.

The model interacts with a plain `dict` named `state`; the executor calls
`save_state` after each turn and `load_state` when constructing the
session. JSON (not pickle) is used so the serialization story matches
SubprocessExecutor, where unpickling attacker-controlled state would be
a sandbox escape.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional

from tsugite.exceptions import StateSerializationError

DEFAULT_MAX_BYTES_PER_KEY = 10 * 1024 * 1024
DEFAULT_MAX_BYTES_TOTAL = 10 * 1024 * 1024


def load_state(path: Path) -> dict[str, Any]:
    """Load a previously-saved state dict. Returns {} if the file doesn't exist."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_state(
    data: dict[str, Any],
    path: Path,
    *,
    session_id: Optional[str] = None,
    max_bytes_per_key: int = DEFAULT_MAX_BYTES_PER_KEY,
    max_bytes_total: int = DEFAULT_MAX_BYTES_TOTAL,
) -> None:
    """Atomically write `data` to `path` with per-key JSON probing and size caps.

    Raises StateSerializationError on a non-JSON value or a cap violation.
    """
    total = 0
    for key, value in data.items():
        try:
            blob = json.dumps(value, ensure_ascii=False)
        except TypeError as exc:
            raise StateSerializationError(
                f"state[{key!r}] is not JSON-serializable ({type(value).__name__}): {exc}",
                session_id=session_id,
                key=key,
                reason="not-json-serializable",
            ) from exc

        size = len(blob.encode("utf-8"))
        if size > max_bytes_per_key:
            raise StateSerializationError(
                f"state[{key!r}] is {size} bytes, exceeds per-key cap of {max_bytes_per_key}",
                session_id=session_id,
                key=key,
                reason="size-cap",
            )
        total += size

    if total > max_bytes_total:
        raise StateSerializationError(
            f"total state size is {total} bytes, exceeds cap of {max_bytes_total}",
            session_id=session_id,
            reason="size-cap",
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, sort_keys=True, ensure_ascii=False)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
