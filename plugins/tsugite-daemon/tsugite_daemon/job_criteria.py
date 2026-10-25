"""Acceptance criteria are strings, or dicts that add timeouts, models and reviews."""

SEVERITIES = ("note", "minor", "major", "blocker")

MAX_PREDICATE_TIMEOUT_SECONDS = 1800

_KIND_KEYS = {
    "cmd": {"cmd", "timeout"},
    "exit_code": {"exit_code", "code", "timeout"},
    "file_exists": {"file_exists"},
    "check": {"check", "models"},
    "review": {"prompt", "agent", "models", "block_on"},
}

_STRING_KEYS = ("cmd", "exit_code", "file_exists", "check", "prompt", "agent")

_REVIEW_TEXT_LIMIT = 80


def dict_kind(ac: dict) -> str:
    for kind in ("cmd", "exit_code", "file_exists", "check"):
        if kind in ac:
            return kind
    if "prompt" in ac or "agent" in ac:
        return "review"
    raise ValueError(
        f"unknown acceptance criterion keys {sorted(ac)}; expected cmd, exit_code, file_exists, check, or prompt/agent"
    )


def _validate(ac: dict) -> dict:
    kind = dict_kind(ac)
    unknown = sorted(set(ac) - _KIND_KEYS[kind])
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} in {kind} criterion; allowed: {sorted(_KIND_KEYS[kind])}")
    out: dict = {}
    for key, value in ac.items():
        if key in _STRING_KEYS:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"criterion key '{key}' must be a non-empty string")
            out[key] = value.strip()
        elif key == "timeout":
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 < value <= MAX_PREDICATE_TIMEOUT_SECONDS
            ):
                raise ValueError(
                    f"criterion key 'timeout' must be a positive number of seconds, at most {MAX_PREDICATE_TIMEOUT_SECONDS}"
                )
            out[key] = value
        elif key == "code":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("criterion key 'code' must be an integer")
            out[key] = value
        elif key == "models":
            if not isinstance(value, list) or not value or not all(isinstance(m, str) and m.strip() for m in value):
                raise ValueError("criterion key 'models' must be a non-empty list of model names")
            out[key] = list(dict.fromkeys(m.strip() for m in value))
        elif key == "block_on":
            if value not in SEVERITIES:
                raise ValueError(f"criterion key 'block_on' must be one of {list(SEVERITIES)}")
            out[key] = value
    if kind == "review" and not ({"prompt", "agent"} & set(out)):
        raise ValueError("a review criterion needs 'prompt' or 'agent'")
    return out


def normalize_criteria(items) -> list:
    """Strip strings, validate dicts, drop empties. A legacy `{text, kind}` dict collapses to its text."""
    out: list = []
    for item in items or []:
        if isinstance(item, dict) and "text" in item and set(item) <= {"text", "kind"}:
            item = item["text"]
        if isinstance(item, dict):
            out.append(_validate(item))
        elif text := str(item).strip():
            out.append(text)
    return out


def ac_text(ac) -> str:
    if isinstance(ac, str):
        return ac
    kind = dict_kind(ac)
    if kind == "cmd":
        return f"cmd:{ac['cmd']}"
    if kind == "exit_code":
        return f"exit_code:{ac['exit_code']}" + (f":{ac['code']}" if "code" in ac else "")
    if kind == "file_exists":
        return f"file_exists:{ac['file_exists']}"
    if kind == "check":
        return ac["check"]
    return f"review: {clip(ac.get('prompt') or ac['agent'], _REVIEW_TEXT_LIMIT)}"


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"
