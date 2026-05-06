"""Time helpers for Tsugite agents."""

from datetime import datetime

from ..renderer import local_tz
from ..tools import tool


@tool
def now() -> str:
    """Return the current datetime as an ISO 8601 string with timezone.

    Saves agents the boilerplate of importing datetime/zoneinfo for a one-shot
    "what time is it" check.
    """
    return datetime.now(tz=local_tz()).isoformat()
