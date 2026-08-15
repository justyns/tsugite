"""Example single-file plugin, enabled by a `path` entry in config.json.

See the "Single-file plugins" section of docs/plugins.md.
"""

import logging

from tsugite.events.bus import subscribe
from tsugite.hooks import hook
from tsugite.tools import tool

logger = logging.getLogger(__name__)


@tool(category="dashboard")
def dashboard_status(service: str = "all") -> str:
    """Report the status of a homelab service.

    Args:
        service: Service to report on, or "all".
    """
    return f"{service}: ok"


@hook("pre_tool_call", tools=["*"])
def log_tool_use(context: dict) -> None:
    logger.info("dashboard: running tool '%s'", context.get("tool"))


@subscribe(event_name="tool_call")
def on_tool_call(event) -> None:
    logger.info("dashboard: bus tool_call %s", event.tool_name)
