"""Example single-file plugin, enabled by a `path` entry in config.json.

See the "Single-file plugins" section of docs/plugins.md.
"""

import logging

from tsugite.events.bus import subscribe
from tsugite.hooks import hook
from tsugite.tools import tool
from tsugite.ui_surfaces import ui_surface

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


@ui_surface(kind="dashboard", label="Homelab", icon="plug", nav=True)
def dashboard_page() -> str:
    """Runs unauthenticated, so keep secrets out of it.

    The host shows the page only once it answers `tsugite:init` with `tsugite:ready`.
    """
    return f"""<!doctype html><meta charset=utf-8><title>Homelab</title>
<body style="font: 1rem system-ui; color: var(--tx0); background: var(--bg1)">
<h1>{dashboard_status()}</h1>
<script>
  addEventListener('message', (event) => {{
    const msg = event.data;
    if (msg?.type !== 'tsugite:init' && msg?.type !== 'tsugite:theme') return;
    for (const [name, value] of Object.entries(msg.theme.tokens)) {{
      document.documentElement.style.setProperty(name, value);
    }}
    if (msg.type === 'tsugite:init') {{
      parent.postMessage({{ type: 'tsugite:ready' }}, location.origin);
    }}
  }});
</script>
</body>"""
