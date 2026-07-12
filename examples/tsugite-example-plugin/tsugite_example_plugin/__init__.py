"""Example tsugite plugin - the agent-facing extension points in one module.

Two entry points in pyproject.toml wire the plugin up. Importing this module
(the first entry point) runs the @tool / @hook / @subscribe decorators; the
daemon-only adapter (HTTP routes + job executors) lives in adapter.py so this
half still loads on an install without the `[daemon]` extra:

    [project.entry-points."tsugite.plugins"]
    example = "tsugite_example_plugin"                         # tool + hook + subscriber

    [project.entry-points."tsugite.adapters"]
    example = "tsugite_example_plugin.adapter:create_adapter"  # HTTP routes + job executors

Enable the adapter half in daemon.yaml:

    plugins:
      example:
        enabled: true
        greeting: "hi"

Other, more specialized groups (each replaces a whole subsystem, so they get
their own file normally, not shown here): `tsugite.providers` (an LLM backend),
`tsugite.attachments` (a URL/file handler), `tsugite.sandbox` (an exec sandbox),
`tsugite.history` (a session-history backend), `tsugite.secrets` (a secret store),
and `tsugite.executors` (a code-execution backend - unrelated to the *job*
executors in adapter.py via `get_job_executors()`, despite the similar name).
"""

from __future__ import annotations

import logging

from tsugite.events.bus import subscribe
from tsugite.hooks import hook
from tsugite.tools import tool

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. AGENT TOOL - a Python function agents can call. The signature (with type
#    hints) and the docstring become the tool schema the model sees. Return any
#    JSON-serializable value. Flags: require_daemon / parent_only / interactive_only.
# ─────────────────────────────────────────────────────────────────────────────
@tool(category="example")
def greet(name: str, excited: bool = False) -> str:
    """Return a greeting for `name`. Set `excited` for an exclamation mark."""
    return f"Hello, {name}{'!' if excited else '.'}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. LIFECYCLE HOOK - a Python callable fired at an agent-loop phase. It receives
#    a context dict. Sync or async. Phases: pre_message, pre_context_build,
#    post_context_build, pre_llm_call, pre_tool_call, pre_response, post_response,
#    pre_compact, post_compact, session_end. (The tool phases pre_tool_call /
#    post_tool need tools=[...] to say which tools match; "*" = all.)
#    On the context-build phases a returned string is injected into the prompt; on
#    other phases the return is ignored (do side effects like logging/notifying).
# ─────────────────────────────────────────────────────────────────────────────
@hook("pre_tool_call", tools=["*"])
def log_tool_use(context: dict) -> None:
    """Fire before any tool call. Side effect only (returns are ignored here)."""
    logger.info("example plugin: about to run tool '%s'", context.get("tool"))


# ─────────────────────────────────────────────────────────────────────────────
# 3. EVENT SUBSCRIBER - react to events on the daemon's event bus. Distinct from
#    a hook: hooks fire at agent-loop phases (and can inject context); subscribers
#    just observe bus events. Filter by name or a predicate.
# ─────────────────────────────────────────────────────────────────────────────
@subscribe(event_name="tool_call")
def on_tool_call(event) -> None:
    """Observe every tool invocation on the bus."""
    logger.info("example plugin: bus tool_call %s", event.tool_name)
