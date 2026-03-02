"""Workspace hooks — fire shell commands after tool calls and lifecycle events."""

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import jinja2
import yaml
from pydantic import BaseModel, Field

from tsugite.events.base import BaseEvent
from tsugite.events.events import ToolCallEvent, ToolResultEvent

if TYPE_CHECKING:
    from tsugite.events.bus import EventBus

logger = logging.getLogger(__name__)

FALSY_STRINGS = {"false", "none", "0", ""}
HOOK_TIMEOUT = 300
_jinja_env = jinja2.Environment()


class HookRule(BaseModel):
    """A hook rule — used for post_tool, pre_compact, post_compact, and pre_message hooks."""

    tools: list[str] = Field(default_factory=list)
    match: Optional[str] = None
    run: str
    wait: bool = False
    capture_as: Optional[str] = None


class HooksConfig(BaseModel):
    """Top-level hooks configuration."""

    post_tool: list[HookRule] = Field(default_factory=list)
    pre_compact: list[HookRule] = Field(default_factory=list)
    post_compact: list[HookRule] = Field(default_factory=list)
    pre_message: list[HookRule] = Field(default_factory=list)


def load_hooks_config(workspace_dir: Path) -> Optional[HooksConfig]:
    """Load hooks config from .tsugite/hooks.yaml. Returns None if missing."""
    hooks_path = workspace_dir / ".tsugite" / "hooks.yaml"
    if not hooks_path.exists():
        return None

    with open(hooks_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data or "hooks" not in data:
        return None

    return HooksConfig.model_validate(data["hooks"])


def _match_passes(jinja_env: jinja2.Environment, match_expr: str, context: dict[str, Any]) -> bool:
    """Evaluate a Jinja2 match expression against context."""
    try:
        result = jinja_env.from_string(match_expr).render(context).strip().lower()
        return result not in FALSY_STRINGS
    except Exception as e:
        logger.warning("Hook match eval failed: %s", e)
        return False


def _execute_hook(cmd: str, cwd: Path, wait: bool, capture: bool = False) -> Optional[str]:
    """Run a shell command. Returns stdout if capture=True."""
    try:
        logger.info("Hook fired: %s", cmd)
        if wait or capture:
            result = subprocess.run(cmd, shell=True, cwd=str(cwd), capture_output=True, timeout=HOOK_TIMEOUT)
            if result.returncode != 0:
                logger.warning("Hook failed (exit %d): %s", result.returncode, cmd)
                return None
            if capture:
                return result.stdout.decode().strip()
        else:
            subprocess.Popen(cmd, shell=True, cwd=str(cwd), stdout=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        logger.warning("Hook timed out after %ds: %s", HOOK_TIMEOUT, cmd)
    except Exception as e:
        logger.warning("Hook error: %s", e)
    return None


def _render_and_execute(
    jinja_env: jinja2.Environment,
    rules: list[HookRule],
    context: dict[str, Any],
    cwd: Path,
) -> dict[str, str]:
    """Render and execute matching hook rules synchronously.

    Returns:
        Dict mapping capture_as names to stdout content. Empty if no captures.
    """
    captured: dict[str, str] = {}
    for rule in rules:
        if rule.match and not _match_passes(jinja_env, rule.match, context):
            continue
        try:
            cmd = jinja_env.from_string(rule.run).render(context)
        except Exception as e:
            logger.warning("Hook render failed: %s", e)
            continue
        output = _execute_hook(cmd, cwd, rule.wait, capture=bool(rule.capture_as))
        if rule.capture_as and output is not None:
            captured[rule.capture_as] = output
    return captured


class HookHandler:
    """Event handler that fires shell commands after successful tool calls."""

    def __init__(self, config: HooksConfig, workspace_dir: Path):
        self.config = config
        self.workspace_dir = workspace_dir
        self._pending: Optional[tuple[str, dict[str, Any]]] = None

    def handle_event(self, event: BaseEvent) -> None:
        if isinstance(event, ToolCallEvent):
            self._pending = (event.tool_name, event.arguments)
        elif isinstance(event, ToolResultEvent):
            if event.success and self._pending:
                self._fire_hooks(*self._pending)
            self._pending = None

    def _fire_hooks(self, tool_name: str, arguments: dict[str, Any]) -> None:
        context: dict[str, Any] = {"tool": tool_name, **arguments}

        if "path" in arguments:
            try:
                context["path"] = str(Path(arguments["path"]).resolve().relative_to(self.workspace_dir.resolve()))
            except ValueError:
                pass

        tool_rules = [
            rule for rule in self.config.post_tool
            if tool_name in rule.tools or "*" in rule.tools
        ]
        _render_and_execute(_jinja_env, tool_rules, context, self.workspace_dir)


def setup_hook_handler(workspace_dir: Path, event_bus: "EventBus") -> None:
    """Load hooks config and subscribe handler to event_bus if rules exist."""
    config = load_hooks_config(workspace_dir)
    if config and config.post_tool:
        event_bus.subscribe(HookHandler(config, workspace_dir).handle_event)


async def _fire_hooks(
    workspace_dir: Path,
    phase: str,
    context: dict[str, Any],
) -> dict[str, str]:
    """Load config and fire hooks for the given phase.

    Returns:
        Dict mapping capture_as names to stdout content. Empty if no captures or no config.
    """
    config = load_hooks_config(workspace_dir)
    if not config:
        return {}

    rules = getattr(config, phase)
    if not rules:
        return {}

    return await asyncio.to_thread(_render_and_execute, _jinja_env, rules, context, workspace_dir)


async def fire_compact_hooks(
    workspace_dir: Path,
    phase: Literal["pre_compact", "post_compact"],
    context: dict[str, Any],
) -> None:
    """Fire pre_compact or post_compact hooks."""
    await _fire_hooks(workspace_dir, phase, context)


async def fire_pre_message_hooks(
    workspace_dir: Path, context: dict[str, Any]
) -> dict[str, str]:
    """Fire pre_message hooks, returning captured variables."""
    return await _fire_hooks(workspace_dir, "pre_message", context)
