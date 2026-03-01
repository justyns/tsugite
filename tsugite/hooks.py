"""Workspace hooks — fire shell commands after successful tool calls."""

import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import jinja2
import yaml
from pydantic import BaseModel, Field

from tsugite.events.base import BaseEvent
from tsugite.events.events import ToolCallEvent, ToolResultEvent

if TYPE_CHECKING:
    from tsugite.events.bus import EventBus

logger = logging.getLogger(__name__)

FALSY_STRINGS = {"false", "none", "0", ""}


class HookRule(BaseModel):
    """A single post-tool hook rule."""

    tools: list[str]
    match: Optional[str] = None
    run: str


class HooksConfig(BaseModel):
    """Top-level hooks configuration."""

    post_tool: list[HookRule] = Field(default_factory=list)


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


class HookHandler:
    """Event handler that fires shell commands after successful tool calls."""

    def __init__(self, config: HooksConfig, workspace_dir: Path):
        self.config = config
        self.workspace_dir = workspace_dir
        self._jinja_env = jinja2.Environment()
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

        for rule in self.config.post_tool:
            if tool_name not in rule.tools and "*" not in rule.tools:
                continue

            if rule.match and not self._match_passes(rule.match, context):
                continue

            try:
                cmd = self._jinja_env.from_string(rule.run).render(context)
                logger.info("Hook fired: %s", cmd)
                subprocess.Popen(
                    cmd,
                    shell=True,
                    cwd=str(self.workspace_dir),
                    stdout=subprocess.DEVNULL,
                )
            except Exception as e:
                logger.warning("Hook failed for %s: %s", tool_name, e)

    def _match_passes(self, match_expr: str, context: dict[str, Any]) -> bool:
        try:
            result = self._jinja_env.from_string(match_expr).render(context).strip().lower()
            return result not in FALSY_STRINGS
        except Exception as e:
            logger.warning("Hook match eval failed: %s", e)
            return False


def setup_hook_handler(workspace_dir: Path, event_bus: "EventBus") -> None:
    """Load hooks config and subscribe handler to event_bus if rules exist."""
    config = load_hooks_config(workspace_dir)
    if config and config.post_tool:
        event_bus.subscribe(HookHandler(config, workspace_dir).handle_event)
