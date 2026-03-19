"""Workspace hooks — fire shell commands after tool calls and lifecycle events."""

import asyncio
import logging
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, NamedTuple, Optional, Union

import jinja2
import yaml
from pydantic import BaseModel, Field

from tsugite.events.base import BaseEvent
from tsugite.events.events import ToolCallEvent, ToolResultEvent
from tsugite.history.models import HookExecution

if TYPE_CHECKING:
    from tsugite.events.bus import EventBus

logger = logging.getLogger(__name__)

FALSY_STRINGS = {"false", "none", "0", ""}
HOOK_TIMEOUT = 300
_jinja_env = jinja2.Environment()
_jinja_env.filters["shell_quote"] = shlex.quote


class HookRule(BaseModel):
    """A hook rule — used for post_tool, pre_compact, post_compact, and pre_message hooks."""

    tools: list[str] = Field(default_factory=list)
    match: Optional[str] = None
    run: Union[str, list[str]]
    wait: bool = False
    capture_as: Optional[str] = None
    only_interactive: bool = False
    name: Optional[str] = None
    env: dict[str, str] = Field(default_factory=dict)


def _build_hook_env(
    context: dict[str, Any],
    custom_env: dict[str, str] | None = None,
    phase: str = "",
    workspace_dir: str = "",
) -> dict[str, str]:
    """Build environment dict for hook subprocess: parent env + TSUGITE_* + custom."""
    env = os.environ.copy()
    if phase:
        env["TSUGITE_HOOK_PHASE"] = phase
    if workspace_dir:
        env["TSUGITE_WORKSPACE_DIR"] = workspace_dir
    for key, value in context.items():
        if isinstance(value, (str, int, float, bool)):
            env[f"TSUGITE_{key.upper()}"] = str(value)
    if custom_env:
        env.update(custom_env)
    return env


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


class HookResult(NamedTuple):
    """Result from executing a single hook."""

    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int


def _execute_hook(
    cmd: Union[str, list[str]],
    cwd: Path,
    env: dict[str, str] | None = None,
) -> HookResult:
    """Run a hook command. Returns structured result."""
    start = time.monotonic()
    try:
        logger.info("Hook fired: %s", cmd)
        result = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=HOOK_TIMEOUT,
            env=env,
        )
        elapsed = int((time.monotonic() - start) * 1000)
        if result.returncode != 0:
            logger.warning("Hook failed (exit %d): %s", result.returncode, cmd)
            if result.stdout.strip():
                logger.warning("Hook stdout: %s", result.stdout.strip())
            if result.stderr.strip():
                logger.warning("Hook stderr: %s", result.stderr.strip())
        return HookResult(
            exit_code=result.returncode,
            stdout=result.stdout.strip(),
            stderr=result.stderr.strip(),
            duration_ms=elapsed,
        )
    except subprocess.TimeoutExpired:
        elapsed = int((time.monotonic() - start) * 1000)
        logger.warning("Hook timed out after %ds: %s", HOOK_TIMEOUT, cmd)
        return HookResult(exit_code=-1, stdout="", stderr=f"Timed out after {HOOK_TIMEOUT}s", duration_ms=elapsed)
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        logger.warning("Hook error: %s", e)
        return HookResult(exit_code=-1, stdout="", stderr=str(e), duration_ms=elapsed)


class HookResults(NamedTuple):
    """Combined results from executing hook rules."""

    captured: dict[str, str]
    executions: list[HookExecution]


def _render_and_execute(
    jinja_env: jinja2.Environment,
    rules: list[HookRule],
    context: dict[str, Any],
    cwd: Path,
    interactive: bool = True,
    on_status: Optional[Callable[[str], None]] = None,
    phase: str = "",
) -> HookResults:
    """Render and execute matching hook rules synchronously.

    Returns:
        HookResults with captured variables and execution records.
    """
    base_env = _build_hook_env(context, phase=phase, workspace_dir=str(cwd))
    captured: dict[str, str] = {}
    executions: list[HookExecution] = []
    for rule in rules:
        if rule.only_interactive and not interactive:
            continue
        if rule.match and not _match_passes(jinja_env, rule.match, context):
            continue
        if on_status:
            label = rule.name or rule.capture_as or "hook"
            on_status(f"Running {label}...")
        try:
            if isinstance(rule.run, list):
                cmd: Union[str, list[str]] = [jinja_env.from_string(part).render(context) for part in rule.run]
            else:
                cmd = jinja_env.from_string(rule.run).render(context)
        except Exception as e:
            logger.warning("Hook render failed: %s", e)
            continue

        if rule.env:
            env = base_env.copy()
            for env_key, env_val in rule.env.items():
                try:
                    env[env_key] = jinja_env.from_string(env_val).render(context)
                except Exception as e:
                    logger.warning("Hook env render failed for %s: %s", env_key, e)
        else:
            env = base_env

        hook_result = _execute_hook(cmd, cwd, env=env)

        if rule.capture_as and hook_result.exit_code == 0 and hook_result.stdout:
            captured[rule.capture_as] = hook_result.stdout

        cmd_str = cmd if isinstance(cmd, str) else shlex.join(cmd)
        executions.append(
            HookExecution(
                phase=phase,
                name=rule.name or rule.capture_as,
                command=cmd_str,
                exit_code=hook_result.exit_code,
                stdout=hook_result.stdout or None,
                stderr=hook_result.stderr or None,
                duration_ms=hook_result.duration_ms,
                timestamp=datetime.now(timezone.utc),
            )
        )
    return HookResults(captured=captured, executions=executions)


class HookHandler:
    """Event handler that fires shell commands after successful tool calls."""

    def __init__(self, config: HooksConfig, workspace_dir: Path, interactive: bool = True):
        self.config = config
        self.workspace_dir = workspace_dir
        self.interactive = interactive
        self._pending: Optional[tuple[str, dict[str, Any]]] = None
        self._executions: list[HookExecution] = []

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

        tool_rules = [rule for rule in self.config.post_tool if tool_name in rule.tools or "*" in rule.tools]
        results = _render_and_execute(
            _jinja_env,
            tool_rules,
            context,
            self.workspace_dir,
            interactive=self.interactive,
            phase="post_tool",
        )
        self._executions.extend(results.executions)

    def drain_executions(self) -> list[HookExecution]:
        """Return and clear accumulated hook executions."""
        execs = self._executions
        self._executions = []
        return execs


# Module-level state so save_run_to_history can retrieve executions
_active_handler: Optional[HookHandler] = None
_pre_message_executions: list[HookExecution] = []


def setup_hook_handler(workspace_dir: Path, event_bus: "EventBus", interactive: bool = True) -> None:
    """Load hooks config and subscribe handler to event_bus if rules exist."""
    global _active_handler
    config = load_hooks_config(workspace_dir)
    if config and config.post_tool:
        _active_handler = HookHandler(config, workspace_dir, interactive=interactive)
        event_bus.subscribe(_active_handler.handle_event)
    else:
        _active_handler = None


def drain_all_executions() -> list[HookExecution]:
    """Drain and return all accumulated hook executions (pre_message + post_tool)."""
    global _pre_message_executions
    execs = _pre_message_executions
    _pre_message_executions = []
    if _active_handler:
        execs.extend(_active_handler.drain_executions())
    return execs


async def _fire_hooks(
    workspace_dir: Path,
    phase: str,
    context: dict[str, Any],
    interactive: bool = True,
    on_status: Optional[Callable[[str], None]] = None,
) -> HookResults:
    """Load config and fire hooks for the given phase.

    Returns:
        HookResults with captured variables and execution records.
    """
    config = load_hooks_config(workspace_dir)
    if not config:
        return HookResults(captured={}, executions=[])

    rules = getattr(config, phase)
    if not rules:
        return HookResults(captured={}, executions=[])

    return await asyncio.to_thread(
        _render_and_execute,
        _jinja_env,
        rules,
        context,
        workspace_dir,
        interactive=interactive,
        on_status=on_status,
        phase=phase,
    )


async def fire_compact_hooks(
    workspace_dir: Path,
    phase: Literal["pre_compact", "post_compact"],
    context: dict[str, Any],
    interactive: bool = True,
) -> list[HookExecution]:
    """Fire pre_compact or post_compact hooks. Returns execution records."""
    results = await _fire_hooks(workspace_dir, phase, context, interactive=interactive)
    return results.executions


async def fire_pre_message_hooks(
    workspace_dir: Path,
    context: dict[str, Any],
    interactive: bool = True,
    on_status: Optional[Callable[[str], None]] = None,
) -> dict[str, str]:
    """Fire pre_message hooks, returning captured variables. Executions are accumulated internally."""
    global _pre_message_executions
    results = await _fire_hooks(workspace_dir, "pre_message", context, interactive=interactive, on_status=on_status)
    _pre_message_executions.extend(results.executions)
    return results.captured
