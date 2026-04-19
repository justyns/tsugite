"""Workspace hooks - fire shell commands after tool calls and lifecycle events."""

import asyncio
import inspect
import json
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
from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    """A hook rule - used for all hook phases (post_tool, pre_message, pre_context_build, etc.)."""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    type: Literal["shell", "agent", "python"] = "shell"
    tools: list[str] = Field(default_factory=list)
    match: Optional[str] = None
    run: Optional[Union[str, list[str]]] = None
    wait: bool = False
    capture_as: Optional[str] = None
    only_interactive: bool = False
    name: Optional[str] = None
    env: dict[str, str] = Field(default_factory=dict)
    agent: Optional[str] = None
    hook_model: Optional[str] = Field(default=None, alias="model")
    max_turns: Optional[int] = None
    hook_callable: Optional[Callable] = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _validate_type_fields(self) -> "HookRule":
        if self.type == "shell":
            if self.run is None:
                raise ValueError("shell hooks require 'run'")
            if self.agent is not None:
                raise ValueError("shell hooks cannot have 'agent'")
        elif self.type == "agent":
            if self.agent is None:
                raise ValueError("agent hooks require 'agent'")
            if self.run is not None:
                raise ValueError("agent hooks cannot have 'run'")
        elif self.type == "python":
            if self.hook_callable is None:
                raise ValueError("python hooks require 'hook_callable'")
            if self.run is not None:
                raise ValueError("python hooks cannot have 'run'")
            if self.agent is not None:
                raise ValueError("python hooks cannot have 'agent'")
        return self


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
    pre_context_build: list[HookRule] = Field(default_factory=list)
    post_context_build: list[HookRule] = Field(default_factory=list)
    pre_tool_call: list[HookRule] = Field(default_factory=list)
    pre_response: list[HookRule] = Field(default_factory=list)
    post_response: list[HookRule] = Field(default_factory=list)
    session_end: list[HookRule] = Field(default_factory=list)

    def merge(self, other: "dict[str, list[HookRule]]") -> None:
        """Append hook rules from another source (e.g. plugins) into this config."""
        for phase, rules in other.items():
            phase_list = getattr(self, phase, None)
            if phase_list is not None:
                phase_list.extend(rules)
            else:
                logger.warning("Unknown hook phase '%s', skipping", phase)


def load_hooks_config(workspace_dir: Path) -> Optional[HooksConfig]:
    """Load hooks config from .tsugite/hooks.yaml, merged with plugin hooks."""
    from tsugite.plugins import get_plugin_hooks

    hooks_path = workspace_dir / ".tsugite" / "hooks.yaml"
    plugin_hooks = get_plugin_hooks()

    yaml_config = None
    if hooks_path.exists():
        with open(hooks_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data and "hooks" in data:
            yaml_config = HooksConfig.model_validate(data["hooks"])

    if not yaml_config and not plugin_hooks:
        return None

    config = yaml_config or HooksConfig()
    if plugin_hooks:
        config.merge(plugin_hooks)

    return config


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


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


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
        elapsed = _elapsed_ms(start)
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
        elapsed = _elapsed_ms(start)
        logger.warning("Hook timed out after %ds: %s", HOOK_TIMEOUT, cmd)
        return HookResult(exit_code=-1, stdout="", stderr=f"Timed out after {HOOK_TIMEOUT}s", duration_ms=elapsed)
    except Exception as e:
        elapsed = _elapsed_ms(start)
        logger.warning("Hook error: %s", e)
        return HookResult(exit_code=-1, stdout="", stderr=str(e), duration_ms=elapsed)


async def _execute_agent_hook(
    rule: HookRule,
    context: dict[str, Any],
    cwd: Path,
) -> HookResult:
    """Run an agent hook. Returns structured result."""
    from tsugite.agent_inheritance import find_agent_file
    from tsugite.agent_runner import run_agent_async
    from tsugite.agent_runner.models import AgentExecutionResult
    from tsugite.options import ExecutionOptions

    start = time.monotonic()
    try:
        agent_path = find_agent_file(rule.agent, cwd)
        if agent_path is None:
            elapsed = _elapsed_ms(start)
            msg = f"Agent not found: {rule.agent}"
            logger.warning("Hook agent error: %s", msg)
            return HookResult(exit_code=-1, stdout="", stderr=msg, duration_ms=elapsed)

        prompt = json.dumps(context, default=str)
        exec_options = ExecutionOptions(
            model_override=rule.hook_model,
            max_turns_override=rule.max_turns,
        )

        logger.info("Hook agent fired: %s", rule.agent)
        result = await asyncio.wait_for(
            run_agent_async(agent_path, prompt, exec_options, context=context),
            timeout=HOOK_TIMEOUT,
        )
        response = result.response if isinstance(result, AgentExecutionResult) else str(result)
        elapsed = _elapsed_ms(start)
        return HookResult(exit_code=0, stdout=response.strip(), stderr="", duration_ms=elapsed)
    except asyncio.TimeoutError:
        elapsed = _elapsed_ms(start)
        logger.warning("Hook agent timed out after %ds: %s", HOOK_TIMEOUT, rule.agent)
        return HookResult(exit_code=-1, stdout="", stderr=f"Timed out after {HOOK_TIMEOUT}s", duration_ms=elapsed)
    except Exception as e:
        elapsed = _elapsed_ms(start)
        logger.warning("Hook agent error: %s", e)
        return HookResult(exit_code=-1, stdout="", stderr=str(e), duration_ms=elapsed)


async def _execute_python_hook(
    rule: HookRule,
    context: dict[str, Any],
) -> HookResult:
    """Run a Python callable hook. Returns structured result."""
    start = time.monotonic()
    try:
        fn = rule.hook_callable
        logger.info("Hook python fired: %s", rule.name or fn.__name__)
        if inspect.iscoroutinefunction(fn):
            result = await asyncio.wait_for(fn(context), timeout=HOOK_TIMEOUT)
        else:
            result = await asyncio.wait_for(asyncio.to_thread(fn, context), timeout=HOOK_TIMEOUT)
        elapsed = _elapsed_ms(start)
        stdout = str(result).strip() if result is not None else ""
        return HookResult(exit_code=0, stdout=stdout, stderr="", duration_ms=elapsed)
    except asyncio.TimeoutError:
        elapsed = _elapsed_ms(start)
        label = rule.name or "python hook"
        logger.warning("Hook python timed out after %ds: %s", HOOK_TIMEOUT, label)
        return HookResult(exit_code=-1, stdout="", stderr=f"Timed out after {HOOK_TIMEOUT}s", duration_ms=elapsed)
    except Exception as e:
        elapsed = _elapsed_ms(start)
        label = rule.name or "python hook"
        logger.warning("Hook python error (%s): %s", label, e)
        return HookResult(exit_code=-1, stdout="", stderr=str(e), duration_ms=elapsed)


class HookResults(NamedTuple):
    """Combined results from executing hook rules."""

    captured: dict[str, str]
    executions: list[HookExecution]


def _render_shell_rule(
    jinja_env: jinja2.Environment,
    rule: HookRule,
    context: dict[str, Any],
    base_env: dict[str, str],
) -> Optional[tuple[Union[str, list[str]], dict[str, str]]]:
    """Render command and env for a shell rule. Returns None on failure."""
    try:
        if isinstance(rule.run, list):
            cmd: Union[str, list[str]] = [jinja_env.from_string(part).render(context) for part in rule.run]
        else:
            cmd = jinja_env.from_string(rule.run).render(context)
    except Exception as e:
        logger.warning("Hook render failed: %s", e)
        return None

    if rule.env:
        env = base_env.copy()
        for env_key, env_val in rule.env.items():
            try:
                env[env_key] = jinja_env.from_string(env_val).render(context)
            except Exception as e:
                logger.warning("Hook env render failed for %s: %s", env_key, e)
    else:
        env = base_env

    return cmd, env


def _build_execution(rule: HookRule, cmd_str: str, hook_result: HookResult, phase: str) -> HookExecution:
    return HookExecution(
        phase=phase,
        name=rule.name or rule.capture_as,
        command=cmd_str,
        exit_code=hook_result.exit_code,
        stdout=hook_result.stdout or None,
        stderr=hook_result.stderr or None,
        duration_ms=hook_result.duration_ms,
        timestamp=datetime.now(timezone.utc),
    )


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
        if rule.type in ("agent", "python"):
            logger.warning("%s hooks are not supported in sync execution (phase=%s), skipping", rule.type, phase)
            continue
        if rule.only_interactive and not interactive:
            continue
        if rule.match and not _match_passes(jinja_env, rule.match, context):
            continue
        if on_status:
            label = rule.name or rule.capture_as or "hook"
            on_status(f"Running {label}...")

        rendered = _render_shell_rule(jinja_env, rule, context, base_env)
        if rendered is None:
            continue
        cmd, env = rendered

        hook_result = _execute_hook(cmd, cwd, env=env)
        cmd_str = cmd if isinstance(cmd, str) else shlex.join(cmd)

        if rule.capture_as and hook_result.exit_code == 0 and hook_result.stdout:
            captured[rule.capture_as] = hook_result.stdout
        executions.append(_build_execution(rule, cmd_str, hook_result, phase))
    return HookResults(captured=captured, executions=executions)


async def _render_and_execute_async(
    jinja_env: jinja2.Environment,
    rules: list[HookRule],
    context: dict[str, Any],
    cwd: Path,
    interactive: bool = True,
    on_status: Optional[Callable[[str], None]] = None,
    on_result: Optional[Callable[["HookExecution"], None]] = None,
    phase: str = "",
) -> HookResults:
    """Render and execute matching hook rules, supporting both shell and agent types."""
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

        if rule.type == "agent":
            hook_result = await _execute_agent_hook(rule, context, cwd)
            cmd_str = f"agent:{rule.agent}"
        elif rule.type == "python":
            hook_result = await _execute_python_hook(rule, context)
            cmd_str = f"python:{rule.name or rule.hook_callable.__name__}"
        else:
            rendered = _render_shell_rule(jinja_env, rule, context, base_env)
            if rendered is None:
                continue
            cmd, env = rendered
            hook_result = await asyncio.to_thread(_execute_hook, cmd, cwd, env)
            cmd_str = cmd if isinstance(cmd, str) else shlex.join(cmd)

        if rule.capture_as and hook_result.exit_code == 0 and hook_result.stdout:
            captured[rule.capture_as] = hook_result.stdout
        execution = _build_execution(rule, cmd_str, hook_result, phase)
        executions.append(execution)
        if on_result:
            on_result(execution)
    return HookResults(captured=captured, executions=executions)


def _log_task_exception(task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception():
        logger.warning("Background hook task failed: %s", task.exception())


def fire_hooks_background(phase: str, context: dict[str, Any], workspace_dir: Path | None = None) -> None:
    """Fire hooks for a phase as a background task."""
    workspace_dir = workspace_dir or Path.cwd()
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(fire_hooks(workspace_dir, phase, context))
        task.add_done_callback(_log_task_exception)
    except RuntimeError:
        logger.debug("No running event loop; %s hooks not scheduled", phase)


class HookHandler:
    """Event handler that fires hooks on tool call events."""

    def __init__(self, config: HooksConfig, workspace_dir: Path, interactive: bool = True):
        self.config = config
        self.workspace_dir = workspace_dir
        self.interactive = interactive
        self._pending: Optional[tuple[str, dict[str, Any]]] = None
        self._executions: list[HookExecution] = []

    def handle_event(self, event: BaseEvent) -> None:
        if isinstance(event, ToolCallEvent):
            self._pending = (event.tool_name, event.arguments)
            self._fire_pre_tool_call(event.tool_name, event.arguments)
        elif isinstance(event, ToolResultEvent):
            if event.success and self._pending:
                self._fire_post_tool(*self._pending)
            self._pending = None

    def _build_tool_context(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        context: dict[str, Any] = {"tool": tool_name, **arguments}
        if "path" in arguments:
            try:
                context["path"] = str(Path(arguments["path"]).resolve().relative_to(self.workspace_dir.resolve()))
            except ValueError:
                pass
        return context

    def _fire_pre_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> None:
        context = self._build_tool_context(tool_name, arguments)
        rules = [r for r in self.config.pre_tool_call if tool_name in r.tools or "*" in r.tools]
        if not rules:
            return
        # Python/agent hooks need async
        has_async = any(r.type in ("python", "agent") for r in rules)
        if has_async:
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(
                    _render_and_execute_async(
                        _jinja_env,
                        rules,
                        context,
                        self.workspace_dir,
                        interactive=self.interactive,
                        phase="pre_tool_call",
                    )
                )
                task.add_done_callback(_log_task_exception)
            except RuntimeError:
                logger.debug("No running event loop; pre_tool_call hooks not scheduled")
        else:
            results = _render_and_execute(
                _jinja_env, rules, context, self.workspace_dir, interactive=self.interactive, phase="pre_tool_call"
            )
            self._executions.extend(results.executions)

    def _fire_post_tool(self, tool_name: str, arguments: dict[str, Any]) -> None:
        context = self._build_tool_context(tool_name, arguments)
        tool_rules = [rule for rule in self.config.post_tool if tool_name in rule.tools or "*" in rule.tools]
        if not tool_rules:
            return
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
    if config and (config.post_tool or config.pre_tool_call):
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


HookPhase = Literal[
    "post_tool",
    "pre_compact",
    "post_compact",
    "pre_message",
    "pre_context_build",
    "post_context_build",
    "pre_tool_call",
    "pre_response",
    "post_response",
    "session_end",
]


async def fire_hooks(
    workspace_dir: Path,
    phase: HookPhase,
    context: dict[str, Any],
    interactive: bool = True,
    on_status: Optional[Callable[[str], None]] = None,
    on_result: Optional[Callable[["HookExecution"], None]] = None,
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

    return await _render_and_execute_async(
        _jinja_env,
        rules,
        context,
        workspace_dir,
        interactive=interactive,
        on_status=on_status,
        on_result=on_result,
        phase=phase,
    )


async def fire_compact_hooks(
    workspace_dir: Path,
    phase: Literal["pre_compact", "post_compact"],
    context: dict[str, Any],
    interactive: bool = True,
) -> list[HookExecution]:
    """Fire pre_compact or post_compact hooks. Returns execution records."""
    results = await fire_hooks(workspace_dir, phase, context, interactive=interactive)
    return results.executions


async def fire_pre_message_hooks(
    workspace_dir: Path,
    context: dict[str, Any],
    interactive: bool = True,
    on_status: Optional[Callable[[str], None]] = None,
    on_result: Optional[Callable[["HookExecution"], None]] = None,
) -> dict[str, str]:
    """Fire pre_message hooks, returning captured variables. Executions are accumulated internally."""
    global _pre_message_executions
    results = await fire_hooks(
        workspace_dir, "pre_message", context, interactive=interactive, on_status=on_status, on_result=on_result
    )
    _pre_message_executions.extend(results.executions)
    return results.captured
