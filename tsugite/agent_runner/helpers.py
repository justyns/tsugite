"""Shared helper functions for agent execution."""

import contextvars
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

from rich.console import Console

from tsugite.console import get_stderr_console

if TYPE_CHECKING:
    from tsugite.options import ExecutionOptions

# Console for warnings and debug output (stderr)
_stderr_console = get_stderr_console()

# Every piece of per-run ambient state below is a ContextVar, set with a token and
# reset on the way out. Two properties matter and only ContextVars give both:
# concurrent daemon sessions (each an `asyncio.to_thread` worker) stay isolated,
# and a NESTED run - an agent hook, which `hooks.py` awaits through
# `asyncio.wait_for` and therefore runs in the parent's own context - restores
# what it found instead of clobbering the parent for the rest of its run.
_current_agent_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("current_agent", default=None)

# Active sandbox policy. Presence means "this agent is running sandboxed"; tools
# read it to propagate the same isolation or to refuse. A nested run leaking its
# teardown here would fail OPEN, so the reset is load-bearing.
_sandbox_context_var: contextvars.ContextVar[Optional["SandboxContext"]] = contextvars.ContextVar(
    "sandbox_context", default=None
)


@dataclass
class SandboxContext:
    """Effective sandbox policy for the currently executing agent.

    Presence of a SandboxContext means the agent is running sandboxed; tools read
    it to propagate the same isolation to anything they spawn.
    """

    allow_domains: List[str] = field(default_factory=list)
    no_network: bool = False
    extra_ro_binds: List[Path] = field(default_factory=list)
    extra_rw_binds: List[Path] = field(default_factory=list)
    pass_env: List[str] = field(default_factory=list)
    workspace_dir: Optional[Path] = None


class SandboxToolDeniedError(RuntimeError):
    """Raised when a host-exec tool is refused because the agent runs sandboxed
    (see the deny_when_sandboxed decorator)."""


def set_sandbox_context(ctx: Optional["SandboxContext"]) -> contextvars.Token:
    """Set (or clear, with None) the active sandbox policy. Returns a reset token."""
    return _sandbox_context_var.set(ctx)


def get_sandbox_context() -> Optional["SandboxContext"]:
    """Return the active sandbox policy, or None when not running sandboxed."""
    return _sandbox_context_var.get()


def reset_sandbox_context(token: contextvars.Token) -> None:
    """Restore whatever sandbox policy was active before the matching set."""
    _sandbox_context_var.reset(token)


def clear_sandbox_context() -> None:
    """Drop the sandbox policy outright.

    For top-level callers that own the whole context (`tsu exec`, test teardown).
    Anything that nests a run inside another must use the token reset instead.
    """
    _sandbox_context_var.set(None)


def sandbox_context_to_override() -> Optional[dict]:
    """Serialize the active sandbox policy as a metadata override dict, or None.

    Spawn tools stamp this onto the records they create (sessions, jobs,
    schedules) so the spawned daemon run inherits the same sandbox when it later
    reaches the adapter chokepoint. The shape matches SandboxSettings so it can
    be validated back there; paths are stringified to survive JSON metadata.
    """
    ctx = get_sandbox_context()
    if ctx is None:
        return None
    return {
        "enabled": True,
        "no_network": ctx.no_network,
        "allow_domains": list(ctx.allow_domains),
        "extra_ro_binds": [str(p) for p in ctx.extra_ro_binds],
        "extra_rw_binds": [str(p) for p in ctx.extra_rw_binds],
        "pass_env": list(ctx.pass_env),
    }


def build_sandbox_policy(
    exec_options: "ExecutionOptions",
    *,
    workspace_dir: Optional[Path] = None,
    agent_config: Any = None,
):
    """Resolve the effective sandbox policy into (SandboxConfig, SandboxContext).

    Returns (None, None) when the sandbox is off. Shared by the agent runner and
    `tsu exec` so the two never drift. Agent frontmatter (network/sandbox) can only
    tighten the CLI/daemon ceiling, never loosen it.

    Raises RuntimeError if the sandbox is requested but bwrap is unavailable.
    """
    from tsugite.agent_runner.runner import resolve_effective_sandbox
    from tsugite.core.sandbox import SandboxConfig, sandbox_available

    sandbox_on, allow_domains, no_network = resolve_effective_sandbox(
        daemon_enabled=exec_options.sandbox,
        daemon_domains=list(exec_options.allow_domains),
        daemon_no_network=exec_options.no_network,
        fm_network=getattr(agent_config, "network", None),
        fm_sandbox=getattr(agent_config, "sandbox", None),
    )
    if not sandbox_on:
        return None, None

    if not sandbox_available():
        raise RuntimeError("No sandbox backend available. Install tsugite-sandbox and bubblewrap, or use --no-sandbox.")

    ctx = SandboxContext(
        allow_domains=allow_domains,
        no_network=no_network,
        extra_ro_binds=list(exec_options.extra_ro_binds),
        extra_rw_binds=list(exec_options.extra_rw_binds),
        pass_env=list(exec_options.pass_env),
        workspace_dir=workspace_dir,
    )
    config = SandboxConfig(
        allowed_domains=ctx.allow_domains,
        no_network=ctx.no_network,
        extra_ro_binds=ctx.extra_ro_binds,
        extra_rw_binds=ctx.extra_rw_binds,
        pass_env=ctx.pass_env,
    )
    return config, ctx


# Per-run policy set by the runner and read by tools during execution.
#
# ContextVars, not module globals: the daemon runs each agent loop in an
# `asyncio.to_thread` worker, so several runs are in flight at once and a global
# would hand one run another's policy. `to_thread` copies the caller's context
# into the worker and discards mutations on the way out, which also scopes these
# to the run that set them without an explicit clear.
_allowed_agents_var: contextvars.ContextVar[Optional[List[str]]] = contextvars.ContextVar(
    "allowed_agents", default=None
)
_allowed_secrets_var: contextvars.ContextVar[Optional[List[str]]] = contextvars.ContextVar(
    "allowed_secrets", default=None
)


def set_current_agent(name: str) -> contextvars.Token:
    """Record the currently executing agent. Returns a reset token."""
    return _current_agent_var.set(name)


def get_current_agent() -> Optional[str]:
    """Return the currently executing agent's name, or None."""
    return _current_agent_var.get()


def reset_current_agent(token: contextvars.Token) -> None:
    """Restore whatever agent was current before the matching set."""
    _current_agent_var.reset(token)


def resolve_current_agent(explicit: Optional[str] = None, default: str = "default") -> str:
    """Resolve agent name: explicit value > current agent context > default."""
    if explicit is not None:
        return explicit
    return get_current_agent() or default


def set_allowed_secrets(secrets: Optional[List[str]]) -> contextvars.Token:
    """Set the secret allowlist for the current run. Returns a reset token.

    An empty list and None both mean unrestricted; agents express a restriction
    by listing names.
    """
    return _allowed_secrets_var.set(secrets)


def reset_allowed_secrets(token: contextvars.Token) -> None:
    """Restore whatever secret allowlist was active before the matching set."""
    _allowed_secrets_var.reset(token)


def get_allowed_secrets() -> Optional[List[str]]:
    """Return the current run's secret allowlist. Empty or None = unrestricted."""
    return _allowed_secrets_var.get()


def clear_current_agent() -> None:
    """Clear the currently executing agent (tests and top-level teardown)."""
    _current_agent_var.set(None)


def set_allowed_agents(agents: Optional[List[str]]) -> contextvars.Token:
    """Set list of allowed agents for spawning in multi-agent mode.

    Returns a reset token.

    Args:
        agents: List of agent names allowed to spawn, or None for unrestricted
    """
    return _allowed_agents_var.set(agents)


def reset_allowed_agents(token: contextvars.Token) -> None:
    """Restore whatever spawn allowlist was active before the matching set."""
    _allowed_agents_var.reset(token)


def get_allowed_agents() -> Optional[List[str]]:
    """Get list of allowed agents for spawning.

    Returns:
        List of allowed agent names, or None if unrestricted
    """
    return _allowed_agents_var.get()


def clear_allowed_agents() -> None:
    """Clear the allowed agents list for the current run."""
    _allowed_agents_var.set(None)


def get_display_console(custom_logger: Optional[Any]) -> Console:
    """Get console for displaying output, with fallback to stderr.

    Args:
        custom_logger: Custom logger instance (may be None)

    Returns:
        Console instance to use for output
    """
    if custom_logger and hasattr(custom_logger, "console"):
        return custom_logger.console
    return _stderr_console


def get_ui_handler(custom_logger: Optional[Any]) -> Optional[Any]:
    """Safely get UI handler from custom logger.

    Args:
        custom_logger: Custom logger instance (may be None)

    Returns:
        UI handler if available, None otherwise
    """
    return custom_logger.ui_handler if custom_logger and hasattr(custom_logger, "ui_handler") else None


def set_multistep_ui_context(custom_logger: Optional[Any], step_number: int, step_name: str, total_steps: int) -> None:
    """Set multistep context in UI handler if available.

    Only a handler that renders a live progress display tracks this, and the handler
    contract is `handle_event` alone - a plugin may ship one subclassing nothing.

    Args:
        custom_logger: Custom logger instance (may be None)
        step_number: Current step number
        step_name: Name of current step
        total_steps: Total number of steps
    """
    setter = getattr(get_ui_handler(custom_logger), "set_multistep_context", None)
    if setter:
        setter(step_number, step_name, total_steps)


def clear_multistep_ui_context(custom_logger: Optional[Any]) -> None:
    """Clear multistep context from UI handler if available.

    Args:
        custom_logger: Custom logger instance (may be None)
    """
    clearer = getattr(get_ui_handler(custom_logger), "clear_multistep_context", None)
    if clearer:
        clearer()


def print_step_progress(
    custom_logger: Optional[Any], step_header: str, message: str, debug: bool = False, style: str = "cyan"
) -> None:
    """Print step progress message using event system.

    Args:
        custom_logger: Custom logger instance (may be None)
        step_header: Step header string
        message: Message to display
        debug: Whether debug mode is active (skips output if True)
        style: Rich style string (e.g., "cyan", "green", "yellow")
    """
    if debug:
        return

    # Emit as StepProgressEvent through event bus
    ui_handler = get_ui_handler(custom_logger)
    if ui_handler:
        from tsugite.events import EventBus, StepProgressEvent

        event_bus = EventBus()
        event_bus.subscribe(ui_handler.handle_event)
        event_bus.emit(StepProgressEvent(message=f"{step_header} {message}", style=style))
