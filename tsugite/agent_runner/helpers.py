"""Shared helper functions for agent execution."""

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

from rich.console import Console

from tsugite.console import get_stderr_console

if TYPE_CHECKING:
    from tsugite.options import ExecutionOptions

# Console for warnings and debug output (stderr)
_stderr_console = get_stderr_console()

# Thread-local storage for tracking currently executing agent
_current_agent_context = threading.local()

# Thread-local storage for the active sandbox policy. Set per-run (in the same
# thread as the agent loop and its parent-only tool dispatch) when an agent runs
# sandboxed; read by host-exec/spawn tools to inherit the sandbox or be denied.
# Thread-local (like _current_agent_context) so concurrent daemon sessions don't
# clobber each other.
_sandbox_context = threading.local()


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
    workspace_dir: Optional[Path] = None


class SandboxToolDeniedError(RuntimeError):
    """Raised when a host-exec tool is refused because the agent runs sandboxed
    (see the deny_when_sandboxed decorator)."""


def set_sandbox_context(ctx: Optional["SandboxContext"]) -> None:
    """Set (or clear, with None) the active sandbox policy for this thread."""
    _sandbox_context.value = ctx


def get_sandbox_context() -> Optional["SandboxContext"]:
    """Return the active sandbox policy, or None when not running sandboxed."""
    return getattr(_sandbox_context, "value", None)


def clear_sandbox_context() -> None:
    """Clear the active sandbox policy from thread-local storage."""
    if hasattr(_sandbox_context, "value"):
        delattr(_sandbox_context, "value")


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
    from tsugite.core.sandbox import BubblewrapSandbox, SandboxConfig

    sandbox_on, allow_domains, no_network = resolve_effective_sandbox(
        daemon_enabled=exec_options.sandbox,
        daemon_domains=list(exec_options.allow_domains),
        daemon_no_network=exec_options.no_network,
        fm_network=getattr(agent_config, "network", None),
        fm_sandbox=getattr(agent_config, "sandbox", None),
    )
    if not sandbox_on:
        return None, None

    if not BubblewrapSandbox.check_available():
        raise RuntimeError("bwrap not found. Install bubblewrap or use --no-sandbox.")

    ctx = SandboxContext(
        allow_domains=allow_domains,
        no_network=no_network,
        extra_ro_binds=list(exec_options.extra_ro_binds),
        extra_rw_binds=list(exec_options.extra_rw_binds),
        workspace_dir=workspace_dir,
    )
    config = SandboxConfig(
        allowed_domains=ctx.allow_domains,
        no_network=ctx.no_network,
        extra_ro_binds=ctx.extra_ro_binds,
        extra_rw_binds=ctx.extra_rw_binds,
    )
    return config, ctx


# Module-level storage for allowed agents (single-threaded CLI execution)
# Subagents run in separate processes, so this doesn't need to be thread-local
_allowed_agents: Optional[List[str]] = None
_allowed_secrets: Optional[List[str]] = None


def set_current_agent(name: str) -> None:
    """Set the name of the currently executing agent in thread-local storage."""
    _current_agent_context.name = name


def get_current_agent() -> Optional[str]:
    """Get the name of the currently executing agent from thread-local storage."""
    return getattr(_current_agent_context, "name", None)


def resolve_current_agent(explicit: Optional[str] = None, default: str = "default") -> str:
    """Resolve agent name: explicit value > current agent context > default."""
    if explicit is not None:
        return explicit
    return get_current_agent() or default


def set_allowed_secrets(secrets: Optional[List[str]]) -> None:
    global _allowed_secrets
    _allowed_secrets = secrets


def get_allowed_secrets() -> Optional[List[str]]:
    return _allowed_secrets


def clear_current_agent() -> None:
    """Clear the currently executing agent from thread-local storage."""
    if hasattr(_current_agent_context, "name"):
        delattr(_current_agent_context, "name")


def set_allowed_agents(agents: Optional[List[str]]) -> None:
    """Set list of allowed agents for spawning in multi-agent mode.

    Args:
        agents: List of agent names allowed to spawn, or None for unrestricted
    """
    global _allowed_agents
    _allowed_agents = agents


def get_allowed_agents() -> Optional[List[str]]:
    """Get list of allowed agents for spawning.

    Returns:
        List of allowed agent names, or None if unrestricted
    """
    return _allowed_agents


def clear_allowed_agents() -> None:
    """Clear the allowed agents list from module-level storage."""
    global _allowed_agents
    _allowed_agents = None


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

    Args:
        custom_logger: Custom logger instance (may be None)
        step_number: Current step number
        step_name: Name of current step
        total_steps: Total number of steps
    """
    ui_handler = get_ui_handler(custom_logger)
    if ui_handler:
        ui_handler.set_multistep_context(step_number, step_name, total_steps)


def clear_multistep_ui_context(custom_logger: Optional[Any]) -> None:
    """Clear multistep context from UI handler if available.

    Args:
        custom_logger: Custom logger instance (may be None)
    """
    ui_handler = get_ui_handler(custom_logger)
    if ui_handler:
        ui_handler.clear_multistep_context()


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
