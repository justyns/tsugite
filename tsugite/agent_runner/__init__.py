"""Agent execution engine - public API."""

# Re-export public functions for backwards compatibility
from tsugite.agent_runner.helpers import (  # noqa: F401
    clear_allowed_agents,
    clear_current_agent,
    get_allowed_agents,
    get_current_agent,
    get_display_console,
    get_ui_handler,
    set_allowed_agents,
    set_current_agent,
)
from tsugite.agent_runner.metrics import StepMetrics, display_step_metrics  # noqa: F401
from tsugite.agent_runner.models import AgentExecutionResult  # noqa: F401
from tsugite.agent_runner.runner import (  # noqa: F401
    _combine_instructions,
    _execute_agent_with_prompt,
    execute_prefetch,
    execute_tool_directives,
    get_default_instructions,
    preview_multistep_agent,
    run_agent,
    run_agent_async,
    run_multistep_agent,
    run_multistep_agent_async,
)
from tsugite.agent_runner.validation import get_agent_info, validate_agent_file  # noqa: F401
from tsugite.tools import call_tool  # noqa: F401 - Re-export for test compatibility

__all__ = [
    "run_agent",
    "run_agent_async",
    "run_multistep_agent",
    "run_multistep_agent_async",
    "preview_multistep_agent",
    "execute_prefetch",
    "execute_tool_directives",
    "get_default_instructions",
    "AgentExecutionResult",
    "StepMetrics",
    "display_step_metrics",
    "validate_agent_file",
    "get_agent_info",
    "get_current_agent",
    "set_current_agent",
    "clear_current_agent",
    "get_allowed_agents",
    "set_allowed_agents",
    "clear_allowed_agents",
    "get_display_console",
    "get_ui_handler",
]
