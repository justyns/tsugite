"""Regression tests for the multi-step rendering bug where the inherited
`default.md` preamble references framework flags (`is_daemon`, etc.) that the
step rendering context never includes.

Symptom (issue #259): any multi-step agent run via `tsu run +<agent>` failed
at the first step with `'is_daemon' is undefined` because the StrictUndefined
Jinja env caught the missing variable in the inherited environment block.

Two layers of defense are tested:

1. `default.md` itself must render against a minimal step_context (defensive
   `| default(...)` filters on every framework flag).
2. The step_context built by `_run_multistep_agent_impl` includes the
   framework flags with safe defaults so any other plugin templates that
   reference them naturally also work.
"""

from pathlib import Path

import pytest

from tsugite.renderer import AgentRenderer


def _builtin_default_body() -> str:
    """Read the body of the shipped default.md (after the YAML frontmatter)."""
    raw = Path("tsugite/builtin_agents/default.md").read_text()
    parts = raw.split("---\n", 2)
    assert len(parts) == 3, "default.md must have YAML frontmatter"
    return parts[2]


def _minimal_step_context() -> dict:
    """The variables a multi-step agent's step_context starts with, before any
    framework flags get injected. Mirrors the keys assembled in
    `_run_multistep_agent_impl` (runner.py around line 1558)."""
    return {
        "user_prompt": "test",
        "is_interactive": False,
        "tools": [],
        "is_subagent": False,
        "parent_agent": None,
        "step_number": 1,
        "step_name": "echo",
        "total_steps": 1,
        "iteration": 1,
        "max_iterations": 1,
        "is_looping_step": False,
        "is_retry": False,
        "retry_count": 0,
        "max_retries": 0,
        "last_error": "",
        "all_errors": [],
    }


def test_default_md_preamble_renders_with_minimal_step_context():
    """The shipped `default.md` must render successfully against a multi-step
    step_context that does NOT include framework flags. Without the defensive
    `| default(false)` filters this raises `'is_daemon' is undefined`.
    """
    body = _builtin_default_body()
    renderer = AgentRenderer()

    rendered = renderer.render(body, _minimal_step_context())

    assert "is_daemon" not in rendered
    assert "Daemon Mode" not in rendered


@pytest.mark.parametrize(
    "flag_template",
    [
        "{% if is_daemon %}DAEMON{% endif %}",
        "{% if is_scheduled %}SCHEDULED{% endif %}",
        "{% if has_notify_tool %}NOTIFY{% endif %}",
        "{% if can_spawn_sessions %}SPAWN{% endif %}",
        "{% if is_channel_session %}CHANNEL{% endif %}",
    ],
)
def test_step_context_provides_framework_flag_defaults(flag_template):
    """`_run_multistep_agent_impl` should populate framework flags with safe
    defaults so user-authored steps and inherited templates can reference them
    without `| default(...)`. We build the step_context the same way the
    runner does and assert the conditional renders without raising.
    """
    from tsugite.agent_runner.runner import _build_multistep_step_context

    step_context = _build_multistep_step_context(prompt="test", context={}, agent_tools=[])

    renderer = AgentRenderer()
    rendered = renderer.render(flag_template, step_context)

    assert rendered == ""


def test_step_context_inherits_caller_provided_flags():
    """When the caller (e.g. daemon adapter) injects framework flags via
    context, the step_context must preserve them - the defaults are only
    fallbacks."""
    from tsugite.agent_runner.runner import _build_multistep_step_context

    caller_context = {"is_daemon": True, "agent_name": "default", "schedule_id": "sched-42"}
    step_context = _build_multistep_step_context(prompt="test", context=caller_context, agent_tools=[])

    assert step_context["is_daemon"] is True
    assert step_context["agent_name"] == "default"
    assert step_context["schedule_id"] == "sched-42"
