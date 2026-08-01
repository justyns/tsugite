"""Regression tests for the multi-step rendering bug where the inherited
`default.md` preamble references framework flags (`is_daemon`, etc.) that the
step rendering context never includes.

Symptom (issue #259): any multi-step agent run via `tsu run +<agent>` failed
at the first step with `'is_daemon' is undefined` because the StrictUndefined
Jinja env caught the missing variable in the inherited environment block.

Two layers of defense are tested:

1. `default.md` itself must render against a minimal step_context (defensive
   `| default(...)` filters on every framework flag).
2. Steps render through the same preparation pipeline as single-shot
   prompts, so the flags a step sees are the flags the pipeline supplies -
   not a separate hardcoded set that can drift from it.
"""

import asyncio
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
    """A deliberately bare step context: no framework flags at all.

    `default.md` must survive this on its own defensive filters, which is what
    protects templates when a caller supplies none of the optional flags."""
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
    ],
)
def test_steps_get_the_same_framework_flags_as_single_shot_prompts(tmp_path, monkeypatch, flag_template):
    """Steps are prepared by AgentPreparer, so a step template sees exactly the
    framework flags a single-shot prompt sees.

    Multi-step used to render against a bespoke context dict with its own
    hardcoded flag defaults, which drifted from the real pipeline. Flags outside
    that set (`can_spawn_jobs`, `is_channel_session`, ...) are supplied by the
    daemon adapter via `context` and guarded with `| default(...)` in templates,
    which is the same contract single-shot agents have always had.
    """
    from tsugite.agent_runner import runner

    agent_file = tmp_path / "flags.md"
    agent_file.write_text(f"""---
name: flag_probe
model: ollama:qwen2.5-coder:7b
extends: none
tools: []
---
<!-- tsu:step name="probe" -->
FLAG[{flag_template}]
""")

    prompts = []

    async def fake_agent_run(self, task, return_full_result=False, stream=False):
        prompts.append(task)
        return "done"

    monkeypatch.setattr("tsugite.core.agent.TsugiteAgent.run", fake_agent_run)

    asyncio.run(runner.run_agent_async(agent_file, "test prompt"))

    assert prompts, "the step never rendered"
    assert "FLAG[]" in prompts[0]


def test_steps_inherit_caller_provided_flags(tmp_path, monkeypatch):
    """When the caller (e.g. daemon adapter) injects framework flags via
    context, a step template must see them rather than a stale default."""
    from tsugite.agent_runner import runner

    agent_file = tmp_path / "inherit.md"
    agent_file.write_text("""---
name: flag_inherit
model: ollama:qwen2.5-coder:7b
extends: none
tools: []
---
<!-- tsu:step name="probe" -->
{% if is_daemon %}DAEMON{% endif %} schedule={{ schedule_id }}
""")

    prompts = []

    async def fake_agent_run(self, task, return_full_result=False, stream=False):
        prompts.append(task)
        return "done"

    monkeypatch.setattr("tsugite.core.agent.TsugiteAgent.run", fake_agent_run)

    asyncio.run(
        runner.run_agent_async(
            agent_file,
            "test prompt",
            context={"is_daemon": True, "schedule_id": "sched-42"},
        )
    )

    assert "DAEMON" in prompts[0]
    assert "schedule=sched-42" in prompts[0]
