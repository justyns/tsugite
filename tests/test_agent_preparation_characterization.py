"""Whole-output characterization of AgentPreparer.prepare().

`prepare()` was one 356-line method. These tests pin every field of the
PreparedAgent it produces for a range of agent shapes, so decomposing it into
named steps is provably behavior-preserving rather than merely test-passing.

They assert on the shape and the parts that must stay stable, not on exact
prompt text, so they survive unrelated wording changes to the runtime
instructions.
"""

import pytest

from tsugite.agent_preparation import AgentPreparer
from tsugite.md_agents import parse_agent


def _prepare(body: str, prompt: str = "do the thing", **kwargs):
    return AgentPreparer().prepare(agent=parse_agent(body), prompt=prompt, **kwargs)


MINIMAL = """---
name: minimal
model: ollama:fake
extends: none
tools: []
---
Hello {{ user_prompt }}
"""

WITH_TOOLS = """---
name: with_tools
model: ollama:fake
extends: none
tools: [read_file, write_file]
instructions: Be terse about {{ agent_name }}.
---
Body for {{ user_prompt }}
"""

WITH_GUARD = """---
name: guarded
model: ollama:fake
extends: none
tools: []
run_if: "false"
---
Never rendered
"""


def test_minimal_agent_prepares_a_complete_bundle():
    prepared = _prepare(MINIMAL)

    assert prepared.agent_config.name == "minimal"
    assert prepared.original_prompt == "do the thing"
    assert prepared.rendered_prompt == "Hello do the thing"
    assert prepared.user_message == prepared.rendered_prompt
    assert prepared.system_message
    assert prepared.combined_instructions
    assert prepared.tools == []
    assert prepared.attachments == []
    assert prepared.skipped is False
    assert prepared.skip_reason is None


def test_context_carries_the_framework_variables_templates_rely_on():
    prepared = _prepare(MINIMAL)
    ctx = prepared.context

    assert ctx["user_prompt"] == "do the thing"
    assert ctx["agent_name"] == "minimal"
    assert ctx["tools"] == []
    assert isinstance(ctx["available_tools"], list)
    for key in ("is_daemon", "is_scheduled", "is_subagent", "is_interactive"):
        assert isinstance(ctx[key], bool)
    for key in ("schedule_id", "conversation_id", "CWD"):
        assert isinstance(ctx[key], str)
    assert "INVOKED_FROM" in ctx
    assert "WORKSPACE_DIR" in ctx


def test_caller_context_overrides_framework_defaults():
    prepared = _prepare(MINIMAL, context={"is_daemon": True, "schedule_id": "sched-1"})

    assert prepared.context["is_daemon"] is True
    assert prepared.context["schedule_id"] == "sched-1"


def test_tools_are_expanded_and_instructions_are_rendered(file_tools):
    prepared = _prepare(WITH_TOOLS)

    assert sorted(t.name for t in prepared.tools) == ["read_file", "write_file"]
    # Agent instructions go through Jinja against the same context.
    assert "Be terse about with_tools." in prepared.combined_instructions
    # The system message advertises the expanded tools.
    assert "read_file" in prepared.system_message


def test_run_if_guard_short_circuits_without_rendering():
    prepared = _prepare(WITH_GUARD)

    assert prepared.skipped is True
    assert "run_if" in prepared.skip_reason
    assert prepared.rendered_prompt == ""
    assert prepared.system_message == ""
    assert prepared.tools == []


def test_skill_bookkeeping_keys_are_always_published():
    """The daemon reads these back off the context to update sticky state."""
    prepared = _prepare(MINIMAL)

    for key in ("_expired_sticky_skills", "_triggered_skill_names", "_auto_loaded_skill_names"):
        assert isinstance(prepared.context[key], list), f"{key} missing or not a list"
    assert isinstance(prepared.expiring_skills, dict)
    assert isinstance(prepared.skills, list)


@pytest.mark.parametrize("body", [MINIMAL, WITH_TOOLS])
def test_prepare_is_deterministic(body, file_tools):
    """Two identical calls must produce identical prompts and context keys."""
    first = _prepare(body)
    second = _prepare(body)

    assert first.rendered_prompt == second.rendered_prompt
    assert first.system_message == second.system_message
    assert first.combined_instructions == second.combined_instructions
    assert set(first.context) == set(second.context)
    assert [t.name for t in first.tools] == [t.name for t in second.tools]


def test_frontmatter_attachments_win_over_a_same_named_caller_attachment(tmp_path):
    """Ordering and dedup in _resolve_attachments are load-bearing.

    Front-matter attachments carry the cache tiers and are the intended source,
    so they must dedupe AHEAD of a same-named attachment the caller passed in.
    Flipping the concatenation order is invisible to every other test here.
    """
    from tsugite.attachments.base import Attachment, AttachmentContentType

    doc = tmp_path / "NOTES.md"
    doc.write_text("from frontmatter")

    caller_copy = Attachment(
        name="NOTES.md",
        content="from caller",
        content_type=AttachmentContentType.TEXT,
        mime_type="text/markdown",
    )

    body = f"""---
name: attach_order
model: ollama:fake
extends: none
tools: []
attachments:
  - {doc}
---
Body
"""
    prepared = _prepare(body, attachments=[caller_copy])

    names = [a.name for a in prepared.attachments]
    assert names.count("NOTES.md") == 1, f"attachment was not deduped: {names}"
    assert prepared.attachments[0].content == "from frontmatter", "caller attachment shadowed the front-matter one"


def test_path_context_drives_the_path_variables_and_the_environment_block(tmp_path):
    """`path_context` is never passed by the other tests, so _resolve_paths and
    the environment block were exercised only on their None branch - deleting
    the block outright passed every one of them."""
    from tsugite.cli.helpers import PathContext

    invoked = tmp_path / "invoked"
    workspace = tmp_path / "ws"
    invoked.mkdir()
    workspace.mkdir()

    prepared = _prepare(
        MINIMAL,
        path_context=PathContext(invoked_from=invoked, workspace_dir=workspace, effective_cwd=workspace),
    )

    assert prepared.context["CWD"] == str(workspace)
    assert prepared.context["INVOKED_FROM"] == str(invoked)
    assert prepared.context["WORKSPACE_DIR"] == str(workspace)
    assert "Invoked from:" in prepared.system_message
    assert str(invoked) in prepared.system_message


def test_sticky_skills_past_their_ttl_are_reported_expired():
    """_load_skills' TTL arithmetic, previously covered only on the empty path."""
    prepared = _prepare(MINIMAL, context={"sticky_skills": {"ghost-skill": 99}, "skill_ttl_default": 3})

    assert "ghost-skill" in prepared.context["_expired_sticky_skills"]
    assert "ghost-skill" not in prepared.expiring_skills
