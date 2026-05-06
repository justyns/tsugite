"""Tests for tsu:exec directive integration in the agent preparation pipeline."""

from tsugite.agent_preparation import AgentPreparer
from tsugite.md_agents import parse_agent_file


class TestExecDirectivePipeline:
    """Pipeline integration tests for the tsu:exec directive.

    These tests exercise the full prepare_agent pipeline. They should fail until
    the parser, runner, and pipeline wiring all land.
    """

    def test_prepare_agent_with_exec_directive_assigns_var(self, tmp_path):
        """Exec block assigns a variable that becomes available in the rendered template."""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text("""---
name: exec_test
extends: none
tools: []
---

<!-- tsu:exec name="dispatch" assign="targets" -->
targets = ["alpha", "beta", "gamma"]
targets
<!-- /tsu:exec -->

Targets: {{ targets }}
""")

        agent = parse_agent_file(agent_file)
        prepared = AgentPreparer().prepare(agent=agent, prompt="run", context={})

        assert "alpha" in prepared.rendered_prompt
        assert "beta" in prepared.rendered_prompt
        assert "gamma" in prepared.rendered_prompt
        # The directive itself should be replaced, not echoed verbatim
        assert "tsu:exec" not in prepared.rendered_prompt

    def test_prepare_agent_exec_sees_caller_context(self, tmp_path):
        """Exec block sees vars passed in via the `context` parameter as Python locals."""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text("""---
name: ctx_then_exec
extends: none
tools: []
---

<!-- tsu:exec name="combine" assign="combined" -->
combined = f"prefix:{caller_value}"
combined
<!-- /tsu:exec -->

{{ combined }}
""")

        agent = parse_agent_file(agent_file)
        prepared = AgentPreparer().prepare(agent=agent, prompt="run", context={"caller_value": "from-context"})

        assert "prefix:from-context" in prepared.rendered_prompt

    def test_prepare_agent_render_mode_executes_by_default(self, tmp_path):
        """Render mode (skip_tool_directives=True) still executes exec blocks."""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text("""---
name: render_exec
extends: none
tools: []
---

<!-- tsu:exec name="compute" assign="answer" -->
answer = 6 * 7
answer
<!-- /tsu:exec -->

Answer: {{ answer }}
""")

        agent = parse_agent_file(agent_file)
        prepared = AgentPreparer().prepare(agent=agent, prompt="run", context={}, skip_tool_directives=True)

        assert "Answer: 42" in prepared.rendered_prompt

    def test_prepare_agent_no_exec_flag_skips_with_placeholders(self, tmp_path):
        """When skip_exec_directives=True, exec is replaced with a placeholder string."""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text("""---
name: skip_exec
extends: none
tools: []
---

<!-- tsu:exec name="compute" assign="answer" -->
answer = 1 / 0  # would crash; placeholder path must not actually run
answer
<!-- /tsu:exec -->

Answer: {{ answer }}
""")

        agent = parse_agent_file(agent_file)
        prepared = AgentPreparer().prepare(agent=agent, prompt="run", context={}, skip_exec_directives=True)

        assert "Answer:" in prepared.rendered_prompt
        # The placeholder mentions the directive name so users can see what was skipped
        assert "compute" in prepared.rendered_prompt
        assert "not executed" in prepared.rendered_prompt.lower()
