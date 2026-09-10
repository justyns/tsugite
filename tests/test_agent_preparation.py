"""Tests for tsu:exec directive integration in the agent preparation pipeline."""

from tsugite.agent_preparation import AgentPreparer
from tsugite.cli.helpers import PathContext
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
        # The placeholder includes the skipped directive's name.
        assert "compute" in prepared.rendered_prompt
        assert "not executed" in prepared.rendered_prompt.lower()

    def test_prepare_agent_no_exec_flag_placeholders_stdout_assign(self, tmp_path):
        """stdout_assign binds a variable on the executing path, so the skipped path
        has to define it too."""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text("""---
name: skip_exec_stdout
extends: none
tools: []
---

<!-- tsu:exec name="calc" assign="val" stdout_assign="logs" -->
print("working")
7
<!-- /tsu:exec -->

Value: {{ val }}
Logs: {{ logs }}
""")

        agent = parse_agent_file(agent_file)
        prepared = AgentPreparer().prepare(agent=agent, prompt="run", context={}, skip_exec_directives=True)

        assert prepared.rendered_prompt.count("not executed in render mode") == 2


class TestFrontMatterAttachmentBase:
    """Front-matter attachment paths resolve against the run's working directory."""

    def _agent(self, path, attachment="AGENTS.md"):
        path.write_text(f"""---
name: attach_test
extends: none
tools: []
attachments:
  - "{attachment}"
---

Task.
""")
        return parse_agent_file(path)

    def test_relative_attachment_resolves_against_effective_cwd(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        nested = workspace / "repos" / "nested"
        nested.mkdir(parents=True)
        (workspace / "AGENTS.md").write_text("workspace rules")
        (nested / "AGENTS.md").write_text("nested repo rules")
        monkeypatch.chdir(nested)
        path_context = PathContext(invoked_from=workspace, workspace_dir=workspace, effective_cwd=workspace)

        agent = self._agent(tmp_path / "agent.md")
        prepared = AgentPreparer().prepare(agent=agent, prompt="run", context={}, path_context=path_context)

        assert [(a.name, a.content) for a in prepared.attachments] == [("AGENTS.md", "workspace rules")]

    def test_attachment_path_renders_workspace_dir(self, tmp_path, monkeypatch):
        workspace = tmp_path / "workspace"
        (workspace / "shared").mkdir(parents=True)
        (workspace / "shared" / "NOTES.md").write_text("shared notes")
        monkeypatch.chdir(tmp_path)
        path_context = PathContext(invoked_from=tmp_path, workspace_dir=workspace, effective_cwd=tmp_path)

        agent = self._agent(tmp_path / "agent.md", "{{ WORKSPACE_DIR }}/shared/NOTES.md")
        prepared = AgentPreparer().prepare(agent=agent, prompt="run", context={}, path_context=path_context)

        assert [(a.name, a.content) for a in prepared.attachments] == [("NOTES.md", "shared notes")]

    def test_attachment_path_can_guard_on_workspace_dir(self, tmp_path, monkeypatch):
        (tmp_path / "NOTES.md").write_text("cwd notes")
        monkeypatch.chdir(tmp_path)

        agent = self._agent(
            tmp_path / "agent.md", "{% if WORKSPACE_DIR %}{{ WORKSPACE_DIR }}/shared/{% endif %}NOTES.md"
        )
        prepared = AgentPreparer().prepare(agent=agent, prompt="run", context={})

        assert [(a.name, a.content) for a in prepared.attachments] == [("NOTES.md", "cwd notes")]

    def test_relative_attachment_resolves_against_cwd_without_path_context(self, tmp_path, monkeypatch):
        (tmp_path / "AGENTS.md").write_text("cwd rules")
        monkeypatch.chdir(tmp_path)

        agent = self._agent(tmp_path / "agent.md")
        prepared = AgentPreparer().prepare(agent=agent, prompt="run", context={})

        assert [(a.name, a.content) for a in prepared.attachments] == [("AGENTS.md", "cwd rules")]
