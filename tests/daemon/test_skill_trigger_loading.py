"""Regression tests for issue #204: trigger-based skills must auto-load in daemon sessions.

The daemon constructs an enriched prompt (wrapping the user message in a
``<message_context>`` XML block) and calls ``run_agent``. This in turn calls
``AgentPreparer.prepare`` which is supposed to discover workspace skills, match
the message against any declared triggers, and load matched skills.

Before the fix the daemon path never threaded its ``Workspace`` through to
``run_agent``/``prepare``, so ``_collect_skill_roots`` only scanned skills
directories relative to the daemon's process CWD. Workspace skills were
invisible and triggers never fired.
"""

from pathlib import Path

from tsugite.agent_preparation import AgentPreparer
from tsugite.md_agents import parse_agent_file
from tsugite.workspace.models import Workspace


def _write_agent(path: Path) -> Path:
    agent_file = path / "agent.md"
    agent_file.write_text(
        "---\nname: test_agent\nextends: none\ntools: []\n---\n\n{{ user_prompt }}\n"
    )
    return agent_file


def _write_uridx_skill(skills_dir: Path) -> None:
    skill_dir = skills_dir / "uridx"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: uridx\ndescription: Search uridx index\ntriggers:\n  - uridx\n  - semantic search\n"
        "---\nUridx body.\n"
    )


def _enriched_prompt(user_message: str, workspace_dir: Path) -> str:
    """Mimic BaseAdapter._build_message_context output."""
    return (
        "<message_context>\n"
        "  <datetime>2026-04-24T00:00:00Z</datetime>\n"
        f"  <working_directory>{workspace_dir}</working_directory>\n"
        "  <source>http</source>\n"
        "  <user_id>tester</user_id>\n"
        "  <context_tokens_used>0</context_tokens_used>\n"
        "  <context_limit>200000</context_limit>\n"
        "</message_context>\n\n"
        f"{user_message}"
    )


class TestDaemonSkillTriggerLoading:
    def test_workspace_trigger_skill_loads_from_separate_cwd(self, tmp_path, monkeypatch):
        """Daemon path: enriched prompt + workspace argument must trigger workspace skills.

        Process CWD is a sibling directory (mirrors the daemon, which runs from
        wherever it was started, not from the user's workspace). The skill lives
        in the workspace's skills/ folder. Without ``workspace=`` threaded through
        prepare(), ``_collect_skill_roots`` cannot find it.
        """
        workspace_dir = tmp_path / "user_workspace"
        workspace_dir.mkdir()
        _write_uridx_skill(workspace_dir / "skills")

        elsewhere = tmp_path / "daemon_cwd"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        agent_path = _write_agent(elsewhere)
        agent = parse_agent_file(agent_path)
        workspace = Workspace.load(workspace_dir)

        prompt = _enriched_prompt(
            "Can you test if you can use uridx to search for similar topics?",
            workspace_dir,
        )

        prepared = AgentPreparer().prepare(
            agent=agent,
            prompt=prompt,
            context={"is_daemon": True},
            workspace=workspace,
        )

        assert "uridx" in prepared.context.get("_triggered_skill_names", [])
        assert "uridx" in [s.name for s in prepared.skills]

    def test_path_context_alone_resolves_workspace_skills(self, tmp_path, monkeypatch):
        """Reproduces issue #204. The daemon's run_agent call chain currently
        does not thread its Workspace object through to prepare(); only
        path_context (with workspace_dir) is available. prepare() must fall
        back to path_context.workspace_dir for skill discovery, otherwise
        workspace-declared triggers never fire from daemon sessions.
        """
        from tsugite.cli.helpers import PathContext

        workspace_dir = tmp_path / "user_workspace"
        workspace_dir.mkdir()
        _write_uridx_skill(workspace_dir / "skills")

        elsewhere = tmp_path / "daemon_cwd"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        agent_path = _write_agent(elsewhere)
        agent = parse_agent_file(agent_path)

        prompt = _enriched_prompt(
            "Can you test if you can use uridx to search for similar topics?",
            workspace_dir,
        )

        path_context = PathContext(
            invoked_from=workspace_dir,
            workspace_dir=workspace_dir,
            effective_cwd=workspace_dir,
        )

        prepared = AgentPreparer().prepare(
            agent=agent,
            prompt=prompt,
            context={"is_daemon": True},
            path_context=path_context,
        )

        assert "uridx" in prepared.context.get("_triggered_skill_names", [])
        assert "uridx" in [s.name for s in prepared.skills]

    def test_multi_word_trigger_fires_in_daemon_path(self, tmp_path, monkeypatch):
        """The phrase 'semantic search' must match a multi-word trigger inside an enriched prompt."""
        workspace_dir = tmp_path / "user_workspace"
        workspace_dir.mkdir()
        _write_uridx_skill(workspace_dir / "skills")

        elsewhere = tmp_path / "daemon_cwd"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        agent_path = _write_agent(elsewhere)
        agent = parse_agent_file(agent_path)
        workspace = Workspace.load(workspace_dir)

        prompt = _enriched_prompt(
            "Please run a semantic search across my notes.",
            workspace_dir,
        )

        prepared = AgentPreparer().prepare(
            agent=agent,
            prompt=prompt,
            context={"is_daemon": True},
            workspace=workspace,
        )

        assert "uridx" in prepared.context.get("_triggered_skill_names", [])
