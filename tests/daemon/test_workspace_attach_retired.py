"""Retiring the WORKSPACE_FILES convention auto-attach (#529).

Identity files reach the model through an agent's front-matter ``attachments:``,
never through convention discovery of the workspace directory. An agent that opts
out of the default base (``extends: none``) and declares no attachments gets no
workspace files auto-attached; a front-matter declaration still resolves.
"""

from pathlib import Path

from tsugite_daemon.adapters.base import BaseAdapter
from tsugite_daemon.config import RuntimeDefaults
from tsugite_daemon.session_store import SessionStore


class _StubAdapter(BaseAdapter):
    def get_platform_name(self) -> str:
        return "test"

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


def _workspace_with_identity(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "PERSONA.md").write_text("# Persona\nYou are helpful.\n")
    (ws / "USER.md").write_text("# User\n- **Name:** Sam\n")
    return ws


def _adapter(ws: Path, agent_file: str) -> _StubAdapter:
    config = RuntimeDefaults(workspace_dir=ws, agent_file=agent_file)
    return _StubAdapter(config, SessionStore(ws.parent / "store.json"))


def test_extends_none_agent_without_attachments_gets_nothing(tmp_path):
    ws = _workspace_with_identity(tmp_path)
    agent = ws / "bare.md"
    agent.write_text("---\nname: bare\nextends: none\n---\n\nHi.\n")

    adapter = _adapter(ws, str(agent))

    assert adapter._get_all_attachments() == []


def test_frontmatter_attachments_still_resolve(tmp_path):
    ws = _workspace_with_identity(tmp_path)
    agent = ws / "declared.md"
    agent.write_text("---\nname: declared\nextends: none\nattachments:\n  - [PERSONA.md, USER.md]\n---\n\nHi.\n")

    adapter = _adapter(ws, str(agent))

    names = {a.name for a in adapter._get_all_attachments()}
    assert names == {"PERSONA.md", "USER.md"}
