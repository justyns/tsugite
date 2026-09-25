"""Cache-tier grouping for auto-attached context files.

An agent's front-matter ``attachments:`` may be a flat list (one cache block) or a
list of groups (ordered cache tiers). Grouping renders one cache-breakpointed
``<context>`` block per tier so a volatile file (now.md) only invalidates its own
tier, keeping the stable files cached.
"""

from __future__ import annotations

import pytest

from tsugite.attachments.base import Attachment, AttachmentContentType
from tsugite.core.agent import TsugiteAgent
from tsugite.md_agents import MAX_ATTACHMENT_TIERS, AgentConfig, AttachmentSpec


def _cfg(attachments):
    return AgentConfig(name="a", description="d", attachments=attachments)


# ── front-matter parsing ──


def test_flat_attachments_stay_one_tier():
    cfg = _cfg(["USER.md", "MEMORY.md"])
    assert cfg.attachments == ["USER.md", "MEMORY.md"]


def test_grouped_attachments_get_tier_indices():
    cfg = _cfg([["USER.md", "IDENTITY.md"], ["MEMORY.md"], ["now.md"]])
    assert [(s.path, s.tier) for s in cfg.attachments] == [
        ("USER.md", 0),
        ("IDENTITY.md", 0),
        ("MEMORY.md", 1),
        ("now.md", 2),
    ]


def test_grouped_dict_items_keep_their_spec_fields():
    cfg = _cfg([[{"path": "notes.md", "assign": "notes"}], ["now.md"]])
    assert cfg.attachments[0].assign == "notes" and cfg.attachments[0].tier == 0
    assert cfg.attachments[1].path == "now.md" and cfg.attachments[1].tier == 1


def test_mixing_flat_items_and_groups_is_rejected():
    with pytest.raises(ValueError, match="flat list or a list of groups"):
        _cfg(["USER.md", ["now.md"]])


def test_too_many_tiers_is_rejected():
    with pytest.raises(ValueError, match="at most"):
        _cfg([["a"]] * (MAX_ATTACHMENT_TIERS + 1))


def test_removal_marker_in_a_group_stays_a_plain_string():
    cfg = _cfg([["USER.md"], ["-OLD.md"]])
    assert "-OLD.md" in cfg.attachments
    assert any(isinstance(s, AttachmentSpec) and s.path == "USER.md" for s in cfg.attachments)


# ── tier threading through resolution ──


def test_resolution_tags_each_attachment_with_its_tier(tmp_path):
    from tsugite.attachments.agent_config import resolve_agent_config_attachments

    (tmp_path / "USER.md").write_text("user")
    (tmp_path / "now.md").write_text("now")
    cfg = _cfg([["USER.md"], ["now.md"]])
    atts, _ = resolve_agent_config_attachments(cfg.attachments, workspace_path=tmp_path)
    by_name = {a.name: a.tier for a in atts}
    assert by_name["USER.md"] == 0
    assert by_name["now.md"] == 1


# ── context-turn partition ──


def _att(name, tier):
    return Attachment(
        name=name, content="x", content_type=AttachmentContentType.TEXT, mime_type="text/plain", tier=tier
    )


def _bare_agent():
    # Skip __init__ (it needs a provider/config); _build_context_turns only touches
    # self.attachments/self.skills and delegates block-building, which we stub.
    return object.__new__(TsugiteAgent)


def test_context_turns_one_block_per_tier():
    agent = _bare_agent()
    agent.attachments = [_att("USER", 0), _att("MEM", 0), _att("now", 1)]
    agent.skills = []
    seen = []

    def stub(atts, skills):
        seen.append(([a.name for a in atts], list(skills)))
        return [1]

    agent._build_context_block = stub
    turns = agent._build_context_turns()
    assert len(turns) == 2
    assert seen == [(["USER", "MEM"], []), (["now"], [])]


def test_skills_ride_the_last_tier():
    agent = _bare_agent()
    agent.attachments = [_att("USER", 0), _att("now", 1)]
    agent.skills = ["skill-obj"]
    seen = []

    def stub(atts, skills):
        seen.append((atts[0].name, list(skills)))
        return [1]

    agent._build_context_block = stub
    agent._build_context_turns()
    assert seen == [("USER", []), ("now", ["skill-obj"])]


def test_flat_attachments_are_a_single_context_turn():
    agent = _bare_agent()
    agent.attachments = [_att("USER", 0), _att("MEM", 0)]
    agent.skills = []
    agent._build_context_block = lambda atts, skills: [len(atts)]
    assert len(agent._build_context_turns()) == 1


def _upload(name):
    a = _att(name, 0)
    a.user_upload = True
    return a


def test_user_uploads_are_excluded_from_the_cached_context_tiers():
    agent = _bare_agent()
    agent.attachments = [_att("USER", 0), _upload("photo.jpg")]
    agent.skills = []
    seen = []

    def stub(atts, skills):
        seen.append([a.name for a in atts])
        return [1]

    agent._build_context_block = stub
    agent._build_context_turns()
    assert seen == [["USER"]]  # the upload does not ride the cached context


def test_upload_blocks_carry_only_user_uploads():
    agent = _bare_agent()
    agent.attachments = [_att("USER", 0), _upload("photo.jpg")]
    captured = []

    def stub(atts, skills):
        captured.append([a.name for a in atts])
        return [{"type": "text", "text": "upload"}]

    agent._build_context_block = stub
    assert agent._build_upload_blocks() == [{"type": "text", "text": "upload"}]
    assert captured == [["photo.jpg"]]


def test_no_uploads_means_no_upload_blocks():
    agent = _bare_agent()
    agent.attachments = [_att("USER", 0)]
    assert agent._build_upload_blocks() == []


# ── base directory for relative paths ──
def test_relative_attachments_resolve_against_the_workspace(tmp_path):
    """A run with `--root` has a CWD outside its workspace, and every agent's
    relative attachment paths still name workspace files."""
    from tsugite.agent_preparation import AgentPreparer

    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "USER.md").write_text("from the workspace")
    elsewhere = tmp_path / "root"
    elsewhere.mkdir()
    (elsewhere / "USER.md").write_text("from the root")

    paths = {"CWD": str(elsewhere), "INVOKED_FROM": str(elsewhere), "WORKSPACE_DIR": str(workspace)}
    atts, _ = AgentPreparer._resolve_attachments(object.__new__(AgentPreparer), _cfg(["USER.md"]), None, paths)

    assert [a.content for a in atts] == ["from the workspace"]


def test_attachments_fall_back_to_cwd_without_a_workspace(tmp_path):
    from tsugite.agent_preparation import AgentPreparer

    (tmp_path / "USER.md").write_text("from the cwd")
    paths = {"CWD": str(tmp_path), "INVOKED_FROM": str(tmp_path), "WORKSPACE_DIR": None}
    atts, _ = AgentPreparer._resolve_attachments(object.__new__(AgentPreparer), _cfg(["USER.md"]), None, paths)

    assert [a.content for a in atts] == ["from the cwd"]
