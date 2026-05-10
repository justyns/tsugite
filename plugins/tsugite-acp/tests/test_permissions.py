"""Slices 9-10: permission policy + integration into ACPClientHandler."""

from __future__ import annotations

import pytest


class TestPermissionPolicy:
    def test_default_allow_with_no_rules(self):
        from tsugite_acp.policy import PermissionPolicy

        p = PermissionPolicy(default="allow")
        assert p.evaluate("Read", {}) == "allow"
        assert p.evaluate("Bash", {"command": "ls"}) == "allow"

    def test_default_deny_with_no_rules(self):
        from tsugite_acp.policy import PermissionPolicy

        p = PermissionPolicy(default="deny")
        assert p.evaluate("Read", {}) == "deny"

    def test_exact_tool_name_allow(self):
        from tsugite_acp.policy import PermissionPolicy

        p = PermissionPolicy(default="deny", allow=["Read", "Write"])
        assert p.evaluate("Read", {}) == "allow"
        assert p.evaluate("Write", {}) == "allow"
        assert p.evaluate("Bash", {"command": "rm"}) == "deny"

    def test_arg_glob_allow(self):
        from tsugite_acp.policy import PermissionPolicy

        p = PermissionPolicy(default="deny", allow=["Bash(git *)"])
        assert p.evaluate("Bash", {"command": "git status"}) == "allow"
        assert p.evaluate("Bash", {"command": "rm -rf /"}) == "deny"

    def test_deny_wins_over_allow(self):
        from tsugite_acp.policy import PermissionPolicy

        p = PermissionPolicy(default="allow", deny=["Bash(rm *)"], allow=["Bash(*)"])
        assert p.evaluate("Bash", {"command": "rm -rf /"}) == "deny"
        assert p.evaluate("Bash", {"command": "git status"}) == "allow"

    def test_unknown_tool_falls_through_to_default(self):
        from tsugite_acp.policy import PermissionPolicy

        p = PermissionPolicy(default="allow", deny=["Bash(*)"])
        assert p.evaluate("SomeNewTool", {}) == "allow"

        q = PermissionPolicy(default="deny", allow=["Read"])
        assert q.evaluate("SomeNewTool", {}) == "deny"

    def test_from_config_dict(self):
        from tsugite_acp.policy import PermissionPolicy

        p = PermissionPolicy.from_config(
            {
                "default": "allow",
                "allow": ["Read", "Bash(git *)"],
                "deny": ["Bash(rm *)"],
            }
        )
        assert p.evaluate("Read", {}) == "allow"
        assert p.evaluate("Bash", {"command": "git pull"}) == "allow"
        assert p.evaluate("Bash", {"command": "rm /tmp/x"}) == "deny"

    def test_from_config_none_returns_default_allow(self):
        from tsugite_acp.policy import PermissionPolicy

        p = PermissionPolicy.from_config(None)
        assert p.evaluate("Anything", {}) == "allow"


class TestHandlerPermissionRoundTrip:
    """Slice 10: agent's request_permission goes through the policy."""

    @pytest.mark.asyncio
    async def test_allow_returns_allowed_outcome_with_first_option(self):
        from acp.schema import AllowedOutcome, PermissionOption, ToolCallUpdate
        from tsugite_acp.client import ACPClientHandler
        from tsugite_acp.policy import PermissionPolicy

        handler = ACPClientHandler(policy=PermissionPolicy(default="allow"))
        options = [
            PermissionOption(option_id="allow", name="Allow", kind="allow_once"),
            PermissionOption(option_id="reject", name="Reject", kind="reject_once"),
        ]
        tool_call = ToolCallUpdate(tool_call_id="tc1", title="Read foo", raw_input={"path": "foo"})

        resp = await handler.request_permission(options=options, session_id="sess", tool_call=tool_call)
        assert isinstance(resp.outcome, AllowedOutcome)
        assert resp.outcome.option_id == "allow"

    @pytest.mark.asyncio
    async def test_deny_returns_first_reject_option(self):
        from acp.schema import AllowedOutcome, PermissionOption, ToolCallUpdate
        from tsugite_acp.client import ACPClientHandler
        from tsugite_acp.policy import PermissionPolicy

        handler = ACPClientHandler(policy=PermissionPolicy(default="deny"))
        options = [
            PermissionOption(option_id="allow", name="Allow", kind="allow_once"),
            PermissionOption(option_id="reject", name="Reject", kind="reject_once"),
        ]
        tool_call = ToolCallUpdate(tool_call_id="tc2", title="Bash rm", raw_input={"command": "rm -rf /"})

        resp = await handler.request_permission(options=options, session_id="sess", tool_call=tool_call)
        assert isinstance(resp.outcome, AllowedOutcome)
        assert resp.outcome.option_id == "reject"

    @pytest.mark.asyncio
    async def test_handler_without_policy_defaults_to_allow(self):
        from acp.schema import AllowedOutcome, PermissionOption, ToolCallUpdate
        from tsugite_acp.client import ACPClientHandler

        handler = ACPClientHandler()
        options = [
            PermissionOption(option_id="ok", name="OK", kind="allow_once"),
            PermissionOption(option_id="no", name="No", kind="reject_once"),
        ]
        tool_call = ToolCallUpdate(tool_call_id="tc3", title="Read", raw_input={})
        resp = await handler.request_permission(options=options, session_id="sess", tool_call=tool_call)
        assert isinstance(resp.outcome, AllowedOutcome)
        assert resp.outcome.option_id == "ok"
