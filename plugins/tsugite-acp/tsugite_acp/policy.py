"""Permission policy for ACP request_permission callbacks.

Rules are simple strings: ``ToolName`` for an exact match, ``ToolName(glob)`` to
match the tool's primary string argument against a shell glob (fnmatch). For
Bash, the primary argument is the ``command`` key; for other tools it falls back
to the full params repr.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

Action = Literal["allow", "deny"]

_RULE_RE = re.compile(r"^(?P<tool>[A-Za-z_][A-Za-z0-9_]*)(?:\((?P<glob>.*)\))?$")


@dataclass(frozen=True)
class _Rule:
    tool: str
    glob: str | None  # None = match the tool name alone
    action: Action


def _parse_rule(rule: str, action: Action) -> _Rule:
    m = _RULE_RE.match(rule.strip())
    if not m:
        raise ValueError(f"invalid permission rule: {rule!r}")
    return _Rule(tool=m.group("tool"), glob=m.group("glob"), action=action)


def _stringify_params(params: dict) -> str:
    if "command" in params:
        return str(params["command"])
    return str(params)


@dataclass
class PermissionPolicy:
    default: Action = "allow"
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)
    _rules: list[_Rule] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        # Deny rules evaluated first so they can override an allow.
        self._rules = [_parse_rule(r, "deny") for r in self.deny] + [_parse_rule(r, "allow") for r in self.allow]

    @classmethod
    def from_config(cls, config: dict | None) -> "PermissionPolicy":
        if not config:
            return cls()
        return cls(
            default=config.get("default", "allow"),
            allow=list(config.get("allow", [])),
            deny=list(config.get("deny", [])),
        )

    def evaluate(self, tool: str, params: dict) -> Action:
        arg_str = _stringify_params(params)
        for rule in self._rules:
            if rule.tool != tool:
                continue
            if rule.glob is None:
                return rule.action
            if fnmatch.fnmatchcase(arg_str, rule.glob):
                return rule.action
        return self.default
