"""Agents: named identities with their own workspace, persisted in daemon.db (`agents` collection)."""

import re
import threading
from dataclasses import asdict, dataclass, replace
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from tsugite.core.record_store import SqliteCollectionStorage
from tsugite.renderer import slugify

if TYPE_CHECKING:
    from tsugite_daemon.session_store import Session

AGENT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

EDITABLE_FIELDS = frozenset({"name", "workspace", "template", "model", "reasoning_effort", "color"})
NULLABLE_FIELDS = frozenset({"model", "reasoning_effort", "color"})

AGENT_COLORS = ("blue", "green", "red", "purple", "orange", "yellow", "teal", "pink")


class UnknownAgentError(ValueError):
    pass


def validate_agent_id(agent_id: str) -> None:
    if not AGENT_ID_PATTERN.fullmatch(agent_id):
        raise ValueError(
            f"Invalid agent id {agent_id!r}: use lowercase letters, digits, '-' and '_' (max 64 characters)"
        )


@dataclass
class Agent:
    id: str
    name: str
    workspace: str
    template: str
    model: Optional[str] = None
    reasoning_effort: Optional[str] = None
    color: Optional[str] = None
    is_default: bool = False
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class AgentStore:
    def __init__(self, path: Path):
        self.state_dir = path.parent
        self._storage = SqliteCollectionStorage.for_state_file(path, "agents")
        self._lock = threading.Lock()
        valid = {f.name for f in dataclass_fields(Agent)}
        self._agents: dict[str, Agent] = {}
        for entry in self._storage.load_all():
            agent = Agent(**{k: v for k, v in entry.items() if k in valid})
            self._agents[agent.id] = agent

    def list(self) -> list[Agent]:
        with self._lock:
            return list(self._agents.values())

    def get(self, agent_id: str) -> Optional[Agent]:
        with self._lock:
            return self._agents.get(agent_id)

    def find_default(self) -> Optional[Agent]:
        with self._lock:
            return next((a for a in self._agents.values() if a.is_default), None)

    def default(self) -> Agent:
        agent = self.find_default()
        if agent is None:
            raise RuntimeError("No default agent: ensure_default has not run")
        return agent

    def effective_id(self, agent_id: Optional[str]) -> str:
        """`agent_id`, or the default agent's id when it is None."""
        return agent_id or self.default().id

    def for_session(self, session: Optional["Session"]) -> Agent:
        """The session's agent, or the default agent when the session has none."""
        if session is None or not session.agent_id:
            return self.default()
        agent = self.get(session.agent_id)
        if agent is None:
            raise UnknownAgentError(f"Unknown agent {session.agent_id!r} for session {session.id}")
        return agent

    def _save(self, agent: Agent) -> None:
        if agent.color is not None and agent.color not in AGENT_COLORS:
            raise ValueError(f"Invalid color {agent.color!r}: use one of {', '.join(AGENT_COLORS)}")
        self._storage.upsert(agent.id, asdict(agent))
        self._agents[agent.id] = agent

    def create(self, agent: Agent) -> Agent:
        validate_agent_id(agent.id)
        with self._lock:
            if agent.id in self._agents:
                raise ValueError(f"Agent already exists: {agent.id}")
            self._save(agent)
        return agent

    def update(self, agent_id: str, **fields) -> Agent:
        unknown = set(fields) - EDITABLE_FIELDS
        if unknown:
            raise ValueError(f"Unknown agent field(s): {', '.join(sorted(unknown))}")
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent is None:
                raise ValueError(f"Unknown agent: {agent_id}")
            agent = replace(agent, **fields)
            self._save(agent)
        return agent

    def ensure_default(self, workspace: Path, template: str) -> None:
        """Create the default agent from the runtime workspace and agent file, or update it to match them."""
        default = self.find_default()
        if default is None:
            agent_id = slugify(workspace.name)
            if not AGENT_ID_PATTERN.fullmatch(agent_id):
                agent_id = "default"
            self.create(
                Agent(
                    id=agent_id,
                    name=workspace.name,
                    workspace=str(workspace),
                    template=template,
                    color="blue",
                    is_default=True,
                )
            )
        elif (default.workspace, default.template) != (str(workspace), template):
            self.update(default.id, workspace=str(workspace), template=template)

    def identity_dir(self, session: Optional["Session"]) -> Path:
        """The workspace holding the session's identity, memory and skills."""
        return Path(self.for_session(session).workspace)

    def working_dir(self, session: Optional["Session"]) -> Path:
        """Where the session's code and file tools run: its `workspace_override`, else the identity workspace."""
        if session is not None and session.workspace_override:
            return Path(session.workspace_override)
        return self.identity_dir(session)
