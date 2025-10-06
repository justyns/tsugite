"""Chat session management for interactive conversations with agents."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tsugite.agent_runner import run_agent
from tsugite.custom_ui import CustomUILogger
from tsugite.md_agents import parse_agent_file


@dataclass
class ChatTurn:
    """Represents one turn in a conversation."""

    timestamp: datetime
    user_message: str
    agent_response: str
    tool_calls: List[str] = field(default_factory=list)
    token_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "user_message": self.user_message,
            "agent_response": self.agent_response,
            "tool_calls": self.tool_calls,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatTurn":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            user_message=data["user_message"],
            agent_response=data["agent_response"],
            tool_calls=data.get("tool_calls", []),
            token_count=data.get("token_count"),
        )


class ChatManager:
    """Manages chat sessions with conversation history."""

    def __init__(
        self,
        agent_path: Path,
        model_override: Optional[str] = None,
        max_history: int = 50,
        custom_logger: Optional[CustomUILogger] = None,
    ):
        """Initialize chat manager.

        Args:
            agent_path: Path to agent markdown file
            model_override: Override agent's default model
            max_history: Maximum turns to keep in context
            custom_logger: Optional custom logger for UI
        """
        self.agent_path = agent_path
        self.model_override = model_override
        self.max_history = max_history
        self.custom_logger = custom_logger
        self.conversation_history: List[ChatTurn] = []
        self.session_start = datetime.now()

    def add_turn(
        self, user_message: str, agent_response: str, tool_calls: List[str] = None, token_count: Optional[int] = None
    ) -> None:
        """Add a turn to conversation history."""
        turn = ChatTurn(
            timestamp=datetime.now(),
            user_message=user_message,
            agent_response=agent_response,
            tool_calls=tool_calls or [],
            token_count=token_count,
        )
        self.conversation_history.append(turn)

        # Prune old history if needed
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history :]

    def run_turn(self, user_input: str) -> str:
        """Execute one chat turn with the agent.

        Args:
            user_input: User's message

        Returns:
            Agent's response
        """
        try:
            result = run_agent(
                agent_path=self.agent_path,
                prompt=user_input,
                model_override=self.model_override,
                custom_logger=self.custom_logger,
                context={"chat_history": self.conversation_history},
                return_token_usage=True,
            )

            # Handle tuple return (response, token_count) or string return
            if isinstance(result, tuple):
                response, token_count = result
            else:
                response = result
                token_count = None

            self.add_turn(user_input, response, token_count=token_count)
            return response

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.add_turn(user_input, error_msg)
            return error_msg

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []

    def save_conversation(self, path: Path) -> None:
        """Save conversation to JSON file."""
        agent = parse_agent_file(self.agent_path)

        data = {
            "agent": agent.config.name or str(self.agent_path),
            "model": self.model_override or agent.config.model,
            "created_at": self.session_start.isoformat(),
            "turns": [turn.to_dict() for turn in self.conversation_history],
            "metadata": {
                "total_turns": len(self.conversation_history),
                "agent_path": str(self.agent_path),
            },
        }

        path.write_text(json.dumps(data, indent=2))

    def load_conversation(self, path: Path) -> None:
        """Load conversation from JSON file."""
        data = json.loads(path.read_text())

        self.conversation_history = []

        for turn_data in data.get("turns", []):
            turn = ChatTurn.from_dict(turn_data)
            self.conversation_history.append(turn)

        if "created_at" in data:
            self.session_start = datetime.fromisoformat(data["created_at"])

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        total_tokens = sum(turn.token_count for turn in self.conversation_history if turn.token_count)

        return {
            "total_turns": len(self.conversation_history),
            "total_tokens": total_tokens if total_tokens > 0 else None,
            "session_duration": (datetime.now() - self.session_start).total_seconds(),
            "agent": str(self.agent_path),
            "model": self.model_override or "default",
        }
