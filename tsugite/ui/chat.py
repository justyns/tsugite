"""Chat session management for interactive conversations with agents."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from tsugite.config import load_config
from tsugite.md_agents import parse_agent_file
from tsugite.ui import CustomUILogger


class ChatTurn(BaseModel):
    """Represents one turn in a conversation."""

    model_config = ConfigDict(
        extra="forbid",
    )

    timestamp: datetime
    user_message: str
    agent_response: str
    tool_calls: List[str] = Field(default_factory=list)
    token_count: Optional[int] = None
    cost: Optional[float] = None


class ChatManager:
    """Manages chat sessions with conversation history."""

    def __init__(
        self,
        agent_path: Path,
        model_override: Optional[str] = None,
        max_history: int = 50,
        custom_logger: Optional[CustomUILogger] = None,
        stream: bool = False,
        disable_history: bool = False,
        resume_conversation_id: Optional[str] = None,
    ):
        """Initialize chat manager.

        Args:
            agent_path: Path to agent markdown file
            model_override: Override agent's default model
            max_history: Maximum turns to keep in context
            custom_logger: Optional custom logger for UI
            stream: Whether to stream responses in real-time
            disable_history: Disable conversation history persistence
            resume_conversation_id: Optional conversation ID to resume (skips auto-generation)
        """
        self.agent_path = agent_path
        self.model_override = model_override
        self.max_history = max_history
        self.custom_logger = custom_logger
        self.stream = stream
        self.conversation_history: List[ChatTurn] = []
        self.session_start = datetime.now()

        # Cumulative metrics tracking
        self.cumulative_tokens = 0
        self.cumulative_cost = 0.0

        # Load agent and tools for /tools command
        self.available_tools: List[Any] = []
        try:
            agent = parse_agent_file(agent_path)
            self._agent_config = agent.config
            if agent.config.tools:
                from tsugite.core.tools import create_tool_from_tsugite
                from tsugite.tools import expand_tool_specs

                expanded = expand_tool_specs(agent.config.tools)
                self.available_tools = [create_tool_from_tsugite(name) for name in expanded]
        except Exception:
            # Don't fail if tools can't be loaded
            pass

        # History support
        self.conversation_id: Optional[str] = resume_conversation_id
        config = load_config()
        history_enabled = getattr(config, "history_enabled", True) and not disable_history

        # Only create new conversation if not resuming
        if history_enabled and not resume_conversation_id:
            try:
                from tsugite.ui.chat_history import start_conversation

                agent = parse_agent_file(agent_path)
                model = model_override or agent.config.model or "unknown"

                self.conversation_id = start_conversation(
                    agent_name=agent.config.name or agent_path.stem,
                    model=model,
                    timestamp=self.session_start,
                )
            except Exception as e:
                # Don't fail if history can't be initialized
                print(f"Warning: Failed to initialize conversation history: {e}")
                self.conversation_id = None

    def load_from_history(self, conversation_id: str, turns: List[Any]) -> None:
        """Load conversation history from JSONL storage.

        Args:
            conversation_id: Conversation ID to resume
            turns: List of Turn objects from history

        Raises:
            RuntimeError: If loading fails
        """
        try:
            from tsugite.history import Turn

            self.conversation_id = conversation_id
            self.conversation_history = []

            # Convert Turn objects from history to ChatTurn objects
            for turn in turns:
                if not isinstance(turn, Turn):
                    continue

                chat_turn = ChatTurn(
                    timestamp=turn.timestamp,
                    user_message=turn.user,
                    agent_response=turn.assistant,
                    tool_calls=turn.tools or [],
                    token_count=turn.tokens,
                    cost=turn.cost,
                )
                self.conversation_history.append(chat_turn)

            # Update session_start to first turn's timestamp if available
            if self.conversation_history:
                # Strip timezone to match datetime.now() (which is naive)
                timestamp = self.conversation_history[0].timestamp
                self.session_start = timestamp.replace(tzinfo=None) if timestamp.tzinfo else timestamp

            # Restore cumulative metrics from loaded history
            self.cumulative_tokens = sum(turn.token_count for turn in self.conversation_history if turn.token_count)
            self.cumulative_cost = sum(turn.cost for turn in self.conversation_history if turn.cost)

            # Prune if history exceeds max_history
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history :]

        except Exception as e:
            raise RuntimeError(f"Failed to load conversation from history: {e}")

    def add_turn(
        self,
        user_message: str,
        agent_response: str,
        tool_calls: List[str] = None,
        token_count: Optional[int] = None,
        cost: Optional[float] = None,
        execution_steps: Optional[list] = None,
        messages: Optional[list] = None,
    ) -> None:
        """Add a turn to conversation history."""
        turn = ChatTurn(
            timestamp=datetime.now(),
            user_message=user_message,
            agent_response=agent_response,
            tool_calls=tool_calls or [],
            token_count=token_count,
            cost=cost,
        )
        self.conversation_history.append(turn)

        # Update cumulative metrics
        if token_count is not None:
            self.cumulative_tokens += token_count
        if cost is not None:
            self.cumulative_cost += cost

        # Save to persistent history if enabled
        if self.conversation_id:
            try:
                from tsugite.ui.chat_history import save_chat_turn

                save_chat_turn(
                    conversation_id=self.conversation_id,
                    user_message=user_message,
                    agent_response=agent_response,
                    tool_calls=tool_calls or [],
                    token_count=token_count,
                    cost=cost,
                    timestamp=turn.timestamp,
                    execution_steps=execution_steps,
                    messages=messages,
                )
            except Exception as e:
                # Don't fail the turn if history save fails
                print(f"Warning: Failed to save turn to history: {e}")

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
        # Import here to avoid circular dependency
        from tsugite.agent_runner import run_agent

        try:
            from tsugite.options import ExecutionOptions

            result = run_agent(
                agent_path=self.agent_path,
                prompt=user_input,
                exec_options=ExecutionOptions(
                    model_override=self.model_override,
                    return_token_usage=True,
                    stream=self.stream,
                    force_text_mode=True,  # Enable text mode for chat UI
                ),
                custom_logger=self.custom_logger,
                context={"chat_history": self.conversation_history},
                continue_conversation_id=self.conversation_id if self.conversation_history else None,
            )

            # Handle result - either string or AgentExecutionResult model
            from tsugite.agent_runner.models import AgentExecutionResult

            if isinstance(result, AgentExecutionResult):
                # New model-based return
                response = result.response
                token_count = result.token_count
                cost = result.cost
            elif isinstance(result, str):
                # Simple string return (shouldn't happen with return_token_usage=True)
                response = result
                token_count = None
                cost = None
            else:
                # Unexpected type, try to convert to string
                response = str(result)
                token_count = None
                cost = None

            self.add_turn(user_input, response, token_count=token_count, cost=cost)

            # Emit cumulative metrics event if we have a custom logger
            if self.custom_logger and self.custom_logger.ui_handler:
                from tsugite.events import CostSummaryEvent, EventBus

                # Get model name for context limit warnings
                model_name = self.model_override
                if not model_name:
                    agent = parse_agent_file(self.agent_path)
                    model_name = agent.config.model

                EventBus().emit(
                    CostSummaryEvent(
                        tokens=token_count,
                        cost=cost,
                        model=model_name,
                        cumulative_tokens=self.cumulative_tokens,
                        cumulative_cost=self.cumulative_cost,
                    )
                )

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
            "turns": [turn.model_dump(mode="json") for turn in self.conversation_history],
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
            turn = ChatTurn.model_validate(turn_data)
            self.conversation_history.append(turn)

        if "created_at" in data:
            # Parse datetime and strip timezone to match datetime.now() (which is naive)
            self.session_start = datetime.fromisoformat(data["created_at"]).replace(tzinfo=None)

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        total_tokens = sum(turn.token_count for turn in self.conversation_history if turn.token_count)
        total_cost = sum(turn.cost for turn in self.conversation_history if turn.cost)

        return {
            "total_turns": len(self.conversation_history),
            "total_tokens": total_tokens if total_tokens > 0 else None,
            "total_cost": total_cost if total_cost > 0 else None,
            "session_duration": (datetime.now() - self.session_start).total_seconds(),
            "agent": str(self.agent_path),
            "model": self.model_override or "default",
        }
