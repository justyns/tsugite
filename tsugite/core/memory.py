"""Agent memory system.

Tracks execution history for building conversation context.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StepResult:
    """Result from a single agent step."""

    step_number: int
    thought: str
    code: str
    output: str
    error: Optional[str] = None
    tools_called: List[str] = field(default_factory=list)
    loaded_skills: Dict[str, str] = field(default_factory=dict)
    xml_observation: Optional[str] = None


@dataclass
class AgentMemory:
    """Memory of agent execution.

    Stores:
    - The task
    - All steps (thought/code/observation)
    - Reasoning content (for o1/o3/Claude)
    - Final answer
    """

    task: str = ""
    steps: List[StepResult] = field(default_factory=list)
    reasoning_history: List[str] = field(default_factory=list)
    final_answer: Optional[Any] = None

    def add_task(self, task: str) -> None:
        """Set the task."""
        self.task = task

    def add_step(
        self,
        thought: str,
        code: str,
        output: str,
        error: Optional[str] = None,
        tools_called: Optional[List[str]] = None,
        loaded_skills: Optional[Dict[str, str]] = None,
        xml_observation: Optional[str] = None,
    ) -> None:
        """Add a step to history."""
        step = StepResult(
            step_number=len(self.steps) + 1,
            thought=thought,
            code=code,
            output=output,
            error=error,
            tools_called=tools_called or [],
            loaded_skills=loaded_skills or {},
            xml_observation=xml_observation,
        )
        self.steps.append(step)

    def add_reasoning(self, reasoning: str) -> None:
        """Add reasoning content (from o1/o3/Claude thinking)."""
        self.reasoning_history.append(reasoning)

    def add_final_answer(self, answer: Any) -> None:
        """Set final answer."""
        self.final_answer = answer
