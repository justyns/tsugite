"""Tests for agent memory system."""

from tsugite.core.memory import AgentMemory, StepResult


def test_step_result_creation():
    """Test creating a StepResult."""
    step = StepResult(
        step_number=1,
        thought="I need to calculate this",
        code="result = 5 + 3",
        output="8",
        error=None,
    )

    assert step.step_number == 1
    assert step.thought == "I need to calculate this"
    assert step.code == "result = 5 + 3"
    assert step.output == "8"
    assert step.error is None


def test_step_result_with_error():
    """Test StepResult with error."""
    step = StepResult(
        step_number=2,
        thought="Try to divide",
        code="result = 10 / 0",
        output="",
        error="ZeroDivisionError: division by zero",
    )

    assert step.error == "ZeroDivisionError: division by zero"


def test_agent_memory_initialization():
    """Test AgentMemory initialization."""
    memory = AgentMemory()

    assert memory.task == ""
    assert memory.steps == []
    assert memory.reasoning_history == []
    assert memory.final_answer is None


def test_add_task():
    """Test adding task to memory."""
    memory = AgentMemory()
    memory.add_task("Calculate 5 + 3")

    assert memory.task == "Calculate 5 + 3"


def test_add_step():
    """Test adding step to memory."""
    memory = AgentMemory()

    memory.add_step(
        thought="I'll calculate this",
        code="result = 5 + 3",
        output="8",
        error=None,
    )

    assert len(memory.steps) == 1
    step = memory.steps[0]
    assert step.step_number == 1
    assert step.thought == "I'll calculate this"
    assert step.code == "result = 5 + 3"
    assert step.output == "8"


def test_add_multiple_steps():
    """Test adding multiple steps."""
    memory = AgentMemory()

    memory.add_step("First thought", "code1", "output1")
    memory.add_step("Second thought", "code2", "output2")
    memory.add_step("Third thought", "code3", "output3")

    assert len(memory.steps) == 3
    assert memory.steps[0].step_number == 1
    assert memory.steps[1].step_number == 2
    assert memory.steps[2].step_number == 3


def test_add_reasoning():
    """Test adding reasoning content (for o1/o3/Claude)."""
    memory = AgentMemory()

    memory.add_reasoning("Step 1 reasoning: analyzing the problem...")
    memory.add_reasoning("Step 2 reasoning: considering different approaches...")

    assert len(memory.reasoning_history) == 2
    assert "analyzing the problem" in memory.reasoning_history[0]
    assert "considering different approaches" in memory.reasoning_history[1]


def test_add_final_answer():
    """Test adding final answer."""
    memory = AgentMemory()

    memory.add_final_answer("The answer is 42")

    assert memory.final_answer == "The answer is 42"


def test_complete_workflow():
    """Test complete memory workflow."""
    memory = AgentMemory()

    # Set task
    memory.add_task("Calculate factorial of 5")

    # Add reasoning (for reasoning models)
    memory.add_reasoning("Thinking about recursive vs iterative approach...")

    # Add steps
    memory.add_step(
        thought="Define factorial function",
        code="def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
        output="Function defined",
    )

    memory.add_step(
        thought="Calculate factorial(5)",
        code="result = factorial(5)",
        output="120",
    )

    # Set final answer
    memory.add_final_answer("120")

    # Verify complete state
    assert memory.task == "Calculate factorial of 5"
    assert len(memory.steps) == 2
    assert len(memory.reasoning_history) == 1
    assert memory.final_answer == "120"


def test_step_with_error_in_workflow():
    """Test handling steps with errors."""
    memory = AgentMemory()

    memory.add_task("Try division")

    # First step fails
    memory.add_step(
        thought="Divide by zero",
        code="result = 10 / 0",
        output="",
        error="ZeroDivisionError: division by zero",
    )

    # Second step succeeds
    memory.add_step(
        thought="Divide by non-zero",
        code="result = 10 / 2",
        output="5.0",
        error=None,
    )

    assert len(memory.steps) == 2
    assert memory.steps[0].error is not None
    assert memory.steps[1].error is None
