"""Tests for the retry system in multi-step agents."""

from pathlib import Path

import pytest

from tsugite.md_agents import StepDirective, extract_step_directives


class TestRetryParsing:
    """Test parsing of retry parameters from step directives."""

    def test_parse_max_retries(self):
        """Test parsing max_retries parameter."""
        content = """
<!-- tsu:step name="test" max_retries="3" -->
Step content
"""
        preamble, steps = extract_step_directives(content)
        assert len(steps) == 1
        assert steps[0].max_retries == 3
        assert steps[0].retry_delay == 0.0

    def test_parse_retry_delay(self):
        """Test parsing retry_delay parameter."""
        content = """
<!-- tsu:step name="test" retry_delay="2.5" -->
Step content
"""
        preamble, steps = extract_step_directives(content)
        assert len(steps) == 1
        assert steps[0].max_retries == 0
        assert steps[0].retry_delay == 2.5

    def test_parse_both_retry_params(self):
        """Test parsing both retry parameters."""
        content = """
<!-- tsu:step name="test" max_retries="5" retry_delay="1.0" -->
Step content
"""
        preamble, steps = extract_step_directives(content)
        assert len(steps) == 1
        assert steps[0].max_retries == 5
        assert steps[0].retry_delay == 1.0

    def test_parse_retry_with_quotes(self):
        """Test parsing with quoted values."""
        content = """
<!-- tsu:step name="test" max_retries='3' retry_delay='1.5' -->
Step content
"""
        preamble, steps = extract_step_directives(content)
        assert len(steps) == 1
        assert steps[0].max_retries == 3
        assert steps[0].retry_delay == 1.5

    def test_default_retry_values(self):
        """Test default values when retry params not specified."""
        content = """
<!-- tsu:step name="test" -->
Step content
"""
        preamble, steps = extract_step_directives(content)
        assert len(steps) == 1
        assert steps[0].max_retries == 0
        assert steps[0].retry_delay == 0.0

    def test_retry_with_other_params(self):
        """Test retry params combined with other parameters."""
        content = """
<!-- tsu:step name="test" assign="result" max_retries="2" retry_delay="0.5" temperature="0.7" -->
Step content
"""
        preamble, steps = extract_step_directives(content)
        assert len(steps) == 1
        assert steps[0].name == "test"
        assert steps[0].assign_var == "result"
        assert steps[0].max_retries == 2
        assert steps[0].retry_delay == 0.5
        assert steps[0].model_kwargs["temperature"] == 0.7


class TestRetryContextVariables:
    """Test retry context variables in templates."""

    def test_retry_context_structure(self):
        """Verify retry context variables have correct structure."""
        step = StepDirective(name="test", content="test content", max_retries=3, retry_delay=1.0)
        assert step.max_retries == 3
        assert step.retry_delay == 1.0


class TestRetryExecution:
    """Test retry execution logic."""

    @pytest.fixture
    def retry_agent_file(self, tmp_path):
        """Create a temporary agent file with retry logic."""
        agent_content = """---
name: test_retry_agent
model: ollama:qwen2.5-coder:7b
tools: [python_interpreter]
max_steps: 5
---

<!-- tsu:step name="attempt" assign="result" max_retries="2" retry_delay="0.1" -->

{% if is_retry %}
Retry attempt {{ retry_count }} of {{ max_retries }}
Previous error: {{ last_error }}
{% else %}
First attempt
{% endif %}

Use python_interpreter to execute: `final_answer("success")`
"""
        agent_path = tmp_path / "retry_agent.md"
        agent_path.write_text(agent_content)
        return agent_path

    def test_retry_agent_structure(self, retry_agent_file):
        """Test that retry agent file is structured correctly."""
        from tsugite.md_agents import parse_agent_file

        agent = parse_agent_file(retry_agent_file)
        assert agent.config.name == "test_retry_agent"
        assert agent.config.max_steps == 5

        from tsugite.md_agents import extract_step_directives, has_step_directives

        assert has_step_directives(agent.content)

        preamble, steps = extract_step_directives(agent.content)
        assert len(steps) == 1
        assert steps[0].name == "attempt"
        assert steps[0].max_retries == 2
        assert steps[0].retry_delay == 0.1
        assert steps[0].assign_var == "result"


class TestRetryExample:
    """Test the example retry agent."""

    def test_example_agent_exists(self):
        """Test that the example retry agent exists."""
        example_path = Path("agents/examples/retry_example.md")
        assert example_path.exists(), "Example retry agent should exist"

    def test_example_agent_structure(self):
        """Test the structure of the example retry agent."""
        from tsugite.md_agents import parse_agent_file

        example_path = Path("agents/examples/retry_example.md")
        if not example_path.exists():
            pytest.skip("Example agent not found")

        agent = parse_agent_file(example_path)
        assert agent.config.name == "retry_example"

        from tsugite.md_agents import extract_step_directives

        preamble, steps = extract_step_directives(agent.content)

        # Should have at least 2 steps with retry
        assert len(steps) >= 2

        # First step should have retry
        assert steps[0].max_retries > 0
        assert steps[0].retry_delay > 0

        # Check that retry context variables are used
        assert "is_retry" in steps[0].content
        assert "retry_count" in steps[0].content
        assert "last_error" in steps[0].content
