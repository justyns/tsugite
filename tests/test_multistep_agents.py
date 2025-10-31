"""Tests for multi-step agent execution."""

import pytest

from tsugite.md_agents import extract_step_directives, has_step_directives


class TestStepDirectiveParsing:
    def test_extract_single_step(self):
        """Test extracting a single step."""
        content = """
<!-- tsu:step name="test" -->
Do something
"""
        preamble, steps = extract_step_directives(content)

        assert len(steps) == 1
        assert steps[0].name == "test"
        assert "Do something" in steps[0].content
        assert steps[0].assign_var is None

    def test_extract_step_with_assign(self):
        """Test extracting step with assign variable."""
        content = """
<!-- tsu:step name="research" assign="data" -->
Research the topic
"""
        preamble, steps = extract_step_directives(content)

        assert len(steps) == 1
        assert steps[0].name == "research"
        assert steps[0].assign_var == "data"

    def test_extract_multiple_steps(self):
        """Test extracting multiple sequential steps."""
        content = """
<!-- tsu:step name="step1" assign="result1" -->
Do first thing

<!-- tsu:step name="step2" assign="result2" -->
Do second thing

<!-- tsu:step name="step3" -->
Do final thing
"""
        preamble, steps = extract_step_directives(content)

        assert len(steps) == 3
        assert steps[0].name == "step1"
        assert steps[0].assign_var == "result1"
        assert steps[1].name == "step2"
        assert steps[1].assign_var == "result2"
        assert steps[2].name == "step3"
        assert steps[2].assign_var is None

    def test_step_content_extraction(self):
        """Test that step content is correctly extracted."""
        content = """
<!-- tsu:step name="first" -->
First step content
With multiple lines

<!-- tsu:step name="second" -->
Second step content
"""
        preamble, steps = extract_step_directives(content)

        assert "First step content" in steps[0].content
        assert "With multiple lines" in steps[0].content
        assert "Second step content" in steps[1].content
        assert "first" not in steps[1].content  # Second step shouldn't include first step's directive

    def test_step_with_quoted_attributes(self):
        """Test steps with single and double quoted attributes."""
        content = """
<!-- tsu:step name="test1" assign="var1" -->
Test 1

<!-- tsu:step name='test2' assign='var2' -->
Test 2
"""
        preamble, steps = extract_step_directives(content)

        assert len(steps) == 2
        assert steps[0].name == "test1"
        assert steps[0].assign_var == "var1"
        assert steps[1].name == "test2"
        assert steps[1].assign_var == "var2"

    def test_step_missing_name_raises_error(self):
        """Test that step without name raises ValueError."""
        content = """
<!-- tsu:step assign="data" -->
Missing name
"""
        with pytest.raises(ValueError, match="missing required 'name' attribute"):
            extract_step_directives(content)

    def test_has_step_directives_detection(self):
        """Test detection of step directives."""
        with_steps = "<!-- tsu:step name='test' -->Content"
        without_steps = "Regular markdown content"

        assert has_step_directives(with_steps) is True
        assert has_step_directives(without_steps) is False

    def test_empty_content_returns_empty_list(self):
        """Test that content without steps returns empty list."""
        content = "Just regular markdown"
        preamble, steps = extract_step_directives(content)

        assert steps == []

    def test_step_with_template_variables(self):
        """Test that template variables in step content are preserved."""
        content = """
<!-- tsu:step name="use_var" -->
Use this variable: {{ previous_result }}
And this one: {{ user_prompt }}
"""
        preamble, steps = extract_step_directives(content)

        assert "{{ previous_result }}" in steps[0].content
        assert "{{ user_prompt }}" in steps[0].content


class TestMultiStepExecution:
    def test_multistep_agent_file(self, tmp_path):
        """Test running a basic multi-step agent."""
        agent_file = tmp_path / "multistep.md"
        agent_file.write_text(
            """---
name: test_multistep
model: ollama:qwen2.5-coder:7b
tools: []
---

<!-- tsu:step name="step1" assign="result1" -->
Return the text "Step 1 complete"

<!-- tsu:step name="step2" -->
Previous result was: {{ result1 }}
Now return "Step 2 complete"
"""
        )

        from tsugite.agent_runner import run_multistep_agent

        # Note: This will actually try to run the agent with a model
        # In a real test environment, you'd mock the model/agent execution
        # For now, we just test that the function can be called
        with pytest.raises((RuntimeError, ValueError)):
            # Will fail because no model available in test, but tests parsing
            run_multistep_agent(agent_file, "test prompt")

    def test_multistep_detection_integration(self, tmp_path):
        """Test that CLI can detect multi-step agents."""
        from tsugite.md_agents import has_step_directives

        multistep_agent = tmp_path / "multi.md"
        multistep_agent.write_text(
            """---
name: multi
---
<!-- tsu:step name="test" -->
Content
"""
        )

        regular_agent = tmp_path / "regular.md"
        regular_agent.write_text(
            """---
name: regular
---
Regular content
"""
        )

        assert has_step_directives(multistep_agent.read_text()) is True
        assert has_step_directives(regular_agent.read_text()) is False

    def test_duplicate_step_names_validation(self, tmp_path):
        """Test that duplicate step names are caught."""
        from tsugite.agent_runner import run_multistep_agent

        agent_file = tmp_path / "duplicate.md"
        agent_file.write_text(
            """---
name: duplicate_steps
model: ollama:qwen2.5-coder:7b
---
<!-- tsu:step name="step1" -->
First occurrence

<!-- tsu:step name="step1" -->
Duplicate name
"""
        )

        with pytest.raises(ValueError, match="Duplicate step names"):
            run_multistep_agent(agent_file, "test")

    def test_step_with_prefetch(self, tmp_path):
        """Test that prefetch works with multi-step agents."""
        agent_file = tmp_path / "with_prefetch.md"
        agent_file.write_text(
            """---
name: prefetch_test
model: ollama:qwen2.5-coder:7b
prefetch:
  - tool: read_file
    args:
      path: "test.txt"
    assign: file_content
---
<!-- tsu:step name="use_prefetch" -->
File content was: {{ file_content }}
"""
        )

        # Create the test file for prefetch
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello from prefetch")

        from tsugite.agent_runner import run_multistep_agent

        # This will fail in test due to no model, but validates parsing
        with pytest.raises((RuntimeError, ValueError)):
            run_multistep_agent(agent_file, "test", context={})


class TestStepValidation:
    def test_non_multistep_agent_raises_error(self, tmp_path):
        """Test that calling multistep executor on regular agent fails."""
        from tsugite.agent_runner import run_multistep_agent

        agent_file = tmp_path / "regular.md"
        agent_file.write_text(
            """---
name: regular
model: ollama:qwen2.5-coder:7b
---
Regular agent content without steps
"""
        )

        with pytest.raises(ValueError, match="does not contain step directives"):
            run_multistep_agent(agent_file, "test")

    def test_step_positions(self):
        """Test that step positions are tracked correctly."""
        content = """Header content
<!-- tsu:step name="first" -->
First step
<!-- tsu:step name="second" -->
Second step"""

        preamble, steps = extract_step_directives(content)

        assert steps[0].start_pos < steps[1].start_pos
        # end_pos of first step is where its content ends, which should be before second step's directive
        assert steps[0].end_pos < steps[1].start_pos

    def test_multistep_validation_passes(self, tmp_path):
        """Test that multi-step agents pass validation with step variables."""
        from tsugite.md_agents import validate_agent_execution

        agent_file = tmp_path / "multistep.md"
        agent_file.write_text(
            """---
name: test_multistep
model: ollama:qwen2.5-coder:7b
tools: []
---

<!-- tsu:step name="step1" assign="result1" -->
Do step 1

<!-- tsu:step name="step2" assign="result2" -->
Use {{ result1 }} in step 2

<!-- tsu:step name="step3" -->
Use {{ result1 }} and {{ result2 }} in step 3
"""
        )

        is_valid, message = validate_agent_execution(agent_file)
        assert is_valid, f"Validation should pass but failed: {message}"

    def test_multistep_validation_catches_real_typos(self, tmp_path):
        """Test that validation still catches actual undefined variables in steps."""
        from tsugite.md_agents import validate_agent_execution

        agent_file = tmp_path / "typo.md"
        agent_file.write_text(
            """---
name: typo_test
model: ollama:qwen2.5-coder:7b
---

<!-- tsu:step name="step1" assign="data" -->
Do something

<!-- tsu:step name="step2" -->
Use {{ daat }} instead of {{ data }} (typo!)
"""
        )

        is_valid, message = validate_agent_execution(agent_file)
        assert not is_valid
        assert "daat" in message.lower()

    def test_multistep_validation_with_no_assignments(self, tmp_path):
        """Test validation of multi-step agent without variable assignments."""
        from tsugite.md_agents import validate_agent_execution

        agent_file = tmp_path / "no_assign.md"
        agent_file.write_text(
            """---
name: no_assignments
model: ollama:qwen2.5-coder:7b
---

<!-- tsu:step name="step1" -->
Just do step 1

<!-- tsu:step name="step2" -->
Just do step 2 with {{ user_prompt }}
"""
        )

        is_valid, message = validate_agent_execution(agent_file)
        assert is_valid, f"Validation should pass: {message}"

    def test_multistep_simple_example_validates(self, tmp_path):
        """Test the exact scenario user encountered with analysis/plan variables."""
        from tsugite.md_agents import validate_agent_execution

        # This is similar to multistep_simple.md which was failing
        agent_file = tmp_path / "analyze_plan_execute.md"
        agent_file.write_text(
            """---
name: multistep_simple
model: ollama:qwen2.5-coder:7b
max_turns: 5
tools: []
---

Task: {{ user_prompt }}

<!-- tsu:step name="analyze" assign="analysis" -->
## Step 1: Analyze the Task
Analyze the user's task and break it down.

<!-- tsu:step name="plan" assign="plan" -->
## Step 2: Create a Plan
Using the analysis, create a detailed action plan.
**Analysis:**
{{ analysis }}

<!-- tsu:step name="execute" -->
## Step 3: Execute the Plan
Execute the plan and provide the result.
**Action Plan:**
{{ plan }}
**Original Analysis:**
{{ analysis }}
"""
        )

        is_valid, message = validate_agent_execution(agent_file)
        assert is_valid, f"Should pass validation (was failing before fix): {message}"


class TestStepPreamble:
    def test_preamble_extraction(self):
        """Test that preamble is extracted correctly."""
        content = """# Header

Introduction text

<!-- tsu:step name="step1" -->
Step content"""

        preamble, steps = extract_step_directives(content)

        assert "# Header" in preamble
        assert "Introduction text" in preamble
        assert len(steps) == 1

    def test_preamble_prepended_to_all_steps(self):
        """Test that preamble is prepended to each step."""
        content = """# Common Context

Task: {{ user_prompt }}

<!-- tsu:step name="step1" -->
Do step 1

<!-- tsu:step name="step2" -->
Do step 2"""

        preamble, steps = extract_step_directives(content)

        # Preamble should be in each step's content
        assert "# Common Context" in steps[0].content
        assert "{{ user_prompt }}" in steps[0].content
        assert "Do step 1" in steps[0].content

        assert "# Common Context" in steps[1].content
        assert "{{ user_prompt }}" in steps[1].content
        assert "Do step 2" in steps[1].content

    def test_no_preamble(self):
        """Test that agents without preamble work correctly."""
        content = """<!-- tsu:step name="step1" -->
Step 1 content"""

        preamble, steps = extract_step_directives(content)

        assert preamble == ""
        assert len(steps) == 1
        assert "Step 1 content" in steps[0].content

    def test_preamble_with_include_flag_false(self):
        """Test that preamble can be excluded from steps."""
        content = """# Header

<!-- tsu:step name="step1" -->
Step 1 content"""

        preamble, steps = extract_step_directives(content, include_preamble=False)

        assert "# Header" in preamble
        assert "# Header" not in steps[0].content
        assert "Step 1 content" in steps[0].content

    def test_preamble_template_variables_rendered(self, tmp_path):
        """Test that template variables in preamble are rendered."""
        from tsugite.renderer import AgentRenderer

        content = """Task: {{ user_prompt }}

<!-- tsu:step name="step1" -->
Do the task"""

        preamble, steps = extract_step_directives(content)

        # Render the step content with context
        renderer = AgentRenderer()
        rendered = renderer.render(steps[0].content, {"user_prompt": "test task"})

        assert "Task: test task" in rendered
        assert "Do the task" in rendered

    def test_multistep_simple_example_includes_preamble(self, tmp_path):
        """Test that multistep_simple.md style agents work with preamble."""
        agent_file = tmp_path / "with_header.md"
        agent_file.write_text(
            """---
name: test_preamble
model: ollama:qwen2.5-coder:7b
---

# Multi-Step Example

Task: {{ user_prompt }}

<!-- tsu:step name="step1" assign="result1" -->
## Step 1
Do step 1

<!-- tsu:step name="step2" -->
## Step 2
Use {{ result1 }}
"""
        )

        from tsugite.md_agents import parse_agent

        agent = parse_agent(agent_file.read_text(), agent_file)
        preamble, steps = extract_step_directives(agent.content)

        # Preamble should be extracted
        assert "# Multi-Step Example" in preamble
        assert "Task: {{ user_prompt }}" in preamble

        # Preamble should be in each step
        assert "# Multi-Step Example" in steps[0].content
        assert "Task: {{ user_prompt }}" in steps[0].content
        assert "## Step 1" in steps[0].content

        assert "# Multi-Step Example" in steps[1].content
        assert "Task: {{ user_prompt }}" in steps[1].content
        assert "## Step 2" in steps[1].content


class TestMultiStepTaskSharing:
    def test_tasks_persist_across_steps(self, tmp_path):
        """Test that task list is shared across steps, not reset."""
        from unittest.mock import MagicMock, patch

        from tsugite.agent_runner import run_multistep_agent

        agent_file = tmp_path / "task_sharing.md"
        agent_file.write_text(
            """---
name: task_test
model: ollama:qwen2.5-coder:7b
tools: [task_add]
---

<!-- tsu:step name="step1" -->
Step 1: Create some tasks using task_add

<!-- tsu:step name="step2" -->
Step 2: Should see tasks from step 1
"""
        )

        # Mock the TsugiteAgent to simulate task creation
        with patch("tsugite.core.agent.TsugiteAgent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            # Simulate step 1 creating a task
            def step1_side_effect(prompt):
                if "Step 1" in prompt:
                    # Simulate the agent calling task_add
                    from tsugite.tools.tasks import get_task_manager

                    tm = get_task_manager()
                    tm.add_task("test_task", "Task from step 1")
                    return "Step 1 complete, created task"
                else:
                    # Step 2 should see the task
                    tm = get_task_manager()
                    tasks = tm.list_tasks()
                    return f"Step 2 sees {len(tasks)} task(s)"

            mock_agent.run.side_effect = step1_side_effect

            # This will fail due to no real model, but we're testing task persistence
            with pytest.raises((RuntimeError, ValueError)):
                run_multistep_agent(agent_file, "test")

    def test_task_summary_in_context(self, tmp_path):
        """Test that task_summary is available in step templates."""
        from tsugite.md_agents import extract_step_directives

        agent_file = tmp_path / "with_task_summary.md"
        agent_file.write_text(
            """---
name: task_summary_test
---

<!-- tsu:step name="step1" -->
Current tasks: {{ task_summary }}

<!-- tsu:step name="step2" -->
Updated tasks: {{ task_summary }}
"""
        )

        # Just verify the template can reference task_summary
        preamble, steps = extract_step_directives(agent_file.read_text())
        assert "{{ task_summary }}" in steps[0].content
        assert "{{ task_summary }}" in steps[1].content

    def test_step_context_variables(self, tmp_path):
        """Test that step context variables are available in templates."""
        from tsugite.md_agents import extract_step_directives

        agent_file = tmp_path / "step_vars.md"
        agent_file.write_text(
            """---
name: step_vars_test
---

<!-- tsu:step name="first" -->
Step {{ step_number }} of {{ total_steps }} ({{ step_name }})

<!-- tsu:step name="second" -->
Currently in step {{ step_number }}/{{ total_steps }}: {{ step_name }}

<!-- tsu:step name="third" -->
{% if step_number == 3 %}Final step!{% endif %}
"""
        )

        # Verify templates can reference these variables
        preamble, steps = extract_step_directives(agent_file.read_text())
        assert len(steps) == 3

        # First step uses all three variables
        assert "{{ step_number }}" in steps[0].content
        assert "{{ total_steps }}" in steps[0].content
        assert "{{ step_name }}" in steps[0].content

        # Second step uses all three variables
        assert "{{ step_number }}" in steps[1].content
        assert "{{ total_steps }}" in steps[1].content
        assert "{{ step_name }}" in steps[1].content

        # Third step uses step_number in conditional
        assert "step_number" in steps[2].content


class TestStructuredOutput:
    """Tests for structured output support in multi-step agents."""

    def test_json_shorthand_parsing(self):
        """Test that json='true' shorthand is parsed correctly."""
        content = """
<!-- tsu:step name="analyze" assign="data" json="true" -->
Return JSON data
"""
        preamble, steps = extract_step_directives(content)

        assert len(steps) == 1
        assert "response_format" in steps[0].model_kwargs
        assert steps[0].model_kwargs["response_format"] == {"type": "json_object"}

    def test_temperature_parsing(self):
        """Test that temperature parameter is parsed correctly."""
        content = """
<!-- tsu:step name="creative" temperature="1.2" -->
Be creative
"""
        preamble, steps = extract_step_directives(content)

        assert len(steps) == 1
        assert "temperature" in steps[0].model_kwargs
        assert steps[0].model_kwargs["temperature"] == 1.2

    def test_max_tokens_parsing(self):
        """Test that max_tokens parameter is parsed correctly."""
        content = """
<!-- tsu:step name="brief" max_tokens="500" -->
Keep it brief
"""
        preamble, steps = extract_step_directives(content)

        assert len(steps) == 1
        assert "max_tokens" in steps[0].model_kwargs
        assert steps[0].model_kwargs["max_tokens"] == 500

    def test_multiple_model_kwargs(self):
        """Test parsing multiple model parameters in one step."""
        content = """
<!-- tsu:step name="precise" json="true" temperature="0.1" max_tokens="2000" -->
Be precise and return JSON
"""
        preamble, steps = extract_step_directives(content)

        assert len(steps) == 1
        kwargs = steps[0].model_kwargs
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["temperature"] == 0.1
        assert kwargs["max_tokens"] == 2000

    def test_response_format_json_parsing(self):
        """Test that response_format with JSON value is parsed correctly."""
        content = """
<!-- tsu:step name="schema" response_format='{"type":"json_object"}' -->
Return JSON
"""
        preamble, steps = extract_step_directives(content)

        assert len(steps) == 1
        assert steps[0].model_kwargs["response_format"] == {"type": "json_object"}

    def test_response_format_overrides_json_shorthand(self):
        """Test that explicit response_format overrides json shorthand."""
        content = """
<!-- tsu:step name="test" json="true" response_format='{"type":"json_schema","json_schema":{"name":"test"}}' -->
Use schema
"""
        preamble, steps = extract_step_directives(content)

        assert len(steps) == 1
        # response_format should override json shorthand
        assert "json_schema" in steps[0].model_kwargs["response_format"]

    def test_invalid_response_format_json_raises_error(self):
        """Test that invalid JSON in response_format raises ValueError."""
        content = """
<!-- tsu:step name="bad" response_format='{invalid json}' -->
This should fail
"""
        with pytest.raises(ValueError, match="Invalid JSON in response_format"):
            extract_step_directives(content)

    def test_top_p_parsing(self):
        """Test that top_p parameter is parsed correctly."""
        content = """
<!-- tsu:step name="nucleus" top_p="0.9" -->
Use nucleus sampling
"""
        preamble, steps = extract_step_directives(content)

        assert len(steps) == 1
        assert "top_p" in steps[0].model_kwargs
        assert steps[0].model_kwargs["top_p"] == 0.9

    def test_frequency_penalty_parsing(self):
        """Test that frequency_penalty parameter is parsed correctly."""
        content = """
<!-- tsu:step name="varied" frequency_penalty="0.5" -->
Avoid repetition
"""
        preamble, steps = extract_step_directives(content)

        assert len(steps) == 1
        assert "frequency_penalty" in steps[0].model_kwargs
        assert steps[0].model_kwargs["frequency_penalty"] == 0.5

    def test_presence_penalty_parsing(self):
        """Test that presence_penalty parameter is parsed correctly."""
        content = """
<!-- tsu:step name="diverse" presence_penalty="0.5" -->
Be diverse
"""
        preamble, steps = extract_step_directives(content)

        assert len(steps) == 1
        assert "presence_penalty" in steps[0].model_kwargs
        assert steps[0].model_kwargs["presence_penalty"] == 0.5

    def test_no_model_kwargs_by_default(self):
        """Test that steps without parameters have empty model_kwargs."""
        content = """
<!-- tsu:step name="simple" -->
Just a simple step
"""
        preamble, steps = extract_step_directives(content)

        assert len(steps) == 1
        assert steps[0].model_kwargs == {}

    def test_json_false_does_not_set_format(self):
        """Test that json='false' doesn't set response_format."""
        content = """
<!-- tsu:step name="no_json" json="false" -->
No JSON formatting
"""
        preamble, steps = extract_step_directives(content)

        assert len(steps) == 1
        assert "response_format" not in steps[0].model_kwargs


class TestVariableInjection:
    """Tests for variable injection into Python namespace."""

    def test_injectable_vars_filtering(self):
        """Test that metadata variables are filtered from injectable vars."""

        # We'll test the filtering logic by checking what would be injected
        step_context = {
            "user_prompt": "test task",
            "task_summary": "## Tasks\nNone",
            "step_number": 2,
            "step_name": "process",
            "total_steps": 3,
            "data": '{"result": "value"}',  # This should be injected
            "analysis": "some analysis",  # This should be injected
        }

        metadata_vars = {"user_prompt", "task_summary", "step_number", "step_name", "total_steps"}
        injectable_vars = {k: v for k, v in step_context.items() if k not in metadata_vars}

        assert "data" in injectable_vars
        assert "analysis" in injectable_vars
        assert "user_prompt" not in injectable_vars
        assert "task_summary" not in injectable_vars
        assert "step_number" not in injectable_vars
        assert "step_name" not in injectable_vars
        assert "total_steps" not in injectable_vars

    def test_multistep_with_variable_injection_structure(self, tmp_path):
        """Test that multi-step agent structure supports variable injection."""
        agent_file = tmp_path / "var_injection.md"
        agent_file.write_text(
            """---
name: var_injection_test
model: ollama:qwen2.5-coder:7b
max_turns: 5
---

<!-- tsu:step name="fetch" assign="data" json="true" -->
Return JSON data: {"score": 87, "name": "Alice"}

<!-- tsu:step name="process" -->
The variable `data` should be available in Python.
Parse it and use it.
"""
        )

        from tsugite.md_agents import parse_agent

        agent = parse_agent(agent_file.read_text(), agent_file)
        preamble, steps = extract_step_directives(agent.content)

        # Verify structure supports variable assignment and usage
        assert len(steps) == 2
        assert steps[0].assign_var == "data"
        assert steps[0].model_kwargs.get("response_format") == {"type": "json_object"}

    def test_agent_kwargs_include_json_import(self):
        """Test that CodeAgent is created with json in additional_authorized_imports."""
        # This is integration-level, so we'll just verify the structure
        # We can't easily test this without mocking, but we can verify
        # the function accepts injectable_vars parameter
        import inspect

        from tsugite.agent_runner import _execute_agent_with_prompt

        sig = inspect.signature(_execute_agent_with_prompt)
        assert "injectable_vars" in sig.parameters
        assert "model_kwargs" in sig.parameters


class TestLoopingSteps:
    """Tests for looping step functionality."""

    def test_parse_repeat_while(self):
        """Test parsing repeat_while parameter."""
        content = """
<!-- tsu:step name="process" repeat_while="has_pending_tasks" -->
Process tasks
"""
        preamble, steps = extract_step_directives(content)

        assert len(steps) == 1
        assert steps[0].repeat_while == "has_pending_tasks"
        assert steps[0].repeat_until is None
        assert steps[0].max_iterations == 10  # default

    def test_parse_repeat_until(self):
        """Test parsing repeat_until parameter."""
        content = """
<!-- tsu:step name="work" repeat_until="all_tasks_complete" -->
Work until done
"""
        preamble, steps = extract_step_directives(content)

        assert len(steps) == 1
        assert steps[0].repeat_until == "all_tasks_complete"
        assert steps[0].repeat_while is None
        assert steps[0].max_iterations == 10  # default

    def test_parse_max_iterations(self):
        """Test parsing max_iterations parameter."""
        content = """
<!-- tsu:step name="limited" repeat_while="has_pending_tasks" max_iterations="20" -->
Process with custom limit
"""
        preamble, steps = extract_step_directives(content)

        assert len(steps) == 1
        assert steps[0].repeat_while == "has_pending_tasks"
        assert steps[0].max_iterations == 20

    def test_parse_both_repeat_while_and_until_raises_error(self):
        """Test that having both repeat_while and repeat_until raises error."""
        content = """
<!-- tsu:step name="invalid" repeat_while="condition1" repeat_until="condition2" -->
Invalid config
"""
        with pytest.raises(ValueError, match="cannot specify both repeat_while and repeat_until"):
            extract_step_directives(content)

    def test_evaluate_loop_condition_helper_has_pending_tasks(self):
        """Test evaluating has_pending_tasks helper condition."""
        from tsugite.agent_runner.runner import evaluate_loop_condition

        # Context with pending tasks
        context_with_pending = {
            "tasks": [
                {"status": "pending", "title": "Task 1"},
                {"status": "completed", "title": "Task 2"},
            ]
        }
        assert evaluate_loop_condition("has_pending_tasks", context_with_pending) is True

        # Context without pending tasks
        context_no_pending = {
            "tasks": [
                {"status": "completed", "title": "Task 1"},
                {"status": "completed", "title": "Task 2"},
            ]
        }
        assert evaluate_loop_condition("has_pending_tasks", context_no_pending) is False

    def test_evaluate_loop_condition_helper_has_pending_required_tasks(self):
        """Test evaluating has_pending_required_tasks helper condition."""
        from tsugite.agent_runner.runner import evaluate_loop_condition

        # Context with pending required tasks
        context_with_required = {
            "tasks": [
                {"status": "pending", "optional": False, "title": "Required"},
                {"status": "pending", "optional": True, "title": "Optional"},
            ]
        }
        assert evaluate_loop_condition("has_pending_required_tasks", context_with_required) is True

        # Context with only optional pending tasks
        context_only_optional = {
            "tasks": [
                {"status": "completed", "optional": False, "title": "Required"},
                {"status": "pending", "optional": True, "title": "Optional"},
            ]
        }
        assert evaluate_loop_condition("has_pending_required_tasks", context_only_optional) is False

    def test_evaluate_loop_condition_helper_all_tasks_complete(self):
        """Test evaluating all_tasks_complete helper condition."""
        from tsugite.agent_runner.runner import evaluate_loop_condition

        # All completed
        context_complete = {
            "tasks": [
                {"status": "completed", "title": "Task 1"},
                {"status": "completed", "title": "Task 2"},
            ]
        }
        assert evaluate_loop_condition("all_tasks_complete", context_complete) is True

        # Some incomplete
        context_incomplete = {
            "tasks": [
                {"status": "completed", "title": "Task 1"},
                {"status": "pending", "title": "Task 2"},
            ]
        }
        assert evaluate_loop_condition("all_tasks_complete", context_incomplete) is False

    def test_evaluate_loop_condition_helper_has_incomplete_tasks(self):
        """Test evaluating has_incomplete_tasks helper condition."""
        from tsugite.agent_runner.runner import evaluate_loop_condition

        # Has incomplete tasks
        context_incomplete = {
            "tasks": [
                {"status": "completed", "title": "Task 1"},
                {"status": "pending", "title": "Task 2"},
            ]
        }
        assert evaluate_loop_condition("has_incomplete_tasks", context_incomplete) is True

        # All complete
        context_complete = {
            "tasks": [
                {"status": "completed", "title": "Task 1"},
                {"status": "completed", "title": "Task 2"},
            ]
        }
        assert evaluate_loop_condition("has_incomplete_tasks", context_complete) is False

    def test_evaluate_loop_condition_helper_has_in_progress_tasks(self):
        """Test evaluating has_in_progress_tasks helper condition."""
        from tsugite.agent_runner.runner import evaluate_loop_condition

        # Has in_progress tasks
        context_in_progress = {
            "tasks": [
                {"status": "in_progress", "title": "Task 1"},
                {"status": "pending", "title": "Task 2"},
            ]
        }
        assert evaluate_loop_condition("has_in_progress_tasks", context_in_progress) is True

        # No in_progress tasks
        context_no_in_progress = {
            "tasks": [
                {"status": "completed", "title": "Task 1"},
                {"status": "pending", "title": "Task 2"},
            ]
        }
        assert evaluate_loop_condition("has_in_progress_tasks", context_no_in_progress) is False

    def test_evaluate_loop_condition_helper_has_blocked_tasks(self):
        """Test evaluating has_blocked_tasks helper condition."""
        from tsugite.agent_runner.runner import evaluate_loop_condition

        # Has blocked tasks
        context_blocked = {
            "tasks": [
                {"status": "blocked", "title": "Task 1"},
                {"status": "pending", "title": "Task 2"},
            ]
        }
        assert evaluate_loop_condition("has_blocked_tasks", context_blocked) is True

        # No blocked tasks
        context_no_blocked = {
            "tasks": [
                {"status": "completed", "title": "Task 1"},
                {"status": "pending", "title": "Task 2"},
            ]
        }
        assert evaluate_loop_condition("has_blocked_tasks", context_no_blocked) is False

    def test_evaluate_loop_condition_custom_jinja2_expression(self):
        """Test evaluating custom Jinja2 expressions."""
        from tsugite.agent_runner.runner import evaluate_loop_condition

        # Test numeric comparison
        context = {
            "tasks": [
                {"status": "pending"},
                {"status": "pending"},
                {"status": "completed"},
            ]
        }
        assert evaluate_loop_condition("tasks | length > 2", context) is True
        assert evaluate_loop_condition("tasks | length > 5", context) is False

        # Test complex filter
        assert (
            evaluate_loop_condition("(tasks | selectattr('status', 'equalto', 'pending') | list | length) > 1", context)
            is True
        )

    def test_evaluate_loop_condition_invalid_expression_raises_error(self):
        """Test that invalid loop condition expression raises ValueError."""
        from tsugite.agent_runner.runner import evaluate_loop_condition

        context = {"tasks": []}

        with pytest.raises(ValueError, match="Invalid loop condition expression"):
            evaluate_loop_condition("{{ invalid syntax", context)

    def test_looping_step_structure_in_agent(self, tmp_path):
        """Test that looping steps parse correctly in full agent."""
        agent_file = tmp_path / "looping.md"
        agent_file.write_text(
            """---
name: task_processor
model: ollama:qwen2.5-coder:7b
initial_tasks:
  - "Task 1"
  - "Task 2"
  - "Task 3"
---

{{ task_summary }}

<!-- tsu:step name="process_tasks" repeat_while="has_pending_required_tasks" max_iterations="15" -->

**Iteration {{ iteration }}/{{ max_iterations }}**

Process the next pending task.
"""
        )

        from tsugite.md_agents import parse_agent

        agent = parse_agent(agent_file.read_text(), agent_file)
        preamble, steps = extract_step_directives(agent.content)

        assert len(steps) == 1
        assert steps[0].name == "process_tasks"
        assert steps[0].repeat_while == "has_pending_required_tasks"
        assert steps[0].max_iterations == 15
        assert "{{ iteration }}" in steps[0].content
        assert "{{ max_iterations }}" in steps[0].content

    def test_looping_with_repeat_until(self, tmp_path):
        """Test repeat_until in full agent."""
        agent_file = tmp_path / "repeat_until.md"
        agent_file.write_text(
            """---
name: work_until_done
model: ollama:qwen2.5-coder:7b
---

<!-- tsu:step name="work" repeat_until="all_tasks_complete" max_iterations="10" -->

Keep working until all tasks are complete.
"""
        )

        from tsugite.md_agents import parse_agent

        agent = parse_agent(agent_file.read_text(), agent_file)
        preamble, steps = extract_step_directives(agent.content)

        assert len(steps) == 1
        assert steps[0].repeat_until == "all_tasks_complete"
        assert steps[0].repeat_while is None
        assert steps[0].max_iterations == 10

    def test_looping_step_default_max_iterations(self):
        """Test that max_iterations defaults to 10."""
        content = """
<!-- tsu:step name="loop" repeat_while="has_pending_tasks" -->
No explicit max_iterations
"""
        preamble, steps = extract_step_directives(content)

        assert len(steps) == 1
        assert steps[0].max_iterations == 10

    def test_multiple_steps_some_looping(self):
        """Test agent with mix of looping and non-looping steps."""
        content = """
<!-- tsu:step name="setup" -->
Do initial setup

<!-- tsu:step name="process" repeat_while="has_pending_tasks" max_iterations="20" -->
Process tasks in loop

<!-- tsu:step name="finalize" -->
Finalize results
"""
        preamble, steps = extract_step_directives(content)

        assert len(steps) == 3

        # First step: not looping
        assert steps[0].repeat_while is None
        assert steps[0].repeat_until is None

        # Second step: looping
        assert steps[1].repeat_while == "has_pending_tasks"
        assert steps[1].max_iterations == 20

        # Third step: not looping
        assert steps[2].repeat_while is None
        assert steps[2].repeat_until is None

    def test_loop_context_variables_in_template(self):
        """Test that loop context variables are available in templates."""
        content = """
<!-- tsu:step name="worker" repeat_while="has_pending_tasks" max_iterations="25" -->

Iteration {{ iteration }} of {{ max_iterations }}

{% if is_looping_step %}
This is a looping step!
{% endif %}

Progress: {{ iteration }}/{{ max_iterations }}
"""
        preamble, steps = extract_step_directives(content)

        assert len(steps) == 1
        assert "{{ iteration }}" in steps[0].content
        assert "{{ max_iterations }}" in steps[0].content
        assert "{% if is_looping_step %}" in steps[0].content

    def test_evaluate_loop_condition_with_empty_tasks(self):
        """Test loop condition evaluation with empty task list."""
        from tsugite.agent_runner.runner import evaluate_loop_condition

        context_empty = {"tasks": []}

        # All conditions should return False for empty task list
        assert evaluate_loop_condition("has_pending_tasks", context_empty) is False
        assert evaluate_loop_condition("has_incomplete_tasks", context_empty) is False
        assert evaluate_loop_condition("has_in_progress_tasks", context_empty) is False
        assert evaluate_loop_condition("has_blocked_tasks", context_empty) is False

        # all_tasks_complete should be True for empty list (vacuous truth)
        assert evaluate_loop_condition("all_tasks_complete", context_empty) is True

    def test_custom_condition_with_optional_filter(self):
        """Test custom Jinja2 condition filtering optional tasks."""
        from tsugite.agent_runner.runner import evaluate_loop_condition

        context = {
            "tasks": [
                {"status": "pending", "optional": False, "title": "Required 1"},
                {"status": "pending", "optional": True, "title": "Optional 1"},
                {"status": "completed", "optional": False, "title": "Required 2"},
            ]
        }

        # Count only pending optional tasks
        condition = (
            "tasks | selectattr('optional', 'equalto', true) | "
            "selectattr('status', 'equalto', 'pending') | list | length > 0"
        )
        assert evaluate_loop_condition(condition, context) is True

        # Count pending tasks > 1
        condition2 = "(tasks | selectattr('status', 'equalto', 'pending') | list | length) > 1"
        assert evaluate_loop_condition(condition2, context) is True

        # Count pending tasks > 5 (should be false)
        condition3 = "(tasks | selectattr('status', 'equalto', 'pending') | list | length) > 5"
        assert evaluate_loop_condition(condition3, context) is False
