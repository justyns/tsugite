"""Integration tests for multi-step agent pipelines with a live LLM."""

from tsugite.agent_runner.runner import run_multistep_agent
from tsugite.options import ExecutionOptions


class TestMultiStepPipeline:
    def test_two_step_pipeline(self, agent_file, work_dir):
        body = """\
<!-- tsu:step name="research" assign="research_data" -->
Write a file called notes.txt containing 'research complete'. Then call return_value with 'done'.

<!-- tsu:step name="summarize" -->
The research step produced: {{ research_data }}

Read notes.txt and call return_value confirming the file exists and what it contains.
"""
        agent = agent_file(name="pipeline", body=body)
        result = run_multistep_agent(
            agent_path=agent,
            prompt="Run the two-step pipeline",
            exec_options=ExecutionOptions(return_token_usage=False),
        )

        assert (work_dir / "notes.txt").exists()
        assert "research" in result.lower() or "notes" in result.lower()

    def test_variable_passing(self, agent_file):
        body = """\
<!-- tsu:step name="generate" assign="magic_number" -->
Call return_value with exactly the text '42'.

<!-- tsu:step name="verify" -->
The previous step returned: {{ magic_number }}

Call return_value confirming the magic number is 42.
"""
        agent = agent_file(name="varpass", body=body)
        result = run_multistep_agent(
            agent_path=agent,
            prompt="Test variable passing",
            exec_options=ExecutionOptions(return_token_usage=False),
        )

        assert "42" in result
