"""Integration tests for real tool execution with a live LLM.

Each test sends a prompt to a real model, which calls real tools.
Assertions verify both the LLM's final answer AND the actual filesystem state.
"""

from .conftest import run_integration_agent


class TestFileOperations:
    def test_write_and_read_file(self, agent_file, work_dir):
        agent = agent_file(name="file_rw")
        result = run_integration_agent(
            agent, "Write 'hello world' to test.txt, read it back, return its contents via return_value."
        )

        assert "hello world" in result.lower()
        written = work_dir / "test.txt"
        assert written.exists(), "Agent should have created test.txt on disk"
        assert "hello world" in written.read_text().lower()

    def test_list_files(self, agent_file, work_dir):
        agent = agent_file(name="list_files")
        result = run_integration_agent(
            agent,
            "Write a file called marker.txt containing 'mark', then list files in the current directory "
            "and return the list via return_value.",
        )

        assert "marker.txt" in result
        marker = work_dir / "marker.txt"
        assert marker.exists()
        assert "mark" in marker.read_text()


class TestShellExecution:
    def test_echo(self, agent_file):
        agent = agent_file(name="shell_echo", tools=["run"])
        result = run_integration_agent(agent, "Run 'echo 42' in the shell and return the output via return_value.")

        assert "42" in result


class TestSearchTools:
    def test_search_code(self, agent_file, work_dir):
        agent = agent_file(
            name="search_code",
            tools=["write_file", "read_file", "run"],
        )
        result = run_integration_agent(
            agent,
            "Write a file test.py containing 'def greet(): pass', then use shell grep to search "
            "for 'def greet' in the current directory, and return the matches via return_value.",
        )

        assert "greet" in result
        written = work_dir / "test.py"
        assert written.exists()
        assert "def greet" in written.read_text()


class TestErrorHandling:
    def test_read_nonexistent_file(self, agent_file, work_dir):
        agent = agent_file(name="error_recovery")
        result = run_integration_agent(
            agent,
            "Try to read a file called nonexistent_file.txt. If it fails, call return_value with exactly 'not found'.",
        )

        assert "not found" in result.lower()
        assert not (work_dir / "nonexistent_file.txt").exists()


class TestReasoning:
    def test_multi_turn_calculation(self, agent_file):
        agent = agent_file(name="reasoning", tools=["run"], max_turns=10)
        result = run_integration_agent(
            agent, "Calculate the factorial of 6 step by step, then call return_value with the number."
        )

        assert "720" in result
