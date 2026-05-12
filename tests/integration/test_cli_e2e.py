"""Integration tests that exercise the full CLI binary via subprocess."""

import subprocess


def _tsu(*args, timeout=120):
    """Run `uv run tsu ...` and return CompletedProcess."""
    return subprocess.run(
        ["uv", "run", "tsu", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class TestRunCommand:
    def test_simple_run(self, agent_file):
        agent = agent_file(name="cli_run", tools=[])
        result = _tsu("run", str(agent), "What is 2+2? Call return_value with just the number.", "--plain")

        assert result.returncode == 0
        assert "4" in result.stdout


class TestRenderCommand:
    def test_render_output(self, agent_file):
        agent = agent_file(name="cli_render")
        result = _tsu("render", str(agent), "test prompt")

        assert result.returncode == 0
        assert "return_value" in result.stdout or "tools" in result.stdout.lower()


class TestValidateCommand:
    def test_valid_agent(self, agent_file):
        agent = agent_file(name="cli_valid")
        result = _tsu("validate", str(agent))

        assert result.returncode == 0

    def test_invalid_agent(self, tmp_path):
        bad = tmp_path / "bad.md"
        bad.write_text("---\nunknown_field_xyz: true\n---\nno name")
        result = _tsu("validate", str(bad))

        assert result.returncode != 0
