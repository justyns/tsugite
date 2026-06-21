"""tsu validate surfaces missing tools: a warning normally, an error when strict_tools is set."""

from typer.testing import CliRunner

from tsugite.cli import app

runner = CliRunner()


def _write(tmp_path, body):
    f = tmp_path / "a.md"
    f.write_text(body)
    return f


def test_validate_warns_on_missing_tool(tmp_path, file_tools):
    agent = _write(tmp_path, "---\nname: a\nextends: none\ntools: [read_file, no_such_tool]\n---\nBody.\n")
    result = runner.invoke(app, ["validate", str(agent)])
    assert result.exit_code == 0, result.output  # valid: missing tool is non-fatal by default
    assert "no_such_tool" in result.output  # surfaced as a warning


def test_validate_strict_tools_errors_on_missing_tool(tmp_path, file_tools):
    agent = _write(tmp_path, "---\nname: a\nextends: none\nstrict_tools: true\ntools: [no_such_tool]\n---\nBody.\n")
    result = runner.invoke(app, ["validate", str(agent)])
    assert result.exit_code == 1, result.output  # strict_tools: missing tool is fatal
    assert "no_such_tool" in result.output


def test_validate_clean_agent_has_no_tool_warning(tmp_path, file_tools):
    agent = _write(tmp_path, "---\nname: a\nextends: none\ntools: [read_file]\n---\nBody.\n")
    result = runner.invoke(app, ["validate", str(agent)])
    assert result.exit_code == 0, result.output
    assert "not installed" not in result.output
