"""Tests for Docker wrapper delegation."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from tsugite.cli import app

runner = CliRunner()


class TestDockerDelegation:
    """Test that --docker flag delegates to tsugite-docker wrapper."""

    def test_docker_flag_triggers_delegation(self, tmp_path):
        """Test that --docker flag attempts to delegate to wrapper."""
        agent_file = tmp_path / "test_agent.md"
        agent_file.write_text(
            """---
name: test
model: openai:gpt-4o-mini
---

Test agent
"""
        )

        with patch("shutil.which") as mock_which:
            mock_which.return_value = None  # Wrapper not in PATH

            result = runner.invoke(
                app,
                ["run", "--docker", str(agent_file), "test prompt"],
                catch_exceptions=False,
            )

            # Should fail with helpful error message
            assert result.exit_code == 1
            assert "tsugite-docker wrapper not found" in result.stdout
            assert "bin/README.md" in result.stdout

    def test_docker_delegation_with_wrapper_available(self, tmp_path):
        """Test delegation when wrapper is available."""
        agent_file = tmp_path / "test_agent.md"
        agent_file.write_text(
            """---
name: test
model: openai:gpt-4o-mini
---

Test agent
"""
        )

        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/local/bin/tsugite-docker"

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                result = runner.invoke(
                    app,
                    ["run", "--docker", str(agent_file), "test prompt"],
                    catch_exceptions=False,
                )

                # Should delegate to wrapper
                assert result.exit_code == 0
                mock_run.assert_called_once()

                # Check command structure
                call_args = mock_run.call_args[0][0]
                assert call_args[0] == "tsugite-docker"
                assert "run" in call_args
                assert "--docker" not in call_args  # Docker flag filtered out

    def test_container_flag_triggers_delegation(self, tmp_path):
        """Test that --container flag also delegates."""
        agent_file = tmp_path / "test_agent.md"
        agent_file.write_text(
            """---
name: test
model: openai:gpt-4o-mini
---

Test agent
"""
        )

        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/local/bin/tsugite-docker"

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                result = runner.invoke(
                    app,
                    ["run", "--container", "my-session", str(agent_file), "test prompt"],
                    catch_exceptions=False,
                )

                # Should delegate to wrapper
                assert result.exit_code == 0
                mock_run.assert_called_once()

                # Check command includes container name
                call_args = mock_run.call_args[0][0]
                assert call_args[0] == "tsugite-docker"
                assert "--container" in call_args
                assert "my-session" in call_args

    def test_network_flag_forwarded(self, tmp_path):
        """Test that --network flag is forwarded to wrapper."""
        agent_file = tmp_path / "test_agent.md"
        agent_file.write_text(
            """---
name: test
model: openai:gpt-4o-mini
---

Test agent
"""
        )

        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/local/bin/tsugite-docker"

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                result = runner.invoke(
                    app,
                    ["run", "--docker", "--network", "none", str(agent_file), "test prompt"],
                    catch_exceptions=False,
                )

                # Should delegate with network flag
                assert result.exit_code == 0
                call_args = mock_run.call_args[0][0]
                assert "--network" in call_args
                assert "none" in call_args

    def test_keep_flag_forwarded(self, tmp_path):
        """Test that --keep flag is forwarded to wrapper."""
        agent_file = tmp_path / "test_agent.md"
        agent_file.write_text(
            """---
name: test
model: openai:gpt-4o-mini
---

Test agent
"""
        )

        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/local/bin/tsugite-docker"

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                result = runner.invoke(
                    app,
                    ["run", "--docker", "--keep", str(agent_file), "test prompt"],
                    catch_exceptions=False,
                )

                # Should delegate with keep flag
                assert result.exit_code == 0
                call_args = mock_run.call_args[0][0]
                assert "--keep" in call_args

    def test_tsugite_flags_preserved(self, tmp_path):
        """Test that tsugite-specific flags are preserved during delegation."""
        agent_file = tmp_path / "test_agent.md"
        agent_file.write_text(
            """---
name: test
model: openai:gpt-4o-mini
---

Test agent
"""
        )

        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/local/bin/tsugite-docker"

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                result = runner.invoke(
                    app,
                    ["run", "--docker", str(agent_file), "test prompt", "--debug", "--verbose"],
                    catch_exceptions=False,
                )

                # Tsugite flags should be preserved
                assert result.exit_code == 0
                call_args = mock_run.call_args[0][0]
                assert "--debug" in call_args
                assert "--verbose" in call_args

    def test_no_delegation_without_docker_flags(self, tmp_path):
        """Test that normal execution works without Docker flags."""
        agent_file = tmp_path / "test_agent.md"
        agent_file.write_text(
            """---
name: test
model: openai:gpt-4o-mini
---

Test agent. Use final_answer() to complete.
"""
        )

        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/local/bin/tsugite-docker"

            with patch("subprocess.run") as mock_run:
                # This should NOT be called for Docker delegation
                runner.invoke(
                    app,
                    ["run", str(agent_file), "test prompt"],
                    catch_exceptions=False,
                )

                # Should execute normally without calling tsugite-docker
                # subprocess.run might be called for other things (uname, etc) but NOT for tsugite-docker
                for call in mock_run.call_args_list:
                    if call[0]:  # If positional args exist
                        cmd = call[0][0]
                        if isinstance(cmd, list) and len(cmd) > 0:
                            assert cmd[0] != "tsugite-docker", (
                                "Should not delegate to tsugite-docker without --docker flag"
                            )
