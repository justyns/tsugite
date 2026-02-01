"""Tests for chat error handling."""

from unittest.mock import patch

from tsugite.ui.chat import ChatManager


def test_run_turn_catches_and_returns_errors(tmp_path):
    """Test that run_turn catches exceptions and returns error messages."""
    # Create a minimal agent file
    agent_file = tmp_path / "test_agent.md"
    agent_file.write_text(
        """---
name: test-agent
extends: none
---

Test agent
"""
    )

    manager = ChatManager(
        agent_path=agent_file,
        model_override="invalid-model-that-does-not-exist",
        disable_history=True,
    )

    # Mock run_agent to raise an exception
    # Note: run_agent is imported inside run_turn(), so we patch it in agent_runner
    with patch("tsugite.agent_runner.run_agent") as mock_run_agent:
        mock_run_agent.side_effect = Exception("Invalid model: invalid-model-that-does-not-exist")

        # run_turn should catch the exception and return an error message
        response = manager.run_turn("test input")

        # Verify error message is returned
        assert response.startswith("Error:")
        assert "Invalid model" in response


def test_error_message_added_to_history(tmp_path):
    """Test that error messages are added to conversation history."""
    agent_file = tmp_path / "test_agent.md"
    agent_file.write_text(
        """---
name: test-agent
extends: none
---

Test agent
"""
    )

    manager = ChatManager(
        agent_path=agent_file,
        model_override="invalid-model",
        disable_history=True,
    )

    with patch("tsugite.agent_runner.run_agent") as mock_run_agent:
        mock_run_agent.side_effect = Exception("Model not found")

        manager.run_turn("test input")

        # Verify error is in history
        assert len(manager.conversation_history) == 1
        turn = manager.conversation_history[0]
        assert turn.user_message == "test input"
        assert turn.agent_response.startswith("Error:")
