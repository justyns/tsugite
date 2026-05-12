"""Integration tests for history save/load with a live LLM.

Note: run_agent() does not save history -- that happens in the CLI layer.
These tests call save_run_to_history() explicitly to test the storage round-trip.

History isolation comes from the autouse `_isolate_data_dirs` fixture in conftest.py.
"""

from tsugite.agent_runner.history_integration import save_run_to_history
from tsugite.history.storage import list_session_files

from .conftest import INTEGRATION_MODEL, run_integration_agent


class TestHistoryRoundTrip:
    def test_history_file_created(self, agent_file):
        agent = agent_file(name="hist_save", tools=[])
        result = run_integration_agent(agent, "Call return_value with 'hello'.")

        session_id = save_run_to_history(
            agent_path=agent,
            agent_name="hist_save",
            prompt="Call return_value with 'hello'.",
            result=result,
            model=INTEGRATION_MODEL,
        )

        assert session_id is not None
        sessions = list_session_files()
        assert len(sessions) >= 1
        assert "hello" in sessions[0].read_text().lower()

    def test_continuation(self, agent_file):
        agent = agent_file(name="hist_cont", tools=[])
        result1 = run_integration_agent(agent, "Call return_value with 'first run'.")

        session_id = save_run_to_history(
            agent_path=agent,
            agent_name="hist_cont",
            prompt="Call return_value with 'first run'.",
            result=result1,
            model=INTEGRATION_MODEL,
        )
        assert session_id is not None

        result2 = run_integration_agent(
            agent,
            "What did I say in the previous message? Call return_value with your answer.",
            continue_conversation_id=session_id,
        )

        assert "first" in result2.lower() or "run" in result2.lower()
