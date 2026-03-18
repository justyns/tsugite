"""Integration tests for usage tracking across the execution pipeline."""

from unittest.mock import patch

import pytest

from tsugite.usage import UsageRecord, UsageStore, reset_usage_store


@pytest.fixture(autouse=True)
def reset_store():
    """Reset the usage store singleton between tests."""
    reset_usage_store()
    yield
    reset_usage_store()


@pytest.fixture
def usage_store(tmp_path):
    """Create a fresh usage store and patch the singleton."""
    store = UsageStore(tmp_path / "usage.db")
    with patch("tsugite.usage.store.get_usage_store", return_value=store):
        yield store


# ── save_run_to_history records to usage store ──


class TestSaveRunToHistoryUsage:
    def test_records_usage_on_save(self, tmp_path, monkeypatch):
        """save_run_to_history should record usage to the SQLite store."""
        store = UsageStore(tmp_path / "usage.db")
        history_dir = tmp_path / "history"
        history_dir.mkdir()

        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: history_dir)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test")

        # Patch get_usage_store at the module where it's imported
        with patch("tsugite.usage.get_usage_store", return_value=store):
            from tsugite.agent_runner.history_integration import save_run_to_history

            # Create a minimal agent file
            agent_file = tmp_path / "test_agent.md"
            agent_file.write_text("---\nname: test_agent\nmodel: openai:gpt-4o\n---\nHello")

            save_run_to_history(
                agent_path=agent_file,
                agent_name="test_agent",
                prompt="test prompt",
                result="test result",
                model="openai:gpt-4o",
                token_count=1500,
                input_tokens=1000,
                output_tokens=500,
                cost=0.03,
                duration_ms=2000,
                schedule_id="test-schedule",
            )

        records = store.query()
        assert len(records) == 1
        r = records[0]
        assert r.agent == "test_agent"
        assert r.model == "openai:gpt-4o"
        assert r.input_tokens == 1000
        assert r.output_tokens == 500
        assert r.total_tokens == 1500
        assert r.cost == 0.03
        assert r.duration_ms == 2000
        assert r.schedule_id == "test-schedule"

    def test_schedule_id_none_for_non_scheduled_runs(self, tmp_path, monkeypatch):
        store = UsageStore(tmp_path / "usage.db")
        history_dir = tmp_path / "history"
        history_dir.mkdir()

        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: history_dir)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test")

        with patch("tsugite.usage.get_usage_store", return_value=store):
            from tsugite.agent_runner.history_integration import save_run_to_history

            agent_file = tmp_path / "test_agent.md"
            agent_file.write_text("---\nname: test_agent\nmodel: openai:gpt-4o\n---\nHello")

            save_run_to_history(
                agent_path=agent_file,
                agent_name="test_agent",
                prompt="test",
                result="result",
                model="openai:gpt-4o",
                token_count=100,
                cost=0.001,
            )

        records = store.query()
        assert len(records) == 1
        assert records[0].schedule_id is None


# ── Token split propagation ──


class TestTokenSplitPropagation:
    def test_agent_result_carries_token_split(self):
        from tsugite.core.agent import AgentResult

        result = AgentResult(
            output="test",
            token_usage=1500,
            cost=0.03,
            input_tokens=1000,
            output_tokens=500,
        )
        assert result.input_tokens == 1000
        assert result.output_tokens == 500

    def test_agent_execution_result_carries_token_split(self):
        from tsugite.agent_runner.models import AgentExecutionResult

        result = AgentExecutionResult(
            response="test",
            token_count=1500,
            input_tokens=1000,
            output_tokens=500,
            cost=0.03,
        )
        assert result.input_tokens == 1000
        assert result.output_tokens == 500

    def test_agent_execution_result_defaults_to_none(self):
        from tsugite.agent_runner.models import AgentExecutionResult

        result = AgentExecutionResult(response="test")
        assert result.input_tokens is None
        assert result.output_tokens is None


# ── Turn model enrichment ──


class TestTurnModelEnrichment:
    def test_turn_with_input_output_tokens(self, tmp_path, monkeypatch):
        from tsugite.history import SessionStorage, Turn

        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test")

        storage = SessionStorage.create(agent_name="test", model="openai:gpt-4o")
        storage.record_turn(
            messages=[{"role": "user", "content": "hello"}],
            final_answer="hi",
            tokens=1500,
            input_tokens=1000,
            output_tokens=500,
            cost=0.03,
        )

        loaded = SessionStorage.load(storage.session_path)
        records = loaded.load_records()
        turns = [r for r in records if isinstance(r, Turn)]
        assert len(turns) == 1
        assert turns[0].input_tokens == 1000
        assert turns[0].output_tokens == 500

    def test_backward_compat_without_token_split(self, tmp_path, monkeypatch):
        from tsugite.history import SessionStorage, Turn

        monkeypatch.setattr("tsugite.history.storage.get_history_dir", lambda: tmp_path)
        monkeypatch.setattr("tsugite.history.storage.get_machine_name", lambda: "test")

        storage = SessionStorage.create(agent_name="test", model="openai:gpt-4o")
        storage.record_turn(
            messages=[{"role": "user", "content": "hello"}],
            final_answer="hi",
            tokens=1500,
            cost=0.03,
        )

        loaded = SessionStorage.load(storage.session_path)
        records = loaded.load_records()
        turns = [r for r in records if isinstance(r, Turn)]
        assert len(turns) == 1
        assert turns[0].input_tokens is None
        assert turns[0].output_tokens is None
        assert turns[0].tokens == 1500


# ── RunResult enrichment ──


class TestRunResultEnrichment:
    @pytest.fixture(autouse=True)
    def _skip_if_no_cronsim(self):
        pytest.importorskip("cronsim")

    def test_run_result_has_tokens_and_cost(self):
        from tsugite.daemon.scheduler import RunResult

        result = RunResult(output="done", session_id="s1", tokens=1500, cost=0.03)
        assert result.tokens == 1500
        assert result.cost == 0.03

    def test_run_result_defaults_to_none(self):
        from tsugite.daemon.scheduler import RunResult

        result = RunResult(output="done")
        assert result.tokens is None
        assert result.cost is None


# ── UsageRecord model ──


class TestUsageRecordModel:
    def test_defaults(self):
        record = UsageRecord(timestamp="2026-03-15T10:00:00", agent="test", model="gpt-4o")
        assert record.input_tokens == 0
        assert record.output_tokens == 0
        assert record.total_tokens == 0
        assert record.cached_tokens == 0
        assert record.cost == 0.0
        assert record.session_id is None
        assert record.schedule_id is None
        assert record.duration_ms is None
        assert record.id is None

    def test_all_fields(self):
        record = UsageRecord(
            timestamp="2026-03-15T10:00:00",
            agent="chat",
            model="openai:gpt-4o",
            session_id="s1",
            schedule_id="morning",
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            cached_tokens=200,
            cost=0.03,
            duration_ms=2500,
            id=42,
        )
        assert record.agent == "chat"
        assert record.id == 42
