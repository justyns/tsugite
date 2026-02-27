"""Tests for structured output validation (Feature #5)."""

from dataclasses import asdict
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tsugite.daemon.adapters.scheduler_adapter import SchedulerAdapter, _extract_json, _validate_output
from tsugite.daemon.scheduler import ScheduleEntry


class TestExtractJson:
    def test_direct_json(self):
        assert _extract_json('{"name": "test", "count": 5}') == {"name": "test", "count": 5}

    def test_code_block_json(self):
        text = 'Here is the result:\n```json\n{"name": "test"}\n```\nDone.'
        assert _extract_json(text) == {"name": "test"}

    def test_mixed_text_with_braces(self):
        text = 'The output is {"status": "ok", "value": 42} as expected.'
        assert _extract_json(text) == {"status": "ok", "value": 42}

    def test_no_json_returns_none(self):
        assert _extract_json("no json here") is None

    def test_empty_string(self):
        assert _extract_json("") is None

    def test_invalid_json(self):
        assert _extract_json("{not valid json}") is None

    def test_nested_json(self):
        text = '{"outer": {"inner": true}, "list": [1, 2, 3]}'
        result = _extract_json(text)
        assert result == {"outer": {"inner": True}, "list": [1, 2, 3]}


class TestValidateOutput:
    def test_all_fields_match(self):
        result = '{"name": "test", "count": 5, "active": true}'
        schema = {"name": "str", "count": "int", "active": "bool"}
        assert _validate_output(result, schema) is True

    def test_missing_field(self):
        result = '{"name": "test"}'
        schema = {"name": "str", "count": "int"}
        assert _validate_output(result, schema) is False

    def test_wrong_type(self):
        result = '{"name": 123}'
        schema = {"name": "str"}
        assert _validate_output(result, schema) is False

    def test_any_type_accepts_anything(self):
        result = '{"data": [1, 2, 3]}'
        schema = {"data": "any"}
        assert _validate_output(result, schema) is True

    def test_bool_vs_int(self):
        """Bool should not pass int validation (bool subclasses int in Python)."""
        result = '{"count": true}'
        schema = {"count": "int"}
        assert _validate_output(result, schema) is False

    def test_int_does_not_pass_bool(self):
        result = '{"active": 1}'
        schema = {"active": "bool"}
        assert _validate_output(result, schema) is False

    def test_no_json_returns_false(self):
        assert _validate_output("no json", {"field": "str"}) is False

    def test_float_type(self):
        result = '{"score": 3.14}'
        schema = {"score": "float"}
        assert _validate_output(result, schema) is True

    def test_list_type(self):
        result = '{"items": [1, 2, 3]}'
        schema = {"items": "list"}
        assert _validate_output(result, schema) is True

    def test_dict_type(self):
        result = '{"config": {"key": "value"}}'
        schema = {"config": "dict"}
        assert _validate_output(result, schema) is True

    def test_empty_schema_always_passes(self):
        result = '{"anything": "goes"}'
        assert _validate_output(result, {}) is True


class TestOutputSchemaField:
    def test_defaults_to_none(self):
        entry = ScheduleEntry(id="t", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *")
        assert entry.output_schema is None
        assert entry.last_output_valid is None

    def test_serialization_roundtrip(self):
        schema = {"name": "str", "count": "int"}
        entry = ScheduleEntry(
            id="t", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *",
            output_schema=schema,
        )
        data = asdict(entry)
        restored = ScheduleEntry(**data)
        assert restored.output_schema == schema


class TestRunAgentValidation:
    @pytest.fixture
    def adapter(self):
        adapter = MagicMock()
        adapter.agent_config = MagicMock()
        adapter.agent_config.workspace_dir = Path("/workspace")
        adapter.handle_message = AsyncMock(return_value='{"name": "test", "count": 5}')
        return adapter

    @pytest.fixture
    def scheduler_adapter(self, tmp_path, adapter):
        return SchedulerAdapter(
            adapters={"bot": adapter},
            schedules_path=tmp_path / "schedules.json",
        )

    @pytest.mark.asyncio
    async def test_sets_valid_true(self, adapter, scheduler_adapter):
        entry = ScheduleEntry(
            id="t", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *",
            output_schema={"name": "str", "count": "int"},
        )

        with patch("tsugite.daemon.adapters.scheduler_adapter.send_notification"):
            await scheduler_adapter._run_agent(entry)

        assert entry.last_output_valid is True

    @pytest.mark.asyncio
    async def test_sets_valid_false(self, adapter, scheduler_adapter):
        adapter.handle_message = AsyncMock(return_value="not json at all")
        entry = ScheduleEntry(
            id="t", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *",
            output_schema={"name": "str"},
        )

        with patch("tsugite.daemon.adapters.scheduler_adapter.send_notification"):
            await scheduler_adapter._run_agent(entry)

        assert entry.last_output_valid is False

    @pytest.mark.asyncio
    async def test_none_when_no_schema(self, adapter, scheduler_adapter):
        entry = ScheduleEntry(
            id="t", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *",
        )

        with patch("tsugite.daemon.adapters.scheduler_adapter.send_notification"):
            await scheduler_adapter._run_agent(entry)

        assert entry.last_output_valid is None

    @pytest.mark.asyncio
    async def test_notification_includes_validation_status(self, adapter, scheduler_adapter):
        adapter.handle_message = AsyncMock(return_value='{"name": "test"}')
        entry = ScheduleEntry(
            id="t", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *",
            output_schema={"name": "str"},
            notify=["test-channel"],
        )
        scheduler_adapter._notification_channels = {
            "test-channel": MagicMock(type="discord", user_id="u1")
        }

        with patch("tsugite.daemon.adapters.scheduler_adapter.send_notification") as mock_notify:
            with patch.object(scheduler_adapter, "_inject_into_user_sessions", new_callable=AsyncMock):
                await scheduler_adapter._run_agent(entry)

            notification_text = mock_notify.call_args[0][0]
            assert "Output validation: passed" in notification_text

    @pytest.mark.asyncio
    async def test_notification_includes_failed_validation(self, adapter, scheduler_adapter):
        adapter.handle_message = AsyncMock(return_value="no json")
        entry = ScheduleEntry(
            id="t", agent="bot", prompt="hi", schedule_type="cron", cron_expr="0 9 * * *",
            output_schema={"name": "str"},
            notify=["test-channel"],
        )
        scheduler_adapter._notification_channels = {
            "test-channel": MagicMock(type="discord", user_id="u1")
        }

        with patch("tsugite.daemon.adapters.scheduler_adapter.send_notification") as mock_notify:
            with patch.object(scheduler_adapter, "_inject_into_user_sessions", new_callable=AsyncMock):
                await scheduler_adapter._run_agent(entry)

            notification_text = mock_notify.call_args[0][0]
            assert "Output validation: FAILED" in notification_text
