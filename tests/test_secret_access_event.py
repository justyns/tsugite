"""Tests for SecretAccessEvent notification."""

import json
from unittest.mock import MagicMock, patch

import pytest

from tsugite.events import SecretAccessEvent
from tsugite.events.base import EventType
from tsugite.events.helpers import emit_secret_access_event
from tsugite.ui.jsonl import JSONLUIHandler


class TestSecretAccessEvent:
    def test_event_creation(self):
        event = SecretAccessEvent(name="forgejo-token")
        assert event.event_type == EventType.SECRET_ACCESS
        assert event.name == "forgejo-token"

    def test_event_is_frozen(self):
        event = SecretAccessEvent(name="token")
        with pytest.raises(Exception):
            event.name = "other"

    def test_no_value_field(self):
        assert "value" not in SecretAccessEvent.model_fields


class TestEmitSecretAccessEvent:
    def test_emits_to_event_bus(self):
        mock_bus = MagicMock()
        with patch("tsugite.ui_context.get_event_bus", return_value=mock_bus):
            emit_secret_access_event("my-secret")
        mock_bus.emit.assert_called_once()
        event = mock_bus.emit.call_args[0][0]
        assert isinstance(event, SecretAccessEvent)
        assert event.name == "my-secret"

    def test_no_op_without_event_bus(self):
        with patch("tsugite.ui_context.get_event_bus", return_value=None):
            emit_secret_access_event("my-secret")


class TestJsonlHandlerSecretAccess:
    def test_handler_outputs_secret_access(self, capsys):
        handler = JSONLUIHandler()
        event = SecretAccessEvent(name="forgejo-token")
        handler.handle_event(event)
        output = capsys.readouterr().out.strip()
        data = json.loads(output)
        assert data["type"] == "secret_access"
        assert data["name"] == "forgejo-token"


class TestGetSecretEmitsEvent:
    @patch("tsugite.events.helpers.emit_secret_access_event")
    @patch("tsugite.secrets.registry.get_registry")
    @patch("tsugite.secrets.get_backend")
    @patch("tsugite.agent_runner.helpers.get_allowed_secrets", return_value=[])
    @patch("tsugite.agent_runner.helpers.resolve_current_agent", return_value="test-agent")
    def test_get_secret_emits_event(self, mock_agent, mock_allowed, mock_backend, mock_registry, mock_emit):
        mock_backend.return_value.get.return_value = "secret-value"
        mock_registry.return_value.register.return_value = "secret-value"

        from tsugite.tools.secrets import get_secret

        result = get_secret("my-token")
        assert result == "secret-value"
        mock_emit.assert_called_once_with("my-token")
