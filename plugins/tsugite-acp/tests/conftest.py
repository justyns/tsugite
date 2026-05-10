"""Shared fixtures for ACP plugin tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_conn():
    """A mock ClientSideConnection with AsyncMock methods.

    Tests can pre-set return values on conn.initialize / new_session / etc.
    """
    from acp.schema import (
        AgentCapabilities,
        InitializeResponse,
        NewSessionResponse,
        PromptCapabilities,
        SessionCapabilities,
        SessionCloseCapabilities,
    )

    conn = MagicMock()
    conn.initialize = AsyncMock(
        return_value=InitializeResponse(
            protocol_version=1,
            agent_capabilities=AgentCapabilities(
                load_session=True,
                prompt_capabilities=PromptCapabilities(image=True, audio=False, embedded_context=True),
                session_capabilities=SessionCapabilities(close=SessionCloseCapabilities()),
            ),
            auth_methods=[],
        )
    )
    conn.new_session = AsyncMock(return_value=NewSessionResponse(session_id="sess-test-1"))
    conn.load_session = AsyncMock(return_value=None)
    conn.prompt = AsyncMock()
    conn.cancel = AsyncMock()
    conn.close_session = AsyncMock()
    conn.close = AsyncMock()
    return conn
