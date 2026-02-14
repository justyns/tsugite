"""Tests for per-adapter interactive tool backends."""

import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

from tsugite.interaction import (
    InteractionBackend,
    NonInteractiveBackend,
    TerminalInteractionBackend,
    get_interaction_backend,
    set_interaction_backend,
)


@pytest.fixture(autouse=True)
def _clear_backend():
    """Reset backend before/after each test."""
    set_interaction_backend(None)
    yield
    set_interaction_backend(None)


class TestContextVar:
    def test_default_is_none(self):
        assert get_interaction_backend() is None

    def test_set_and_get(self):
        backend = NonInteractiveBackend()
        set_interaction_backend(backend)
        assert get_interaction_backend() is backend

    def test_clear(self):
        set_interaction_backend(NonInteractiveBackend())
        set_interaction_backend(None)
        assert get_interaction_backend() is None


class TestNonInteractiveBackend:
    def test_yes_no_defaults_yes(self):
        backend = NonInteractiveBackend()
        assert backend.ask_user("Continue?", "yes_no") == "yes"

    def test_yes_no_custom_default(self):
        backend = NonInteractiveBackend(default_yes_no="no")
        assert backend.ask_user("Continue?", "yes_no") == "no"

    def test_choice_returns_first(self):
        backend = NonInteractiveBackend()
        assert backend.ask_user("Pick:", "choice", ["a", "b", "c"]) == "a"

    def test_choice_custom_index(self):
        backend = NonInteractiveBackend(default_choice_index=2)
        assert backend.ask_user("Pick:", "choice", ["a", "b", "c"]) == "c"

    def test_choice_index_clamped(self):
        backend = NonInteractiveBackend(default_choice_index=99)
        assert backend.ask_user("Pick:", "choice", ["a", "b"]) == "b"

    def test_text_raises(self):
        backend = NonInteractiveBackend()
        with pytest.raises(RuntimeError, match="non-interactive"):
            backend.ask_user("Name?", "text")


class TestTerminalInteractionBackend:
    def test_delegates_to_handle_question(self, monkeypatch):
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

        with patch("tsugite.tools.interactive.Confirm.ask", return_value=True):
            backend = TerminalInteractionBackend()
            result = backend.ask_user("Ok?", "yes_no")
            assert result == "yes"


class TestProtocol:
    def test_non_interactive_satisfies_protocol(self):
        assert isinstance(NonInteractiveBackend(), InteractionBackend)

    def test_terminal_satisfies_protocol(self):
        assert isinstance(TerminalInteractionBackend(), InteractionBackend)


class TestAskUserDispatch:
    """Test that ask_user tool dispatches through backend when set."""

    @pytest.fixture
    def interactive_tool(self, reset_tool_registry):
        from tsugite.tools import tool
        from tsugite.tools.interactive import ask_user

        tool(ask_user)

    def test_dispatches_to_backend(self, interactive_tool):
        mock_backend = MagicMock(spec=InteractionBackend)
        mock_backend.ask_user.return_value = "backend answer"
        set_interaction_backend(mock_backend)

        from tsugite.tools import call_tool

        result = call_tool("ask_user", question="test?", question_type="text")
        assert result == "backend answer"
        mock_backend.ask_user.assert_called_once_with("test?", "text", None)

    def test_falls_back_to_tty(self, interactive_tool, monkeypatch):
        """No backend set + interactive TTY = original behavior."""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

        with patch("tsugite.tools.interactive.Confirm.ask", return_value=False):
            from tsugite.tools import call_tool

            result = call_tool("ask_user", question="Sure?", question_type="yes_no")
            assert result == "no"

    def test_raises_non_interactive_no_backend(self, interactive_tool, monkeypatch):
        """No backend + not interactive = error."""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

        from tsugite.tools import call_tool

        with pytest.raises(RuntimeError, match="non-interactive"):
            call_tool("ask_user", question="test?", question_type="text")

    def test_validation_before_dispatch(self, interactive_tool):
        """Validation happens before backend dispatch."""
        mock_backend = MagicMock(spec=InteractionBackend)
        set_interaction_backend(mock_backend)

        from tsugite.tools.interactive import ask_user

        with pytest.raises(ValueError, match="Invalid question_type"):
            ask_user(question="test?", question_type="invalid")
        mock_backend.ask_user.assert_not_called()


class TestAskUserBatchDispatch:
    """Test that ask_user_batch dispatches through backend when set."""

    @pytest.fixture
    def batch_tool(self, reset_tool_registry):
        from tsugite.tools import tool
        from tsugite.tools.interactive import ask_user_batch

        tool(ask_user_batch)

    def test_batch_dispatches_to_backend(self, batch_tool):
        mock_backend = MagicMock(spec=InteractionBackend)
        mock_backend.ask_user.side_effect = ["Alice", "yes"]
        set_interaction_backend(mock_backend)

        from tsugite.tools import call_tool

        result = call_tool(
            "ask_user_batch",
            questions=[
                {"id": "name", "question": "Name?", "type": "text"},
                {"id": "confirm", "question": "Ok?", "type": "yes_no"},
            ],
        )
        assert result == {"name": "Alice", "confirm": "yes"}
        assert mock_backend.ask_user.call_count == 2


class TestHTTPInteractionBackend:
    def test_emits_event_and_blocks(self):
        from tsugite.daemon.adapters.http import HTTPInteractionBackend, SSEProgressHandler

        progress = MagicMock(spec=SSEProgressHandler)
        backend = HTTPInteractionBackend(progress)

        def respond_later():
            import time

            time.sleep(0.1)
            backend.submit_response("user said hello")

        t = threading.Thread(target=respond_later)
        t.start()

        result = backend.ask_user("What?", "text")
        t.join()

        assert result == "user said hello"
        progress._emit.assert_called_once_with(
            "ask_user", {"question": "What?", "question_type": "text"}
        )

    def test_timeout_raises(self):
        from tsugite.daemon.adapters.http import HTTPInteractionBackend, SSEProgressHandler

        progress = MagicMock(spec=SSEProgressHandler)
        backend = HTTPInteractionBackend(progress)
        backend.TIMEOUT = 0.1  # very short for test

        with pytest.raises(RuntimeError, match="Timed out"):
            backend.ask_user("Quick?", "text")


class TestContextPropagationThroughThreads:
    """Test that interaction backend propagates through _run_async_in_sync_context."""

    def test_backend_propagates_through_tool_execution_thread(self):
        """Verify contextvar survives the ThreadPoolExecutor in _run_async_in_sync_context."""
        import asyncio
        import concurrent.futures
        import contextvars

        mock_backend = MagicMock(spec=InteractionBackend)
        mock_backend.ask_user.return_value = "thread answer"
        set_interaction_backend(mock_backend)

        # Simulate what _run_async_in_sync_context does: copy context, run in new thread
        ctx = contextvars.copy_context()

        def run_in_thread():
            # This is what ask_user does inside the tool
            from tsugite.interaction import get_interaction_backend
            backend = get_interaction_backend()
            if backend is not None:
                return backend.ask_user("test?", "text")
            return "NO BACKEND"

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(ctx.run, run_in_thread)
            result = future.result()

        assert result == "thread answer"

    def test_backend_lost_without_context_propagation(self):
        """Without ctx.run(), the backend is NOT available in the new thread."""
        import concurrent.futures

        mock_backend = MagicMock(spec=InteractionBackend)
        set_interaction_backend(mock_backend)

        def run_in_thread():
            from tsugite.interaction import get_interaction_backend
            return get_interaction_backend()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_in_thread)
            result = future.result()

        # Without context propagation, backend is None in new thread
        assert result is None
