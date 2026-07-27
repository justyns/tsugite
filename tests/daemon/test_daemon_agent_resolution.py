"""A daemon adapter can be registered under a NAME that differs from its agent
file's config name (the registry key is the daemon-config agent name). When an
agent runs, the run context must expose that REGISTERED name so spawn/start
-session tools resolve to an agent that actually has a live adapter instead of
the agent-file config name (which has no adapter and fails with "No adapter for
agent ...").

The registered name rides a ContextVar (not a threading.local) so it reaches the
code-execution worker thread through the context asyncio.to_thread copies.
"""

import asyncio
import concurrent.futures
import contextvars
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tsugite.agent_runner.helpers import (
    get_current_daemon_agent,
    set_current_agent,
    set_current_daemon_agent,
)


class TestDaemonAgentContextVar:
    def test_default_is_none(self):
        assert get_current_daemon_agent() is None

    def test_set_and_get(self):
        set_current_daemon_agent("registered-name")
        assert get_current_daemon_agent() == "registered-name"

    def test_propagates_through_worker_thread_via_copied_context(self):
        """The value reaches a worker thread through copy_context().run(), exactly
        as asyncio.to_thread copies the run context into the code executor. A
        threading.local set on the async handler thread would NOT survive this."""
        set_current_daemon_agent("registered-name")
        ctx = contextvars.copy_context()

        def read_in_thread():
            return get_current_daemon_agent()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            result = ex.submit(ctx.run, read_in_thread).result()

        assert result == "registered-name"

    def test_lost_without_context_propagation(self):
        """Without ctx.run(), a fresh thread has its own empty context, so the
        value is invisible. This is why the daemon must set it INSIDE the run
        context that to_thread copies, not merely somewhere on the handler."""
        set_current_daemon_agent("registered-name")

        def read_in_thread():
            return get_current_daemon_agent()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            result = ex.submit(read_in_thread).result()

        assert result is None


def _mock_session_runner(monkeypatch):
    """Patch the sessions tool module so start/spawn capture the Session they
    would hand to the runner, without a real daemon."""
    from tsugite.tools import sessions as sessions_mod

    captured = {}

    def fake_call(fn, session, *a, **k):
        captured["session"] = session
        return session  # a Session dataclass; the tool does asdict(result)

    monkeypatch.setattr(sessions_mod, "_session_runner", SimpleNamespace(start_session=None))
    monkeypatch.setattr(sessions_mod, "_call", fake_call)
    return sessions_mod, captured


class TestStartSessionAgentResolution:
    def test_prefers_daemon_agent_over_config_name(self, monkeypatch):
        sessions_mod, captured = _mock_session_runner(monkeypatch)

        set_current_agent("config-name")
        set_current_daemon_agent("registered-name")

        sessions_mod.start_session(prompt="do work")

        assert captured["session"].agent == "registered-name"

    def test_falls_back_to_config_name(self, monkeypatch):
        sessions_mod, captured = _mock_session_runner(monkeypatch)

        set_current_agent("config-name")

        sessions_mod.start_session(prompt="do work")

        assert captured["session"].agent == "config-name"

    def test_defaults_when_nothing_set(self, monkeypatch):
        sessions_mod, captured = _mock_session_runner(monkeypatch)

        sessions_mod.start_session(prompt="do work")

        assert captured["session"].agent == "default"

    def test_explicit_agent_arg_wins(self, monkeypatch):
        sessions_mod, captured = _mock_session_runner(monkeypatch)

        set_current_daemon_agent("registered-name")

        sessions_mod.start_session(prompt="do work", agent="explicit")

        assert captured["session"].agent == "explicit"


@pytest.mark.asyncio
async def test_run_session_sets_daemon_agent_before_handle_message(tmp_path):
    """_run_session records session.agent (the resolved adapter key) as the daemon
    agent before handle_message, so a nested spawn inside the run inherits a name
    that still has a live adapter."""
    from tsugite_daemon.session_runner import SessionRunner
    from tsugite_daemon.session_store import Session, SessionSource, SessionStore

    captured = {}

    async def capture(**kwargs):
        captured["daemon_agent"] = get_current_daemon_agent()
        return "done"

    adapter = MagicMock()
    adapter.handle_message = AsyncMock(side_effect=capture)
    adapter.agent_config = MagicMock()
    adapter._resolve_agent_path = MagicMock(return_value=None)

    store = SessionStore(tmp_path / "session_store.json")
    runner = SessionRunner(store, {"registered-name": adapter})
    session = Session(
        id="s1",
        agent="registered-name",
        source=SessionSource.BACKGROUND.value,
        prompt="task",
    )
    runner.start_session(session)
    await asyncio.sleep(0.3)

    assert captured["daemon_agent"] == "registered-name"
