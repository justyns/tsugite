"""The restart_daemon agent tool: preflight, then ask the user, then restart.

The @tool(require_daemon=True) decorator returns the bare function, so the tool is
called directly here; the module globals are wired through set_restart_controller()
with a real loop running in a background thread, the way the gateway wires them.
"""

import asyncio
import threading
from threading import Thread

import pytest

import tsugite.tools.daemon_control as daemon_control
from tests.interaction_doubles import FakeBackend, SpyNonInteractive
from tsugite.interaction import set_interaction_backend
from tsugite.tools import list_tools, set_daemon_mode


class _ControllerStub:
    """Stands in for the Gateway: reports preflight problems, counts restarts."""

    def __init__(self, problems=None):
        self.problems = problems or []
        self.restarts = 0
        self.restarted = threading.Event()

    def preflight_restart(self):
        return list(self.problems)

    def request_restart(self):
        self.restarts += 1
        self.restarted.set()


@pytest.fixture(autouse=True)
def _clear_backend(monkeypatch):
    monkeypatch.setattr(daemon_control, "_approval_pending", threading.Lock())
    set_interaction_backend(None)
    yield
    set_interaction_backend(None)


@pytest.fixture
def approval_done(monkeypatch):
    """Set once the background approval flow has finished, whatever it decided."""
    done = threading.Event()
    flow = daemon_control._approve_then_restart

    def _flow(*args):
        try:
            flow(*args)
        finally:
            done.set()

    monkeypatch.setattr(daemon_control, "_approve_then_restart", _flow)
    return done


@pytest.fixture
def notified(monkeypatch):
    """Messages the flow sends back to the originating session."""
    sent = []
    monkeypatch.setattr(daemon_control, "get_current_session_id", lambda: "origin-1")
    monkeypatch.setattr(daemon_control, "session_reply", lambda msg, session_id: sent.append((session_id, msg)))
    return sent


@pytest.fixture
def wire():
    """Wire a controller stub onto a real background loop, as the gateway does."""
    loop = asyncio.new_event_loop()
    thread = Thread(target=loop.run_forever, daemon=True)
    thread.start()

    def _wire(controller):
        daemon_control.set_restart_controller(controller, loop)
        return controller

    yield _wire

    daemon_control.set_restart_controller(None, None)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)
    loop.close()


@pytest.fixture
def blocked_backend():
    """An approving backend that holds its prompt open until `release` is set."""
    release = threading.Event()

    class _BlockedBackend(FakeBackend):
        def ask_user(self, question, question_type="text", options=None):
            release.wait(5)
            return super().ask_user(question, question_type, options)

    backend = _BlockedBackend("Approve")
    set_interaction_backend(backend)
    return backend, release


class TestApprovalGate:
    def test_approve_requests_the_restart(self, wire):
        controller = wire(_ControllerStub())
        backend = FakeBackend("Approve")
        set_interaction_backend(backend)

        result = daemon_control.restart_daemon(reason="picked up dashboard.py")
        assert controller.restarted.wait(2)

        assert len(backend.calls) == 1
        assert backend.calls[0][1] == "approval", "must route through request_approval, not a bare ask_user"
        assert "picked up dashboard.py" in backend.calls[0][0]
        assert result == "Asked the user to approve a restart."
        assert controller.restarts == 1

    def test_returns_while_the_approval_is_pending(self, wire, blocked_backend):
        controller = wire(_ControllerStub())
        _, release = blocked_backend

        result = daemon_control.restart_daemon(reason="new plugin")

        assert result == "Asked the user to approve a restart."
        assert controller.restarts == 0
        release.set()
        assert controller.restarted.wait(2)

    def test_second_call_while_pending_does_not_prompt_again(self, wire, blocked_backend):
        controller = wire(_ControllerStub())
        backend, release = blocked_backend

        daemon_control.restart_daemon(reason="new plugin")
        second = daemon_control.restart_daemon(reason="again")

        assert second == "A restart approval is already pending."
        release.set()
        assert controller.restarted.wait(2)
        assert len(backend.calls) == 1
        assert controller.restarts == 1

    def test_deny_does_not_restart_and_tells_the_session(self, wire, approval_done, notified):
        controller = wire(_ControllerStub())
        backend = FakeBackend("Deny")
        set_interaction_backend(backend)

        result = daemon_control.restart_daemon(reason="new plugin")
        assert approval_done.wait(2)

        assert result == "Asked the user to approve a restart."
        assert len(backend.calls) == 1
        assert controller.restarts == 0
        assert notified == [("origin-1", "The user denied the daemon restart.")]

    def test_approval_failure_tells_the_session(self, wire, approval_done, notified):
        wire(_ControllerStub())

        class _BrokenBackend(FakeBackend):
            def ask_user(self, question, question_type="text", options=None):
                raise RuntimeError("prompt exploded")

        set_interaction_backend(_BrokenBackend("Approve"))

        daemon_control.restart_daemon(reason="new plugin")
        assert approval_done.wait(2)

        assert notified == [("origin-1", "The daemon restart failed: prompt exploded")]

    def test_deny_releases_the_lock_even_if_notifying_the_session_fails(self, wire, approval_done, monkeypatch):
        wire(_ControllerStub())
        monkeypatch.setattr(daemon_control, "get_current_session_id", lambda: "origin-1")

        def _broken_reply(msg, session_id):
            raise RuntimeError("session gone")

        monkeypatch.setattr(daemon_control, "session_reply", _broken_reply)
        set_interaction_backend(FakeBackend("Deny"))

        daemon_control.restart_daemon(reason="new plugin")
        assert approval_done.wait(2)

        assert daemon_control._approval_pending.acquire(blocking=False)

    def test_approval_keeps_the_lock_so_a_second_call_does_not_prompt(self, wire, approval_done):
        controller = wire(_ControllerStub())
        backend = FakeBackend("Approve")
        set_interaction_backend(backend)

        daemon_control.restart_daemon(reason="new plugin")
        assert approval_done.wait(2)
        second = daemon_control.restart_daemon(reason="again")

        assert second == "A restart approval is already pending."
        assert len(backend.calls) == 1
        assert controller.restarts == 1

    def test_non_interactive_never_prompts_and_never_restarts(self, wire):
        controller = wire(_ControllerStub())
        spy = SpyNonInteractive()
        set_interaction_backend(spy)

        result = daemon_control.restart_daemon(reason="new plugin")

        assert result == "Restart not approved."
        assert spy.calls == []
        assert controller.restarts == 0

    def test_local_plugin_files_are_listed_in_the_prompt(self, wire, approval_done, xdg_config_file):
        plugin_file = xdg_config_file.parent / "dashboard.py"
        plugin_file.write_text("MARKER = 'loaded'\n")
        xdg_config_file.write_text('{"plugins": {"dashboard": {"path": "dashboard.py"}}}')
        wire(_ControllerStub())
        backend = FakeBackend("Deny")
        set_interaction_backend(backend)

        daemon_control.restart_daemon(reason="new plugin")
        assert approval_done.wait(2)

        assert len(backend.calls) == 1
        assert str(plugin_file) in backend.calls[0][0]

    def test_sandboxed_never_reaches_the_prompt(self, wire):
        from tsugite.agent_runner.helpers import (
            SandboxContext,
            SandboxToolDeniedError,
            clear_sandbox_context,
            set_sandbox_context,
        )

        controller = wire(_ControllerStub())
        set_interaction_backend(FakeBackend("Approve"))
        set_sandbox_context(SandboxContext())
        try:
            with pytest.raises(SandboxToolDeniedError, match="restart_daemon"):
                daemon_control.restart_daemon(reason="new plugin")
        finally:
            clear_sandbox_context()
        assert controller.restarts == 0


class TestPreflight:
    def test_problems_block_the_prompt(self, wire):
        controller = wire(_ControllerStub(problems=["dashboard.py line 3: invalid syntax"]))
        backend = FakeBackend("Approve")
        set_interaction_backend(backend)

        result = daemon_control.restart_daemon(reason="new plugin")

        assert backend.calls == [], "the user must not be asked to approve a restart the preflight already rejected"
        assert controller.restarts == 0
        assert "dashboard.py line 3: invalid syntax" in result

    def test_unwired_controller_reports_instead_of_prompting(self):
        daemon_control.set_restart_controller(None, None)
        backend = FakeBackend("Approve")
        set_interaction_backend(backend)

        result = daemon_control.restart_daemon(reason="new plugin")

        assert backend.calls == []
        assert "not available" in result


@pytest.fixture
def daemon_mode():
    set_daemon_mode(True)
    yield
    set_daemon_mode(False)


class TestRegistration:
    def test_registered_in_daemon_mode(self, daemon_mode):
        assert "restart_daemon" in list_tools()

    def test_absent_outside_daemon_mode(self):
        assert "restart_daemon" not in list_tools()

    def test_routes_to_the_parent_process(self, daemon_mode):
        """require_daemon keeps the re-exec in the parent, out of a sandboxed child."""
        from tsugite.core.subprocess_executor import SubprocessExecutor
        from tsugite.core.tools import create_tool_from_tsugite

        ex = SubprocessExecutor()
        ex.set_tools([create_tool_from_tsugite("restart_daemon")])

        assert "restart_daemon" in ex._parent_only_tools
        assert "restart_daemon" not in ex._local_tools
