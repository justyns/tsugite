"""The restart_daemon agent tool: preflight, then ask the user, then restart.

The @tool(require_daemon=True) decorator returns the bare function, so the tool is
called directly here; the module globals are wired through set_restart_controller()
with a real loop running in a background thread, the way the gateway wires them.
"""

import asyncio
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

    def preflight_restart(self):
        return list(self.problems)

    def request_restart(self):
        self.restarts += 1


@pytest.fixture(autouse=True)
def _clear_backend():
    set_interaction_backend(None)
    yield
    set_interaction_backend(None)


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


class TestApprovalGate:
    def test_approve_requests_the_restart(self, wire):
        controller = wire(_ControllerStub())
        backend = FakeBackend("Approve")
        set_interaction_backend(backend)

        result = daemon_control.restart_daemon(reason="picked up dashboard.py")

        assert len(backend.calls) == 1
        assert backend.calls[0][1] == "approval", "must route through request_approval, not a bare ask_user"
        assert "picked up dashboard.py" in backend.calls[0][0]
        assert controller.restarts == 1
        assert result == "Approved. The daemon restarts once this turn finishes."

    def test_deny_does_not_restart(self, wire):
        controller = wire(_ControllerStub())
        set_interaction_backend(FakeBackend("Deny"))

        result = daemon_control.restart_daemon(reason="new plugin")

        assert controller.restarts == 0
        assert result == "Restart not approved."

    def test_non_interactive_never_prompts_and_never_restarts(self, wire):
        controller = wire(_ControllerStub())
        spy = SpyNonInteractive()
        set_interaction_backend(spy)

        daemon_control.restart_daemon(reason="new plugin")

        assert spy.calls == []
        assert controller.restarts == 0

    def test_local_plugin_files_are_listed_in_the_prompt(self, wire, xdg_config_file):
        plugin_file = xdg_config_file.parent / "dashboard.py"
        plugin_file.write_text("MARKER = 'loaded'\n")
        xdg_config_file.write_text('{"plugins": {"dashboard": {"path": "dashboard.py"}}}')
        wire(_ControllerStub())
        backend = FakeBackend("Deny")
        set_interaction_backend(backend)

        daemon_control.restart_daemon(reason="new plugin")

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
