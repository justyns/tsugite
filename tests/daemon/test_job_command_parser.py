"""Tests for the /job slash command's AC + flag parsing helpers."""

from unittest.mock import MagicMock

import pytest
from tsugite_daemon.commands import _parse_acceptance_criteria, cmd_job


def test_empty_returns_empty_list():
    assert _parse_acceptance_criteria(None) == []
    assert _parse_acceptance_criteria("") == []
    assert _parse_acceptance_criteria([]) == []


def test_plain_pipe_separated_strings():
    out = _parse_acceptance_criteria("tests pass|PR open")
    assert out == ["tests pass", "PR open"]


def test_legacy_dict_entries_coerced_to_strings():
    """Old callers passing `{text, kind}` dicts get coerced to plain strings."""
    out = _parse_acceptance_criteria([{"text": "x", "kind": "ui"}, "y"])
    assert out == ["x", "y"]


def test_json_array_parsed():
    out = _parse_acceptance_criteria('["tests pass", "PR open"]')
    assert out == ["tests pass", "PR open"]


class _StubOrchestrator:
    def __init__(self):
        self.calls: list[dict] = []

    async def create_and_start_job(self, **kwargs):
        self.calls.append(kwargs)
        job = MagicMock(id="job-stub")
        started = MagicMock(id="session-stub")
        return job, started


class _StubAdapter:
    def __init__(self):
        self.session_store = MagicMock()
        primary = MagicMock(id="parent-1")
        self.session_store.find_default_session.return_value = primary
        self.session_store.get_session.return_value = primary
        self.agent_name = "default"


@pytest.fixture
def stub_orchestrator(monkeypatch):
    orch = _StubOrchestrator()
    monkeypatch.setattr("tsugite.tools.jobs._jobs_orchestrator", orch)
    return orch


@pytest.mark.asyncio
async def test_cmd_job_threads_max_attempts(stub_orchestrator):
    adapter = _StubAdapter()
    await cmd_job(
        adapter=adapter,
        user_id="user-1",
        prompt="do thing",
        max_attempts=5,
    )
    assert stub_orchestrator.calls[-1]["max_attempts"] == 5


@pytest.mark.asyncio
async def test_cmd_job_threads_notify_when(stub_orchestrator):
    adapter = _StubAdapter()
    await cmd_job(
        adapter=adapter,
        user_id="user-1",
        prompt="x",
        notify_when="stuck",
    )
    assert stub_orchestrator.calls[-1]["notify_when"] == "stuck"


@pytest.mark.asyncio
async def test_cmd_job_threads_executor(stub_orchestrator):
    adapter = _StubAdapter()
    await cmd_job(adapter=adapter, user_id="user-1", prompt="x", executor="cc")
    assert stub_orchestrator.calls[-1]["executor"] == "cc"


@pytest.mark.asyncio
async def test_cmd_job_default_executor_is_agent(stub_orchestrator):
    adapter = _StubAdapter()
    await cmd_job(adapter=adapter, user_id="user-1", prompt="x")
    assert stub_orchestrator.calls[-1]["executor"] == "agent"


@pytest.mark.asyncio
async def test_cmd_job_threads_effort(stub_orchestrator):
    adapter = _StubAdapter()
    await cmd_job(adapter=adapter, user_id="user-1", prompt="x", executor="cc", effort="max")
    assert stub_orchestrator.calls[-1]["effort"] == "max"
    await cmd_job(adapter=adapter, user_id="user-1", prompt="x")
    assert stub_orchestrator.calls[-1]["effort"] is None, "effort defaults to None (executor's own default)"


@pytest.mark.asyncio
async def test_cmd_job_default_notify_when_is_none(stub_orchestrator):
    """When no --notify-when is given, the orchestrator receives None and
    Job.__post_init__ normalises to 'never'."""
    adapter = _StubAdapter()
    await cmd_job(adapter=adapter, user_id="user-1", prompt="x")
    assert stub_orchestrator.calls[-1]["notify_when"] is None


# ── Job host-session anchoring (the "wrong session" fix) ────────────────────
#
# The Jobs-tab "new job" modal is never opened from inside a conversation, so it
# no longer guesses a parent chat to attach the Job to. When cmd_job gets no
# session_id it provisions a fresh, non-primary host session instead of silently
# landing the Job on whatever chat happened to be the user's primary.


def _real_adapter(tmp_path):
    from tsugite_daemon.session_store import SessionStore

    store = SessionStore(tmp_path / "session_store.json")
    adapter = MagicMock()
    adapter.session_store = store
    adapter.agent_name = "default"
    adapter.event_bus = None
    return adapter, store


@pytest.mark.asyncio
async def test_cmd_job_no_session_creates_fresh_host_session(stub_orchestrator, tmp_path):
    """No session_id (Jobs-tab modal): a new host session is created and the Job
    anchors to it - NOT to the user's primary chat."""
    adapter, store = _real_adapter(tmp_path)
    await cmd_job(adapter=adapter, user_id="user-1", prompt="build the thing\nmore detail")

    assert len(store._sessions) == 1, "a fresh host session should have been created"
    new_id = next(iter(store._sessions))
    host = store._sessions[new_id]
    assert stub_orchestrator.calls[-1]["parent_session_id"] == new_id
    assert host.is_primary is False, "host session must not steal the user's primary flag"
    assert host.user_id == "user-1"
    assert "build the thing" in (host.title or ""), "host session titled from the prompt"


@pytest.mark.asyncio
async def test_cmd_job_valid_session_anchors_there_no_extra_session(stub_orchestrator, tmp_path):
    """In-conversation path (composer /job): an existing session_id anchors the
    Job there and does NOT spawn an extra host session."""
    from tsugite_daemon.session_store import Session, SessionSource

    adapter, store = _real_adapter(tmp_path)
    store.create_session(
        Session(id="sess-existing", agent="default", source=SessionSource.INTERACTIVE.value, user_id="user-1")
    )
    await cmd_job(adapter=adapter, user_id="user-1", prompt="x", session_id="sess-existing")

    assert stub_orchestrator.calls[-1]["parent_session_id"] == "sess-existing"
    assert len(store._sessions) == 1, "no extra host session when anchoring to a real one"


@pytest.mark.asyncio
async def test_cmd_job_invalid_session_raises_friendly_command_error(stub_orchestrator, tmp_path):
    """A stale/invalid explicit session_id raises CommandError (HTTP maps it to a
    clean 400 with the message; Discord renders the text) - never an unhandled
    ValueError -> HTTP 500 - and spawns nothing."""
    from tsugite_daemon.commands import CommandError

    adapter, store = _real_adapter(tmp_path)
    with pytest.raises(CommandError, match="not found"):
        await cmd_job(adapter=adapter, user_id="user-1", prompt="x", session_id="does-not-exist")

    assert stub_orchestrator.calls == [], "no Job should be spawned for an invalid session"
    assert len(store._sessions) == 0, "no stray host session for an explicit-but-invalid id"
