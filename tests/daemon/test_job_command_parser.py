"""Tests for the /job slash command's AC + flag parsing helpers."""

from unittest.mock import MagicMock

import pytest

from tsugite.daemon.commands import _parse_acceptance_criteria, cmd_job


def test_empty_returns_empty_list():
    assert _parse_acceptance_criteria(None) == []
    assert _parse_acceptance_criteria("") == []
    assert _parse_acceptance_criteria([]) == []


def test_plain_pipe_separated_strings_default_to_llm_kind():
    out = _parse_acceptance_criteria("tests pass|PR open")
    assert out == [
        {"text": "tests pass", "kind": "llm"},
        {"text": "PR open", "kind": "llm"},
    ]


def test_ac_kind_inferred_from_double_colon_syntax():
    out = _parse_acceptance_criteria("tests pass::test|button renders::ui|curl returns 200::cmd")
    assert out == [
        {"text": "tests pass", "kind": "test"},
        {"text": "button renders", "kind": "ui"},
        {"text": "curl returns 200", "kind": "cmd"},
    ]


def test_unknown_kind_falls_back_to_llm():
    out = _parse_acceptance_criteria("does the right thing::garbage")
    assert out == [{"text": "does the right thing", "kind": "llm"}]


def test_dict_entries_pass_through():
    out = _parse_acceptance_criteria([{"text": "x", "kind": "ui"}])
    assert out == [{"text": "x", "kind": "ui"}]


def test_json_array_parsed_with_kind_suffix():
    out = _parse_acceptance_criteria('["tests pass::test", "PR open"]')
    assert out == [
        {"text": "tests pass", "kind": "test"},
        {"text": "PR open", "kind": "llm"},
    ]


class _StubOrchestrator:
    def __init__(self):
        self.calls: list[dict] = []

    def create_and_start_job(self, **kwargs):
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
async def test_cmd_job_legacy_notify_true_aliases_to_terminal(stub_orchestrator, caplog):
    adapter = _StubAdapter()
    import logging

    caplog.set_level(logging.WARNING)
    await cmd_job(
        adapter=adapter,
        user_id="user-1",
        prompt="x",
        notify=True,
    )
    assert stub_orchestrator.calls[-1]["notify_when"] == "terminal"
    # A deprecation warning should have been logged.
    assert any("notify" in rec.message and "deprecated" in rec.message.lower() for rec in caplog.records)


@pytest.mark.asyncio
async def test_cmd_job_explicit_notify_when_wins_over_legacy_notify(stub_orchestrator):
    adapter = _StubAdapter()
    await cmd_job(
        adapter=adapter,
        user_id="user-1",
        prompt="x",
        notify=True,
        notify_when="done",
    )
    # When both are supplied, notify_when takes precedence; no deprecation alias.
    assert stub_orchestrator.calls[-1]["notify_when"] == "done"
