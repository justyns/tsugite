"""Tests for the backend-driven approval primitive."""

import pytest

from tests.interaction_doubles import FakeBackend, SpyNonInteractive
from tsugite.approval import request_approval
from tsugite.interaction import NonInteractiveBackend, set_interaction_backend


@pytest.fixture(autouse=True)
def _clear_backend():
    """Reset the interaction backend before/after each test."""
    set_interaction_backend(None)
    yield
    set_interaction_backend(None)


class TestDecisionMapping:
    def test_approve(self):
        set_interaction_backend(FakeBackend("Approve"))
        assert request_approval("Fetch x?") == "approve"

    def test_deny(self):
        set_interaction_backend(FakeBackend("Deny"))
        assert request_approval("Fetch x?") == "deny"

    def test_always(self):
        set_interaction_backend(FakeBackend("Always allow"))
        assert request_approval("Fetch x?", allow_always=True) == "always"

    def test_options_built_without_always(self):
        backend = FakeBackend("Approve")
        set_interaction_backend(backend)
        request_approval("Fetch x?")
        question, qtype, options = backend.calls[0]
        assert qtype == "approval"
        assert options == ["Approve", "Deny"]

    def test_options_include_always_when_allowed(self):
        backend = FakeBackend("Approve")
        set_interaction_backend(backend)
        request_approval("Fetch x?", allow_always=True)
        assert backend.calls[0][2] == ["Approve", "Deny", "Always allow"]

    def test_detail_folded_into_question(self):
        backend = FakeBackend("Approve")
        set_interaction_backend(backend)
        request_approval("Fetch content?", detail="Domain: evil.test")
        question = backend.calls[0][0]
        assert "Fetch content?" in question
        assert "Domain: evil.test" in question

    def test_unrecognized_answer_denies(self):
        set_interaction_backend(FakeBackend("something unexpected"))
        assert request_approval("Fetch x?") == "deny"


class TestFailClosed:
    def test_no_backend_denies_without_prompting(self):
        # Fixture cleared the backend: no interactive surface is available.
        assert request_approval("Fetch x?") == "deny"

    def test_non_interactive_denies_without_prompting(self):
        spy = SpyNonInteractive()
        set_interaction_backend(spy)
        assert request_approval("Fetch x?", allow_always=True) == "deny"
        # Load-bearing guard: fail-closed must short-circuit BEFORE any prompt.
        assert spy.calls == []


class TestNonInteractiveApprovalBackend:
    """The scheduler/headless backend must never auto-approve."""

    def test_returns_deny_option(self):
        backend = NonInteractiveBackend()
        result = backend.ask_user("Fetch x?", "approval", ["Approve", "Deny", "Always allow"])
        assert result == "Deny"

    def test_never_returns_approve(self):
        backend = NonInteractiveBackend()
        result = backend.ask_user("Fetch x?", "approval", ["Approve", "Deny"])
        assert result != "Approve"

    def test_no_options_returns_deny_literal(self):
        backend = NonInteractiveBackend()
        assert backend.ask_user("Fetch x?", "approval") == "deny"


class TestTerminalApprovalMapping:
    """handle_question_by_type routes 'approval' through the choice renderer."""

    def test_approval_maps_to_choice(self, monkeypatch):
        import tsugite.tools.interactive as interactive

        captured = {}

        def fake_choice(question, options, console, flush_fn):
            captured["question"] = question
            captured["options"] = options
            return options[0]

        monkeypatch.setattr(interactive, "ask_choice_question", fake_choice)
        result = interactive.handle_question_by_type(
            "approval", "Fetch x?", ["Approve", "Deny"], console=None, flush_fn=lambda: None
        )
        assert captured["options"] == ["Approve", "Deny"]
        assert result == "Approve"

    def test_approval_without_options_raises(self):
        import tsugite.tools.interactive as interactive

        with pytest.raises(ValueError):
            interactive.handle_question_by_type("approval", "Fetch x?", None, console=None, flush_fn=lambda: None)
