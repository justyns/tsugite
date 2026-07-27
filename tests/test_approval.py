"""Tests for the backend-driven approval primitive."""

import pytest

from tsugite.approval import request_approval
from tsugite.interaction import NonInteractiveBackend, set_interaction_backend


@pytest.fixture(autouse=True)
def _clear_backend():
    """Reset the interaction backend before/after each test."""
    set_interaction_backend(None)
    yield
    set_interaction_backend(None)


class _FakeBackend:
    """Records every ask_user call and returns a canned answer."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def ask_user(self, question, question_type="text", options=None):
        self.calls.append((question, question_type, options))
        return self.answer


class _SpyNonInteractive(NonInteractiveBackend):
    """A NonInteractiveBackend that records whether it was prompted."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def ask_user(self, question, question_type="text", options=None):
        self.calls.append((question, question_type, options))
        return super().ask_user(question, question_type, options)


class TestDecisionMapping:
    def test_approve(self):
        set_interaction_backend(_FakeBackend("Approve"))
        assert request_approval("Fetch x?") == "approve"

    def test_deny(self):
        set_interaction_backend(_FakeBackend("Deny"))
        assert request_approval("Fetch x?") == "deny"

    def test_always(self):
        set_interaction_backend(_FakeBackend("Always allow"))
        assert request_approval("Fetch x?", allow_always=True) == "always"

    def test_options_built_without_always(self):
        backend = _FakeBackend("Approve")
        set_interaction_backend(backend)
        request_approval("Fetch x?")
        question, qtype, options = backend.calls[0]
        assert qtype == "approval"
        assert options == ["Approve", "Deny"]

    def test_options_include_always_when_allowed(self):
        backend = _FakeBackend("Approve")
        set_interaction_backend(backend)
        request_approval("Fetch x?", allow_always=True)
        assert backend.calls[0][2] == ["Approve", "Deny", "Always allow"]

    def test_detail_folded_into_question(self):
        backend = _FakeBackend("Approve")
        set_interaction_backend(backend)
        request_approval("Fetch content?", detail="Domain: evil.test")
        question = backend.calls[0][0]
        assert "Fetch content?" in question
        assert "Domain: evil.test" in question

    def test_unrecognized_answer_denies(self):
        set_interaction_backend(_FakeBackend("something unexpected"))
        assert request_approval("Fetch x?") == "deny"


class TestFailClosed:
    def test_no_backend_denies_without_prompting(self):
        # Fixture cleared the backend: no interactive surface is available.
        assert request_approval("Fetch x?") == "deny"

    def test_non_interactive_denies_without_prompting(self):
        spy = _SpyNonInteractive()
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
