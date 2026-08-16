"""Test doubles for the InteractionBackend protocol.

Shared so the ask_user signature is written down once: a change to the protocol
in tsugite/interaction.py otherwise surfaces as a wrong-arity TypeError in
whichever suite was not updated.
"""

from tsugite.interaction import NonInteractiveBackend


class FakeBackend:
    """Records every ask_user call and returns a canned answer."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def ask_user(self, question, question_type="text", options=None):
        self.calls.append((question, question_type, options))
        return self.answer


class SpyNonInteractive(NonInteractiveBackend):
    """A NonInteractiveBackend that records whether it was prompted."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def ask_user(self, question, question_type="text", options=None):
        self.calls.append((question, question_type, options))
        return super().ask_user(question, question_type, options)
