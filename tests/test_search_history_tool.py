import pytest

from tsugite.agent_runner import reset_current_agent, set_current_agent
from tsugite.secrets.registry import get_registry
from tsugite.tools.history import search_history

from .history_helpers import seed_history_session


@pytest.fixture
def as_alpha(history_dir):
    token = set_current_agent("alpha")
    yield
    reset_current_agent(token)
    get_registry().clear()


def _seed(sid, agent, text):
    return seed_history_session(sid, agent=agent, events=[("user_input", {"text": text})])


def test_default_scope_is_own_agent_and_agent_widens(as_alpha):
    _seed("s_alpha", "alpha", "deploy the thing")
    _seed("s_beta", "beta", "deploy the other thing")

    assert [h["session_id"] for h in search_history("deploy")] == ["s_alpha"]
    assert [h["session_id"] for h in search_history("deploy", agent="beta")] == ["s_beta"]
    assert {h["session_id"] for h in search_history("deploy", agent="*")} == {"s_alpha", "s_beta"}


def test_limit_and_since_apply(as_alpha):
    for i in range(3):
        _seed(f"s{i}", "alpha", "deploy")

    assert len(search_history("deploy", limit=2)) == 2
    assert len(search_history("deploy", since="2000-01-01")) == 3
    assert search_history("deploy", since="2999-01-01") == []


def test_snippet_masks_registered_secrets(as_alpha):
    get_registry().register("api", "hunter2-secret-value", agent="alpha")
    _seed("s1", "alpha", "the password is hunter2-secret-value ok")

    [hit] = search_history("password")

    assert "hunter2-secret-value" not in hit["snippet"]
