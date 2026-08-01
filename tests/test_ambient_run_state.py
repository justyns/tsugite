"""Per-run ambient state must not be shared between concurrent runs.

The daemon runs each agent loop in an `asyncio.to_thread` worker, so several
runs are in flight at once. State scoped to "the current run" therefore has to
be context-local; a plain module global silently hands one run another run's
policy.
"""

import asyncio
import threading

import pytest

from tsugite.agent_runner.helpers import (
    get_allowed_agents,
    get_allowed_secrets,
    set_allowed_agents,
    set_allowed_secrets,
)


@pytest.mark.asyncio
async def test_allowed_secrets_do_not_leak_between_concurrent_runs():
    """Agent A's secret allowlist must not become agent B's.

    `get_secret` gates on this list and treats an empty list as "all allowed",
    so a clobbered value either hands a restricted agent the full keyring or
    locks an unrestricted one out.
    """
    observed: dict[str, list] = {}
    both_have_set = threading.Barrier(2, timeout=5)

    def run_as(name: str, allowlist: list[str]):
        set_allowed_secrets(allowlist)
        both_have_set.wait()
        observed[name] = get_allowed_secrets()

    await asyncio.gather(
        asyncio.to_thread(run_as, "restricted", ["only_this_one"]),
        asyncio.to_thread(run_as, "unrestricted", []),
    )

    assert observed["restricted"] == ["only_this_one"], (
        f"restricted run saw another run's secret allowlist: {observed['restricted']}"
    )
    assert observed["unrestricted"] == []


@pytest.mark.asyncio
async def test_allowed_agents_do_not_leak_between_concurrent_runs():
    """Same contract for the spawn allowlist used by multi-agent mode.

    Both runs are held at a barrier after setting their policy, so the second
    write always lands before the first read. Without that the threads finish
    too fast to interleave and a shared global would pass by luck.
    """
    observed: dict[str, list] = {}
    both_have_set = threading.Barrier(2, timeout=5)

    def run_as(name: str, allowlist: list[str]):
        set_allowed_agents(allowlist)
        both_have_set.wait()
        observed[name] = get_allowed_agents()

    await asyncio.gather(
        asyncio.to_thread(run_as, "a", ["helper_a"]),
        asyncio.to_thread(run_as, "b", ["helper_b"]),
    )

    assert observed["a"] == ["helper_a"], f"run A saw run B's spawn allowlist: {observed['a']}"
    assert observed["b"] == ["helper_b"], f"run B saw run A's spawn allowlist: {observed['b']}"


@pytest.mark.asyncio
async def test_a_nested_run_restores_the_parents_policy():
    """A hook agent runs inside the parent run's context, not a fresh task.

    `hooks.py` awaits `run_agent_async` through `asyncio.wait_for`, which on
    3.12+ does not wrap its argument in a Task. So the nested run shares the
    caller's context: without token-scoped reset, the child's policy overwrites
    the parent's for the rest of the parent's run, and the child's teardown
    leaves the parent's spawn allowlist as None - which reads as UNRESTRICTED.
    """
    from tsugite.agent_runner.helpers import clear_allowed_agents

    set_allowed_secrets(["parent-only"])
    set_allowed_agents(["parent-helper"])

    async def nested_run():
        secrets_token = set_allowed_secrets(["child-only"])
        agents_token = set_allowed_agents(["child-helper"])
        try:
            assert get_allowed_secrets() == ["child-only"]
        finally:
            _reset(secrets_token, agents_token)

    def _reset(secrets_token, agents_token):
        from tsugite.agent_runner.helpers import reset_allowed_agents, reset_allowed_secrets

        reset_allowed_secrets(secrets_token)
        reset_allowed_agents(agents_token)

    await asyncio.wait_for(nested_run(), timeout=5)

    assert get_allowed_secrets() == ["parent-only"], (
        f"nested run clobbered the parent's secret allowlist: {get_allowed_secrets()}"
    )
    assert get_allowed_agents() == ["parent-helper"], (
        f"nested run clobbered the parent's spawn allowlist: {get_allowed_agents()}"
    )
    clear_allowed_agents()


@pytest.mark.asyncio
async def test_a_nested_run_restores_the_parents_sandbox_context():
    """Same shape, higher stakes: the sandbox gate fails OPEN.

    `_execute_agent_with_prompt` clears the sandbox context unconditionally in
    its finally. A nested agent hook runs one on the same thread, so after it
    returns `deny_when_sandboxed` stops denying and `spawn_agent` stops
    propagating isolation to children, for the rest of a sandboxed parent's run.
    """
    from tsugite.agent_runner.helpers import (
        SandboxContext,
        get_sandbox_context,
        reset_sandbox_context,
        set_sandbox_context,
    )

    parent = SandboxContext(no_network=True)
    set_sandbox_context(parent)

    async def nested_run():
        token = set_sandbox_context(SandboxContext(allow_domains=["child.example"]))
        try:
            assert get_sandbox_context().allow_domains == ["child.example"]
        finally:
            reset_sandbox_context(token)

    await asyncio.wait_for(nested_run(), timeout=5)

    assert get_sandbox_context() is parent, "nested run dropped the parent's sandbox policy"
    reset_sandbox_context(set_sandbox_context(None))
