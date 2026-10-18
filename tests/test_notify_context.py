"""`notify_user` must see the channels a scheduled run configured.

The daemon runs the agent loop, and every tool call inside it, on a worker thread
via `copy_context().run()` (adapters/base.py). A `threading.local` set on the
event-loop thread does not survive that hop, so the tool reported "no channels
configured" and sent nothing while the channels were configured.
"""

import concurrent.futures
import contextvars

from tsugite.tools import notify as notify_module
from tsugite.tools.notify import notify_context, notify_user


def _run_in_worker(fn):
    """Run `fn` the way the daemon runs a tool call: a copied context, other thread."""
    ctx = contextvars.copy_context()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(ctx.run, fn).result()


def test_channels_reach_a_tool_call_on_a_worker_thread(monkeypatch):
    sent = []
    monkeypatch.setattr(notify_module, "send_notification", lambda msg, chans: sent.append(msg) or {"ok": True})

    with notify_context([("discord", object())]):
        result = _run_in_worker(lambda: notify_user("the nightly run finished"))

    assert result == {"ok": True}, f"the tool could not see its own run's channels: {result}"
    assert sent == ["the nightly run finished"]


def test_no_channels_outside_a_notify_context():
    assert "error" in notify_user("nobody asked for this")


def test_channels_do_not_leak_out_of_the_context(monkeypatch):
    monkeypatch.setattr(notify_module, "send_notification", lambda msg, chans: {"ok": True})

    with notify_context([("discord", object())]):
        pass

    assert "error" in notify_user("after the run")


def test_two_concurrent_runs_do_not_see_each_others_channels():
    """Both schedule coroutines run on the one event-loop thread, so thread-local
    scoping gave zero isolation: one run read the other's channels, and whichever
    finished first blinded the other. Each asyncio task gets its own context copy."""
    import asyncio

    seen = {}

    async def run(name, channels):
        with notify_context([(channels, None)]):
            await asyncio.sleep(0)  # yield so the other task interleaves inside the context
            seen[name] = notify_module._channels.get()
            await asyncio.sleep(0)

    async def main():
        await asyncio.gather(run("alice", "alice-ntfy"), run("bob", "bob-slack"))

    asyncio.run(main())

    assert seen["alice"] == [("alice-ntfy", None)], f"alice saw another run's channels: {seen['alice']}"
    assert seen["bob"] == [("bob-slack", None)], f"bob saw another run's channels: {seen['bob']}"


def test_on_notify_runs_with_the_message_and_without_channels():
    seen = []

    with notify_context([], on_notify=lambda message: seen.append(message) or True):
        result = _run_in_worker(lambda: notify_user("card only"))

    assert seen == ["card only"]
    assert result == {"session": {"status": "sent"}}


def test_on_notify_without_a_target_reports_skipped():
    with notify_context([], on_notify=lambda message: False):
        result = _run_in_worker(lambda: notify_user("nobody home"))

    assert result == {"session": {"status": "skipped"}}


def test_on_notify_failure_still_sends_to_channels(monkeypatch):
    sent = []
    monkeypatch.setattr(notify_module, "send_notification", lambda msg, chans: sent.append(msg) or {"ok": True})

    def broken_card(message):
        raise RuntimeError("store down")

    with notify_context([("discord", object())], on_notify=broken_card):
        result = _run_in_worker(lambda: notify_user("disk is full"))

    assert sent == ["disk is full"]
    assert result == {"ok": True, "session": {"status": "failed", "error": "store down"}}
