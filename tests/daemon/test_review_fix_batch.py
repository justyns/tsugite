"""Tests for the bug-hunt fixes: VAPID claim isolation, compaction-chain
walking, superseded-compact guard, webhook hardening, JSON-safe webhook
templates."""

import json
import sys
import types

import pytest

from tsugite.daemon.gateway import _render_webhook_body
from tsugite.daemon.session_store import Session, SessionSource, SessionStore

# ── push: shared claims must not be mutated across sends ──


@pytest.mark.asyncio
async def test_send_web_push_does_not_let_claims_mutation_leak(monkeypatch, tmp_path):
    """pywebpush pins `aud` onto the claims dict it receives. With a shared
    dict, the first subscriber's push origin poisons every later send."""
    from tsugite.daemon import push as push_mod

    seen_auds = []

    def fake_webpush(subscription_info, data, vapid_private_key, vapid_claims):
        # Simulate pywebpush's in-place aud pinning.
        if not vapid_claims.get("aud"):
            vapid_claims["aud"] = subscription_info["endpoint"].rsplit("/", 2)[0]
        seen_auds.append(vapid_claims["aud"])

    fake_mod = types.SimpleNamespace(webpush=fake_webpush, WebPushException=Exception)
    monkeypatch.setitem(sys.modules, "pywebpush", fake_mod)

    shared_claims = {"sub": "mailto:t@example.com"}
    sub_a = {"endpoint": "https://fcm.googleapis.com/send/aaa", "keys": {}}
    sub_b = {"endpoint": "https://updates.push.mozilla.com/send/bbb", "keys": {}}
    await push_mod.send_web_push(sub_a, {"title": "x"}, "key", shared_claims)
    await push_mod.send_web_push(sub_b, {"title": "x"}, "key", shared_claims)

    assert "aud" not in shared_claims, "caller's claims dict must stay pristine"
    assert seen_auds[1].startswith("https://updates.push.mozilla.com"), (
        f"second send must get ITS endpoint's aud, got {seen_auds}"
    )


# ── session store: compaction chains ──


def _mk(store, sid):
    s = Session(id=sid, agent="a", source=SessionSource.INTERACTIVE.value, user_id="u")
    store.create_session(s)
    return s


def test_resolve_compacted_successor_walks_multi_hop_chain(tmp_path):
    """A stale tab can hold a session compacted twice (A→B→C); resolution must
    land on the live tail, not the intermediate superseded session."""
    store = SessionStore(tmp_path / "s.json")
    _mk(store, "a")
    _mk(store, "b")
    _mk(store, "c")
    store._sessions["a"].superseded_by = "b"
    store._sessions["b"].superseded_by = "c"
    resolved = store.resolve_compacted_successor("a")
    assert resolved is not None and resolved.id == "c"


def test_resolve_compacted_successor_tolerates_cycles(tmp_path):
    store = SessionStore(tmp_path / "s.json")
    _mk(store, "a")
    _mk(store, "b")
    store._sessions["a"].superseded_by = "b"
    store._sessions["b"].superseded_by = "a"  # corrupt chain
    resolved = store.resolve_compacted_successor("a")
    assert resolved is not None  # must terminate, not loop forever


def test_compact_session_refuses_already_superseded(tmp_path):
    """Re-compacting a superseded session would fork the chain (two live
    successors, nondeterministic routing)."""
    store = SessionStore(tmp_path / "s.json")
    _mk(store, "a")
    _mk(store, "b")
    store._sessions["a"].superseded_by = "b"
    store._sessions["a"].message_count = 5
    with pytest.raises(ValueError, match="already compacted"):
        store.compact_session("a")


# ── gateway: webhook template substitution ──


def test_render_webhook_body_escapes_for_json_templates():
    template = '{"text": "{message}"}'
    msg = 'done", "channel": "#admin\nnewline'
    body = _render_webhook_body(template, msg)
    parsed = json.loads(body)
    assert parsed == {"text": msg}, "message must arrive intact as a single JSON string"


def test_render_webhook_body_leaves_plain_text_templates_alone():
    template = "ALERT: {message}"
    body = _render_webhook_body(template, 'line1\n"quoted"')
    assert body == 'ALERT: line1\n"quoted"'
