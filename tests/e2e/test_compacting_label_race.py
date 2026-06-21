"""sessionCompactingLabel must (1) honor `replaced_count = 0` instead of falling
through to "compacting" via the strict-truthy gate, and (2) communicate that
counts are still pending instead of flashing the same plain "compacting" label
that gets shown when counts never arrive.
"""

from tsugite_daemon.session_store import Session, SessionSource

from tsugite.history.storage import generate_session_id

from .helpers import CONV_VIEW, open_conversations, reload_conversations_view, wait_for_session_in_list


def _make_session(store, user_id):
    sid = generate_session_id("test-agent")
    s = Session(id=sid, agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id=user_id)
    store.create_session(s)
    return s


def _label(page, sid):
    return page.evaluate(
        f"(() => {{ const v = Alpine.$data(document.querySelector({CONV_VIEW!r})); "
        f"return v.sessionCompactingLabel(v.findSession({sid!r})); }})()"
    )


def test_label_shows_zero_replaced_count_instead_of_plain_compacting(authenticated_page, e2e_session_store):
    """`replaced_count = 0` is rare but valid (e.g. session already mostly empty).
    Pre-fix the strict-truthy gate falls through to plain "compacting" even though
    we have valid count data.
    """
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    a = _make_session(e2e_session_store, user_id)
    open_conversations(page)
    reload_conversations_view(page)
    wait_for_session_in_list(page, a.id)

    page.evaluate(
        f"""
        const v = Alpine.$data(document.querySelector({CONV_VIEW!r}));
        const st = v._sessionState({a.id!r});
        st.compacting = true;
        st.compactingCounts = {{ replaced_count: 0, retained_count: 2 }};
        """
    )

    label = _label(page, a.id)
    assert "0" in label, f"label should show 0 turns when replaced_count=0, got {label!r}"
    assert "keeping 2" in label, f"label should show keeping count when retained_count is set, got {label!r}"


def test_label_distinguishes_preparing_from_active_compacting(authenticated_page, e2e_session_store):
    """Pre-counts window: `compacting=true` but no `compactingCounts`. The label
    must signal that work is in flight without conflating it with the "we have
    counts but the user can't read them yet" branch.
    """
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    a = _make_session(e2e_session_store, user_id)
    open_conversations(page)
    reload_conversations_view(page)
    wait_for_session_in_list(page, a.id)

    page.evaluate(
        f"""
        const v = Alpine.$data(document.querySelector({CONV_VIEW!r}));
        const st = v._sessionState({a.id!r});
        st.compacting = true;
        st.compactingCounts = null;
        """
    )

    label = _label(page, a.id)
    assert "preparing" in label.lower(), (
        f"label should distinguish the pre-counts window (e.g. 'preparing to compact…'), got {label!r}"
    )

    page.evaluate(
        f"""
        const v = Alpine.$data(document.querySelector({CONV_VIEW!r}));
        v._sessionState({a.id!r}).compactingCounts = {{ replaced_count: 3, retained_count: 1 }};
        """
    )

    label = _label(page, a.id)
    assert "summariz" in label.lower(), (
        f"label should switch to 'summarizing N turns…' once counts arrive, got {label!r}"
    )
    assert "3 turn" in label, f"label should show replaced_count, got {label!r}"
    assert "keeping 1" in label, f"label should show retained_count, got {label!r}"
