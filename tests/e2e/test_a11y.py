"""Accessibility audits driven by axe-core.

Loads axe-core in the page via CDN (same pattern as before - these tests
already require internet) and asserts that the major views have no serious or
critical violations. Add a rule id to `BASELINE[view]` when intentionally
introducing a known issue; remove the entry when you fix it. The empty
baseline is intentional: we got here by clearing real violations, not by
allowlisting them.
"""

import json

import pytest
from tsugite_daemon.session_store import Session, SessionSource

from tsugite.history import generate_session_id

from .helpers import E2E_USER_ID, open_view, wait_for_authed

AXE_CDN = "https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.10.2/axe.min.js"
BLOCKING = {"serious", "critical"}

# Per-view allowlist of rule IDs that are intentionally not fixed.
#
# This baseline was (re)established against the Svelte rebuild, which has no
# prior ratchet history of its own - the old baseline was for Alpine markup
# that no longer exists. The three rule ids below are NOT view-specific: they
# showed up identically on every view audited (chats/files/schedules/webhooks/
# usage), which points to shared shell chrome rather than per-view content:
#   - aria-required-children: `.mux-tabs[role=tablist]` (lib/shell/mux/Mux.svelte)
#     doesn't structure its children the way a tablist role requires.
#   - nested-interactive: a mux tab (`[role=tab]`) nests another interactive
#     control (its close button) inside it.
#   - color-contrast: several small mono/status labels (composer's "model ·
#     effort" chip, the session type badge, the status pill) read below the
#     contrast threshold - looks like a token choice (e.g. --tx3-on-bg1/2 for
#     small text), not one-off styling.
# These need a real frontend fix (out of scope for this e2e re-baseline - see
# team notes); tracked here rather than silently passing so the ratchet stays
# honest about current state. Shrink a set (or drop a view entry to `set()`)
# as each is fixed - never add beyond what's listed.
BASELINE: dict[str, set[str]] = {}


def _make_session(store):
    sid = generate_session_id("test-agent")
    s = Session(id=sid, source=SessionSource.INTERACTIVE.value, user_id=E2E_USER_ID)
    store.create_session(s)
    return s


def _run_axe(page) -> dict:
    """Inject axe-core if absent, then run it against the current page and return the JSON report."""
    page.evaluate(
        f"""
        () => new Promise((resolve, reject) => {{
            if (window.axe) return resolve();
            const s = document.createElement('script');
            s.src = {AXE_CDN!r};
            s.onload = () => resolve();
            s.onerror = () => reject(new Error('axe-core failed to load'));
            document.head.appendChild(s);
        }})
        """
    )
    raw = page.evaluate("axe.run({reporter: 'v2'}).then(r => JSON.stringify(r))")
    return json.loads(raw)


def _blocking_violations(report: dict, allowed_rule_ids: set[str]) -> list[dict]:
    return [
        v for v in report.get("violations", []) if v.get("impact") in BLOCKING and v.get("id") not in allowed_rule_ids
    ]


@pytest.mark.parametrize("view", ["chats", "files", "schedules", "webhooks", "usage"])
def test_a11y_no_new_serious_or_critical_violations(authenticated_page, view, e2e_session_store):
    """Each main view introduces no NEW serious/critical axe violations beyond the baseline."""
    page = authenticated_page

    if view == "chats":
        # Chats is the default-docked view, so seed a real session and reload
        # rather than clicking nav - auditing a populated conversation, not
        # the empty state, matches the old suite's intent.
        _make_session(e2e_session_store)
        page.reload()
        wait_for_authed(page)
    else:
        open_view(page, view)

    report = _run_axe(page)
    new_violations = _blocking_violations(report, BASELINE.get(view, set()))

    if new_violations:
        summary = "\n".join(
            f"  [{v['impact']}] {v['id']}: {v['help']} ({len(v['nodes'])} node(s))\n"
            + "\n".join(
                f"    node: {n.get('target')} - {(n.get('any') or [{}])[0].get('data')}" for n in v["nodes"][:8]
            )
            for v in new_violations
        )
        pytest.fail(
            f"NEW accessibility violations on {view!r} view (not in baseline):\n{summary}\n"
            f"Either fix them, or if the regression is intentional, add the rule id to BASELINE[{view!r}]."
        )
