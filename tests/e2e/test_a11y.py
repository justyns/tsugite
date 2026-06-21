"""Accessibility audits driven by axe-core.

Loads axe-core in the page via CDN (same pattern as Alpine — these tests
already require internet) and asserts that the major Console views have no
serious or critical violations. Add a rule id to `BASELINE[view]` when
intentionally introducing a known issue; remove the entry when you fix it.
The empty baseline is intentional: we got here by clearing real violations,
not by allowlisting them.
"""

import json

import pytest
from tsugite_daemon.session_store import Session, SessionSource

from tsugite.history.storage import generate_session_id

from .helpers import open_conversations, reload_conversations_view

AXE_CDN = "https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.10.2/axe.min.js"
BLOCKING = {"serious", "critical"}

# Per-view allowlist of rule IDs that are intentionally not fixed.
# Empty = the suite expects zero serious/critical violations.
BASELINE: dict[str, set[str]] = {
    "conversations": set(),
    "workspace": set(),
    "schedules": set(),
    "webhooks": set(),
    "usage": set(),
}


def _make_session(store, user_id):
    sid = generate_session_id("test-agent")
    s = Session(id=sid, agent="test-agent", source=SessionSource.INTERACTIVE.value, user_id=user_id)
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


@pytest.mark.parametrize("view", ["conversations", "workspace", "schedules", "webhooks", "usage"])
def test_a11y_no_new_serious_or_critical_violations(authenticated_page, view, e2e_session_store):
    """Each main view introduces no NEW serious/critical axe violations beyond the baseline."""
    page = authenticated_page

    if view == "conversations":
        user_id = page.evaluate("Alpine.store('app').userId")
        _make_session(e2e_session_store, user_id)
        open_conversations(page)
        reload_conversations_view(page)
    else:
        page.locator(".console-tabs button.console-tab", has_text=view).click()
        page.wait_for_function(f"Alpine.store('app').view === {view!r}", timeout=3000)

    report = _run_axe(page)
    new_violations = _blocking_violations(report, BASELINE.get(view, set()))

    if new_violations:
        summary = "\n".join(
            f"  [{v['impact']}] {v['id']}: {v['help']} ({len(v['nodes'])} node(s))" for v in new_violations
        )
        pytest.fail(
            f"NEW accessibility violations on {view!r} view (not in baseline):\n{summary}\n"
            f"Either fix them, or if the regression is intentional, add the rule id to BASELINE[{view!r}]."
        )
