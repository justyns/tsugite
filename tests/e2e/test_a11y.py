"""Accessibility audits driven by axe-core.

Loads axe-core in the page via CDN (same pattern as Alpine — these tests
already require internet) and asserts that the major Console views have no
NEW serious or critical violations beyond a known baseline. The baseline
documents pre-existing issues so this test acts as a ratchet: any new rule
hit fails the build immediately, but day-one shipping issues don't.

Documented baseline (fix these and tighten the allowlist as they get
addressed):
- color-contrast (all views): theme colors below 4.5:1 in some places
- nested-interactive (conversations): a button nests another interactive element
- scrollable-region-focusable (usage view): one keyboard-inaccessible region
- select-name (usage view): two unlabelled <select> elements
"""

import json

import pytest

from tsugite.daemon.session_store import Session, SessionSource
from tsugite.history.storage import generate_session_id

from .helpers import CONV_VIEW, open_conversations, reload_conversations_view


AXE_CDN = "https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.10.2/axe.min.js"
BLOCKING = {"serious", "critical"}

# Per-view allowlist of rule IDs that currently fail but predate this test.
# Remove entries as they get fixed in the templates.
BASELINE = {
    "conversations": {"color-contrast", "nested-interactive"},
    "workspace": {"color-contrast"},
    "schedules": {"color-contrast"},
    "webhooks": {"color-contrast"},
    "usage": {"color-contrast", "scrollable-region-focusable", "select-name"},
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
        v
        for v in report.get("violations", [])
        if v.get("impact") in BLOCKING and v.get("id") not in allowed_rule_ids
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
            f"  [{v['impact']}] {v['id']}: {v['help']} ({len(v['nodes'])} node(s))"
            for v in new_violations
        )
        pytest.fail(
            f"NEW accessibility violations on {view!r} view (not in baseline):\n{summary}\n"
            f"Either fix them, or if the regression is intentional, add the rule id to BASELINE[{view!r}]."
        )
