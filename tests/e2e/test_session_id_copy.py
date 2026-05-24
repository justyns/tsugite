"""The first-8-chars session-id breadcrumb segment must be a clickable button that copies the full ID."""


def test_breadcrumb_session_id_is_clickable_and_copies_full_id(chat_page, e2e_session_store):
    page = chat_page
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])

    selected_id = page.evaluate("Alpine.$data(document.querySelector('[x-data*=conversationsView]')).selectedSessionId")
    assert selected_id, "chat_page fixture should have a session selected"

    seg_cur = page.locator(".console-thread-header .path .seg.cur").first
    page.screenshot(path="/tmp/tsugite-issue-state.png", full_page=True)

    tag = seg_cur.evaluate("el => el.tagName.toLowerCase()")
    assert tag == "button", f"breadcrumb session-id segment must be a <button> for click-to-copy, got <{tag}>"

    title = seg_cur.get_attribute("title") or ""
    assert selected_id in title, (
        f"breadcrumb title attribute must expose the full session ID for hover/inspection, got title={title!r}"
    )

    seg_cur.click()

    page.wait_for_selector(".toast-stack .console-toast.toast-success", timeout=2000)
    clipboard = page.evaluate("navigator.clipboard.readText()")
    assert clipboard == selected_id, f"clicking the breadcrumb must copy the full session ID, got {clipboard!r}"
