"""Two small composer/thread rendering bugs:

1. The 'progress:' info bubble rendered agent markdown as plain text (x-text)
   while every other prose bubble goes through renderHtml/marked.
2. Pasting large text inline (via the paste banner) set messageText
   programmatically, so the @input-bound auto-resize never fired and the
   textarea stayed collapsed until the next keystroke.
"""

from .helpers import CONV_VIEW


def test_info_bubble_renders_markdown(chat_page):
    page = chat_page

    page.evaluate(
        """(sel) => {
            const v = Alpine.$data(document.querySelector(sel));
            const sid = v.selectedSessionId;
            v._handleProgressEvent(
                {type: 'info', message: 'see **bold move** and `inline_code` at [docs](https://example.com)'},
                sid,
            );
        }""",
        CONV_VIEW,
    )
    bubble = page.locator(".console-turn.info .console-turn-bubble").last
    bubble.wait_for(state="visible", timeout=3000)
    page.screenshot(path="/tmp/tsugite-issue-424-state.png", full_page=True)

    html = bubble.inner_html()
    text = bubble.inner_text()
    assert "**bold move**" not in text, "markdown must be rendered, not shown literally"
    assert "<strong>bold move</strong>" in html
    assert "<code>inline_code</code>" in html
    assert 'href="https://example.com"' in html


def test_inline_paste_grows_textarea(chat_page):
    page = chat_page

    heights = page.evaluate(
        """(sel) => {
            const v = Alpine.$data(document.querySelector(sel));
            const ta = v.$refs.messageInput;
            const before = ta.offsetHeight;
            const bigText = Array.from({length: 40}, (_, i) => `pasted line ${i}`).join('\\n');
            v.onPaste({
                clipboardData: { getData: () => bigText },
                preventDefault: () => {},
            });
            const bannerShown = v.showPasteBanner;
            v.dismissPasteBanner();
            return new Promise(resolve => setTimeout(() => resolve({
                before,
                after: ta.offsetHeight,
                bannerShown,
                hasText: v.messageText.includes('pasted line 39'),
            }), 50));
        }""",
        CONV_VIEW,
    )
    page.screenshot(path="/tmp/tsugite-issue-432-state.png", full_page=True)

    assert heights["bannerShown"], "large paste must trigger the banner path"
    assert heights["hasText"], "inline paste must land in messageText"
    assert heights["after"] > heights["before"], (
        f"textarea must grow to fit pasted text without a keystroke "
        f"(before={heights['before']}px, after={heights['after']}px)"
    )
