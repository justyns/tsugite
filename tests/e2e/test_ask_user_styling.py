"""The ask_user / ask_question prompt buttons must use the app's button styling.

They referenced `.hbtn`, which is scoped to `.console-thread-header`, so inside
`.console-ask-user` they rendered as bare default browser buttons.
"""

from .helpers import CONV_VIEW


def _var_rgb(page, name):
    """Resolve a CSS custom property (hex) to the rgb() form getComputedStyle returns."""
    return page.evaluate(
        "(n) => { const el = document.createElement('div');"
        " el.style.color = getComputedStyle(document.body).getPropertyValue(n).trim();"
        " document.body.appendChild(el); const c = getComputedStyle(el).color; el.remove(); return c; }",
        name,
    )


def _inject_ask_user(page, question_type, options=None):
    page.evaluate(
        """([sel, qt, opts]) => {
            const v = Alpine.$data(document.querySelector(sel));
            v._sessionState(v.selectedSessionId).messages.push({
                type: 'ask_user', question: 'Proceed?', questionType: qt,
                options: opts || [], answered: false, answer: '', inputValue: '',
            });
        }""",
        [CONV_VIEW, question_type, options],
    )
    page.wait_for_selector(".console-ask-user .actions button", timeout=5000)


def test_ask_user_yes_no_buttons_use_app_styling(chat_page):
    page = chat_page
    _inject_ask_user(page, "yes_no")
    page.screenshot(path="/tmp/tsugite-issue-state.png", full_page=True)

    yes_bg = page.locator(".console-ask-user .actions button", has_text="yes").first.evaluate(
        "el => getComputedStyle(el).backgroundColor"
    )
    no_bg = page.locator(".console-ask-user .actions button", has_text="no").first.evaluate(
        "el => getComputedStyle(el).backgroundColor"
    )
    # Affirmative action gets the app's primary (lavender) treatment; secondary
    # gets the surface treatment. On master both were bare browser buttons.
    assert yes_bg == _var_rgb(page, "--lavender"), f"'yes' not primary-styled: {yes_bg}"
    assert no_bg == _var_rgb(page, "--surface0"), f"'no' not secondary-styled: {no_bg}"


def test_ask_user_text_send_button_is_primary(chat_page):
    page = chat_page
    _inject_ask_user(page, "text")
    send_bg = page.locator(".console-ask-user .actions button", has_text="send").first.evaluate(
        "el => getComputedStyle(el).backgroundColor"
    )
    assert send_bg == _var_rgb(page, "--lavender"), f"'send' not primary-styled: {send_bg}"


def test_ask_user_choice_buttons_styled(chat_page):
    page = chat_page
    _inject_ask_user(page, "choice", options=["alpha", "beta"])
    btn = page.locator(".console-ask-user .actions button", has_text="alpha").first
    bg = btn.evaluate("el => getComputedStyle(el).backgroundColor")
    assert bg == _var_rgb(page, "--surface0"), f"choice option not styled: {bg}"
