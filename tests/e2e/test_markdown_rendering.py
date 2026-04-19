"""E2E tests for markdown rendering in `.msg.agent` bubbles.

Written before swapping the hand-rolled renderer in `utils.js` for the `marked`
library. Regression cases pin pre-existing behavior; the table/alignment/style
cases drive the new behavior.
"""

from unittest.mock import patch

from tsugite.history.storage import SessionStorage


def _seed_agent_turn(e2e_adapter, e2e_tmp, label, final_answer):
    """Seed a fresh session whose single turn has the given markdown final_answer."""
    unique_user = f"md-user-{label}"
    session = e2e_adapter.session_store.get_or_create_interactive(unique_user, "test-agent")
    history_dir = e2e_tmp / f"history-{label}"
    history_dir.mkdir(exist_ok=True)
    session_path = history_dir / f"{session.id}.jsonl"

    storage = SessionStorage.create("test-agent", model="test", session_path=session_path)
    storage.record_turn(
        messages=[{"role": "user", "content": "show"}],
        final_answer=final_answer,
    )
    return history_dir, unique_user, session.id


def _open_session(page, user_id, session_id):
    page.evaluate(f"localStorage.setItem('tsugite_user_id', {user_id!r})")
    page.goto(page.url.split("#")[0] + f"#conversations?session={session_id}")
    page.reload()
    page.wait_for_function("!Alpine.store('app').authRequired", timeout=5000)
    page.wait_for_function(f"Alpine.store('app').userId === {user_id!r}", timeout=3000)
    page.wait_for_selector(".msg.agent", timeout=5000)


def test_markdown_regression_basic_formatting(authenticated_page, e2e_adapter, e2e_tmp):
    """All pre-existing markdown features still render after the parser swap."""
    page = authenticated_page

    md = (
        "# H1\n"
        "## H2\n"
        "### H3\n"
        "#### H4\n\n"
        "A paragraph with **bold**, *italic*, and `inline`.\n\n"
        "- ul item 1\n"
        "- ul item 2\n\n"
        "1. ol item 1\n"
        "2. ol item 2\n\n"
        "> a blockquote\n\n"
        "---\n\n"
        "A [link](https://example.com).\n\n"
        "```python\nprint('hello')\n```\n"
    )

    history_dir, user_id, session_id = _seed_agent_turn(e2e_adapter, e2e_tmp, "regression", md)

    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
        _open_session(page, user_id, session_id)
        agent = page.locator(".msg.agent").last

        assert agent.locator("h1").count() >= 1
        assert agent.locator("h2").count() >= 1
        assert agent.locator("h3").count() >= 1
        assert agent.locator("h4").count() >= 1

        assert agent.locator("strong").first.text_content() == "bold"
        assert agent.locator("em").first.text_content() == "italic"
        assert agent.locator("code").first.text_content() == "inline"

        assert agent.locator("ul > li").count() == 2
        assert agent.locator("ol > li").count() == 2

        assert agent.locator("blockquote").count() == 1
        assert agent.locator("hr").count() == 1

        link = agent.locator("a").first
        assert link.get_attribute("href") == "https://example.com"

        pre_code = agent.locator("pre code")
        assert pre_code.count() == 1
        assert "print('hello')" in (pre_code.first.text_content() or "")


def test_markdown_gfm_table_renders(authenticated_page, e2e_adapter, e2e_tmp):
    """Simple GFM table produces <table><thead><th>...<tbody><tr><td>."""
    page = authenticated_page

    md = (
        "| Name | Score |\n"
        "| ---- | ----- |\n"
        "| Alice | 10 |\n"
        "| Bob | 20 |\n"
    )

    history_dir, user_id, session_id = _seed_agent_turn(e2e_adapter, e2e_tmp, "table", md)

    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
        _open_session(page, user_id, session_id)
        agent = page.locator(".msg.agent").last

        assert agent.locator("table").count() == 1
        assert agent.locator("table thead th").count() == 2
        assert agent.locator("table tbody tr").count() == 2
        assert (agent.locator("table thead th").first.text_content() or "").strip() == "Name"
        last_cell = agent.locator("table tbody tr").nth(1).locator("td").last
        assert (last_cell.text_content() or "").strip() == "20"


def test_markdown_gfm_table_alignment(authenticated_page, e2e_adapter, e2e_tmp):
    """GFM alignment syntax yields text-align on th/td via inline style."""
    page = authenticated_page

    md = (
        "| L | C | R |\n"
        "| :--- | :---: | ---: |\n"
        "| a | b | c |\n"
    )

    history_dir, user_id, session_id = _seed_agent_turn(e2e_adapter, e2e_tmp, "align", md)

    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
        _open_session(page, user_id, session_id)
        agent = page.locator(".msg.agent").last

        ths = agent.locator("table thead th")
        tds = agent.locator("table tbody tr").first.locator("td")

        def _align(loc, i):
            return loc.nth(i).evaluate("el => getComputedStyle(el).textAlign")

        assert _align(ths, 0) in ("left", "start", "-webkit-left")
        assert _align(ths, 1) in ("center", "-webkit-center")
        assert _align(ths, 2) in ("right", "end", "-webkit-right")
        assert _align(tds, 0) in ("left", "start", "-webkit-left")
        assert _align(tds, 1) in ("center", "-webkit-center")
        assert _align(tds, 2) in ("right", "end", "-webkit-right")


def test_markdown_table_styling(authenticated_page, e2e_adapter, e2e_tmp):
    """Computed CSS matches the design: no uppercase header, last row no border."""
    page = authenticated_page

    md = (
        "| h1 | h2 |\n"
        "| --- | --- |\n"
        "| a | b |\n"
        "| c | d |\n"
    )

    history_dir, user_id, session_id = _seed_agent_turn(e2e_adapter, e2e_tmp, "style", md)

    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
        _open_session(page, user_id, session_id)
        agent = page.locator(".msg.agent").last

        th = agent.locator("table th").first
        assert th.evaluate("el => getComputedStyle(el).textTransform") == "none"
        weight = th.evaluate("el => getComputedStyle(el).fontWeight")
        assert weight in ("600", "700", "bold"), f"unexpected th font-weight: {weight}"

        last_td = agent.locator("table tbody tr").last.locator("td").first
        border = last_td.evaluate(
            "el => ({ style: getComputedStyle(el).borderBottomStyle, "
            "width: getComputedStyle(el).borderBottomWidth })"
        )
        assert border["style"] == "none" or border["width"] == "0px", (
            f"last row should have no bottom border; got {border}"
        )


def test_markdown_wide_table_scrolls_inside_bubble(authenticated_page, e2e_adapter, e2e_tmp):
    """A wide table scrolls inside the bubble; the page itself does not overflow."""
    page = authenticated_page

    cols = 8
    header = "| " + " | ".join(f"col{i}" for i in range(cols)) + " |\n"
    sep = "| " + " | ".join("---" for _ in range(cols)) + " |\n"
    row = "| " + " | ".join("very long cell content " * 3 for _ in range(cols)) + " |\n"
    md = header + sep + row + row

    history_dir, user_id, session_id = _seed_agent_turn(e2e_adapter, e2e_tmp, "wide-table", md)

    page.set_viewport_size({"width": 400, "height": 800})

    with patch("tsugite.daemon.adapters.http.get_history_dir", return_value=history_dir):
        _open_session(page, user_id, session_id)
        table = page.locator(".msg.agent table").last

        dims = table.evaluate(
            "el => ({ scrollWidth: el.scrollWidth, clientWidth: el.clientWidth })"
        )
        assert dims["scrollWidth"] > dims["clientWidth"], (
            f"expected table to be horizontally scrollable; got {dims}"
        )

        page_dims = page.evaluate(
            "() => ({ scrollWidth: document.documentElement.scrollWidth, "
            "clientWidth: document.documentElement.clientWidth })"
        )
        assert page_dims["scrollWidth"] <= page_dims["clientWidth"] + 1, (
            f"page itself should not horizontally scroll; got {page_dims}"
        )
