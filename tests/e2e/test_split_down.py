"""Dropping a surface on the bottom third of a pane stacks it below."""

from playwright.sync_api import expect

PANE = '[data-testid="mux-pane"]'

DRAG_BOTTOM = """(pane, type) => {
  const r = pane.getBoundingClientRect();
  window.dt ??= new DataTransfer();
  window.dt.setData('application/x-tsugite-surface', JSON.stringify({ kind: 'terminal', params: {} }));
  pane.dispatchEvent(new DragEvent(type, {
    bubbles: true, cancelable: true, dataTransfer: window.dt,
    clientX: r.left + r.width / 2, clientY: r.top + r.height * 0.9,
  }));
}"""


def test_dropping_on_the_bottom_third_stacks_the_pane_below(chat_page):
    page = chat_page
    pane = page.locator(PANE).first
    pane.evaluate(DRAG_BOTTOM, "dragover")
    pane.evaluate(DRAG_BOTTOM, "drop")

    expect(page.locator(PANE)).to_have_count(2)
    top, bottom = (page.locator(PANE).nth(i).bounding_box() for i in (0, 1))
    assert bottom["y"] >= top["y"] + top["height"]
