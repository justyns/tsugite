"""File upload via picker and large-paste-as-file flows in the composer."""

import pytest

from tsugite.daemon.session_store import Session, SessionSource
from tsugite.history.storage import generate_session_id

PILL = ".console-pending-files .file-pill"
BANNER = ".console-paste-banner"
CONV = "[x-data=conversationsView]"


def _conv(page, expr):
    return page.evaluate(f"Alpine.$data(document.querySelector('{CONV}')).{expr}")


def _dispatch_paste(page, text):
    page.evaluate(
        """(text) => {
            const ta = document.getElementById('message-input');
            ta.focus();
            const dt = new DataTransfer();
            dt.setData('text/plain', text);
            ta.dispatchEvent(new ClipboardEvent('paste', { clipboardData: dt, bubbles: true, cancelable: true }));
        }""",
        text,
    )


@pytest.fixture
def composer_page(authenticated_page, base_url, e2e_session_store):
    page = authenticated_page
    user_id = page.evaluate("Alpine.store('app').userId")
    session = Session(
        id=generate_session_id("test-agent"),
        agent="test-agent",
        source=SessionSource.INTERACTIVE.value,
        user_id=user_id,
    )
    e2e_session_store.create_session(session)
    # Reload re-runs init() → reload() → autoSelectInteractive(), which picks the seeded
    # interactive session (matching userId). Setting hash post-load skips reload and the
    # fallback selectSession({conversation_id: id}) misses state/source fields, hiding the composer.
    page.goto(base_url)
    page.wait_for_function("!Alpine.store('app').authRequired", timeout=5000)
    page.wait_for_function("Alpine.store('app').selectedAgent", timeout=5000)
    page.wait_for_selector("#message-input", state="visible", timeout=10000)
    return page, session


def test_file_picker_attaches_file(composer_page, tmp_path):
    page, _ = composer_page
    sample = tmp_path / "hello.txt"
    sample.write_text("hello world", encoding="utf-8")

    page.locator("input[type='file']").set_input_files(str(sample))

    page.wait_for_selector(PILL, timeout=5000)
    assert "hello.txt" in (page.locator(PILL).first.text_content() or "")
    assert _conv(page, "pendingFiles.length") == 1


def test_remove_pending_file(composer_page, tmp_path):
    page, _ = composer_page
    sample = tmp_path / "drop.txt"
    sample.write_text("bye", encoding="utf-8")
    page.locator("input[type='file']").set_input_files(str(sample))
    page.wait_for_selector(PILL, timeout=5000)

    page.locator(f"{PILL} button").first.click()

    assert _conv(page, "pendingFiles.length") == 0


def test_send_message_uploads_attached_file(composer_page, e2e_workspace, mock_chat, tmp_path):
    mock_chat("ack")
    page, session = composer_page

    body = "line one\nline two\nline three\n"
    sample = tmp_path / "notes.txt"
    sample.write_text(body, encoding="utf-8")

    page.locator("input[type='file']").set_input_files(str(sample))
    page.wait_for_selector(PILL, timeout=5000)

    page.locator("#message-input").fill("please read the attachment")
    page.locator(".console-send-btn").click()

    page.wait_for_function(
        f"!Alpine.$data(document.querySelector('{CONV}')).sendingBySession['{session.id}']",
        timeout=15000,
    )

    uploads_dir = e2e_workspace / "uploads"
    matches = [p for p in uploads_dir.glob("notes*.txt") if p.read_text(encoding="utf-8") == body]
    assert matches, f"upload not persisted with expected body: {list(uploads_dir.iterdir())}"
    assert _conv(page, "pendingFiles.length") == 0


def test_small_paste_does_not_trigger_banner(composer_page):
    page, _ = composer_page
    _dispatch_paste(page, "just a small note")
    assert _conv(page, "showPasteBanner") is False


@pytest.mark.parametrize(
    "label,payload",
    [
        ("over_500_chars", "x" * 800),
        ("over_11_lines", "\n".join(f"line {i}" for i in range(15))),
    ],
)
def test_large_paste_shows_banner(composer_page, label, payload):
    page, _ = composer_page
    _dispatch_paste(page, payload)
    page.wait_for_selector(BANNER, timeout=3000)
    text = page.locator(BANNER).text_content() or ""
    assert str(len(payload)) in text


def test_accept_paste_as_file(composer_page):
    page, _ = composer_page
    _dispatch_paste(page, "B" * 1200)
    page.wait_for_selector(BANNER, timeout=3000)

    page.locator(f"{BANNER} button", has_text="attach as file").click()

    page.wait_for_selector(PILL, timeout=3000)
    assert "pasted-" in (page.locator(PILL).first.text_content() or "")
    assert _conv(page, "showPasteBanner") is False


def test_dismiss_paste_banner_inserts_text(composer_page):
    page, _ = composer_page
    payload = "Q" * 900
    _dispatch_paste(page, payload)
    page.wait_for_selector(BANNER, timeout=3000)

    page.locator(f"{BANNER} button[aria-label='Dismiss']").click()

    assert page.locator("#message-input").input_value() == payload
    assert _conv(page, "pendingFiles.length") == 0


def test_paste_modal_attaches_custom_filename(composer_page):
    page, _ = composer_page
    page.locator(".console-icon-btn", has_text="＋").click()
    page.locator(".setting-pop-row", has_text="paste as file").click()

    page.wait_for_selector(".paste-modal.open .paste-modal-text", timeout=3000)
    page.locator("#paste-modal-filename").fill("my-snippet.txt")
    page.locator("#paste-modal-text").fill("body content for the snippet")
    page.locator(".paste-modal.open .btn-primary").click()

    page.wait_for_selector(PILL, timeout=3000)
    assert "my-snippet.txt" in (page.locator(PILL).first.text_content() or "")


def test_large_paste_then_send_uploads_file(composer_page, e2e_workspace, mock_chat):
    mock_chat("got it")
    page, session = composer_page

    payload = "PAYLOAD-" + ("Z" * 2000)
    _dispatch_paste(page, payload)
    page.wait_for_selector(BANNER, timeout=3000)
    page.locator(f"{BANNER} button", has_text="attach as file").click()
    page.wait_for_selector(PILL, timeout=3000)

    page.locator("#message-input").fill("please process")
    page.locator(".console-send-btn").click()

    page.wait_for_function(
        f"!Alpine.$data(document.querySelector('{CONV}')).sendingBySession['{session.id}']",
        timeout=15000,
    )

    uploads_dir = e2e_workspace / "uploads"
    matches = [p for p in uploads_dir.glob("pasted-*.txt") if p.read_text(encoding="utf-8") == payload]
    assert matches, f"pasted upload not persisted: {list(uploads_dir.iterdir())}"
