"""The web UI must not advertise features that don't exist (dead buttons, inert shortcuts)."""

from pathlib import Path

import tsugite_daemon

WEB_DIR = Path(tsugite_daemon.__file__).parent / "web"


def _web_text():
    parts = []
    for path in WEB_DIR.rglob("*"):
        if path.suffix in {".html", ".js", ".css"} and path.is_file():
            parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def test_no_coming_soon_affordances():
    assert "coming soon" not in _web_text().lower()


def test_no_inert_command_palette_shortcut():
    text = _web_text()
    assert "⌘K" not in text


def test_composer_does_not_advertise_unimplemented_tokens():
    index = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert "@ agent" not in index
    assert "# file" not in index
    assert "⌘↩" not in index
