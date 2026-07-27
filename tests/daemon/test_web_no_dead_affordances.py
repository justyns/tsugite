"""The web UI must not advertise features that don't exist (dead buttons, inert shortcuts).

Scans the MOUNTED surfaces: views, the app shell, and the hand-authored shell files
(``frontend/index.html``, ``frontend/public/``). The component library
(``src/lib/components/``) is deliberately excluded — a component's own markup is a
definition, not an advertisement; the affordance only exists once a mounted surface
renders it, and that surface is what gets scanned. The dist under
``tsugite_daemon/web/`` is generated (Vite); we only assert it's present so a
missing build is caught here rather than at runtime.
"""

from functools import lru_cache
from pathlib import Path

import pytest
import tsugite_daemon

WEB_DIR = Path(tsugite_daemon.__file__).parent / "web"
FRONTEND = WEB_DIR.parent.parent / "frontend"
FRONTEND_SRC = FRONTEND / "src"

requires_src = pytest.mark.skipif(not FRONTEND_SRC.is_dir(), reason="frontend/src not present")


@lru_cache(maxsize=1)
def _source_text() -> str:
    mounted_roots = (
        (FRONTEND_SRC / "views", {".svelte", ".ts"}),
        (FRONTEND_SRC / "lib" / "shell", {".svelte", ".ts"}),
        (FRONTEND / "public", {".js", ".json", ".html"}),
    )
    parts = []
    for root, suffixes in mounted_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.suffix in suffixes and path.is_file():
                parts.append(path.read_text(encoding="utf-8"))
    for single in (FRONTEND_SRC / "App.svelte", FRONTEND / "index.html"):
        if single.is_file():
            parts.append(single.read_text(encoding="utf-8"))
    return "\n".join(parts)


@pytest.mark.skipif(
    FRONTEND_SRC.is_dir() and not (WEB_DIR / "index.html").exists(),
    reason="source checkout without a local web build (dist is gitignored; run `mise run web-build`)",
)
def test_built_dist_present():
    """Installed packages must bundle the UI; a source checkout may not have built it yet."""
    assert (WEB_DIR / "index.html").exists(), "web UI dist missing - run `mise run web-build`"


@requires_src
def test_no_coming_soon_affordances():
    assert "coming soon" not in _source_text().lower()


@requires_src
def test_command_palette_shortcut_is_wired():
    # The palette is real now (searchable commands + chat sessions), so the ⌘K
    # affordance must exist AND be wired - the old assert-absent guard flips to
    # asserting the advertised shortcut has a live opener.
    text = _source_text()
    assert "⌘K" in text
    assert "openPalette" in text


@requires_src
def test_composer_does_not_advertise_unimplemented_tokens():
    text = _source_text()
    assert "@ agent" not in text
    assert "# file" not in text
    assert "⌘↩" not in text
