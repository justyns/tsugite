"""Tests for the plugin UI surface registry."""

from pathlib import Path

from tsugite.ui_surfaces import register_ui_surface, registered_ui_surfaces, ui_surface


class TestRegisterUISurface:
    def test_registers_the_descriptor_authors_write(self):
        assets = Path("/plugins/dash/ui")
        register_ui_surface(
            kind="doc",
            label="Document",
            icon="files",
            entry="ui/editor.html",
            assets=assets,
            nav=True,
            mode="workspace",
            params=["path"],
            events=["doc_update"],
        )

        assert registered_ui_surfaces() == {
            "": [
                {
                    "kind": "doc",
                    "label": "Document",
                    "icon": "files",
                    "entry": "ui/editor.html",
                    "assets": assets,
                    "nav": True,
                    "mode": "workspace",
                    "params": ["path"],
                    "events": ["doc_update"],
                }
            ]
        }

    def test_omits_what_the_caller_left_out(self):
        """The daemon's collector owns the defaults, so absent keys stay absent."""
        register_ui_surface(kind="dash", entry="ui/index.html")

        assert registered_ui_surfaces() == {"": [{"kind": "dash", "entry": "ui/index.html"}]}


class TestUISurfaceDecorator:
    def test_registers_the_page_callable(self):
        @ui_surface(kind="dash", label="Homelab", nav=True)
        def dashboard_page() -> str:
            return "<h1>ok</h1>"

        assert registered_ui_surfaces() == {
            "": [{"kind": "dash", "label": "Homelab", "nav": True, "page": dashboard_page}]
        }

    def test_returns_the_function_unchanged(self):
        def page() -> str:
            return "<h1>ok</h1>"

        assert ui_surface(kind="dash")(page) is page
        assert page() == "<h1>ok</h1>"
