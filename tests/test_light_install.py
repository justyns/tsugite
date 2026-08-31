"""Phase 0: a base install without optional deps must import and degrade gracefully.

These tests prove the kernel + default batteries stand alone: importing the tool
modules must not require ddgs / daemon / web deps, and the tools that do need them
must raise a clear "install the extra" error instead of a confusing traceback.
"""

import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest


def test_extract_article_without_readability(monkeypatch):
    """The readability extraction path raises a clear install hint when missing."""
    monkeypatch.setitem(sys.modules, "readability", None)
    from tsugite.tools.http import _extract_article

    with pytest.raises(RuntimeError, match=r"tsugite-cli\[web\]"):
        _extract_article("<html><body><p>hi</p></body></html>")


def test_schedule_module_imports_without_daemon():
    """tools.schedule must import even when tsugite_daemon.scheduler is unimportable."""
    script = textwrap.dedent(
        """
        import sys
        class _Blocker:
            def find_spec(self, name, path=None, target=None):
                if name == "tsugite_daemon.scheduler" or name.split(".")[0] == "cronsim":
                    raise ModuleNotFoundError(name)
                return None
        sys.meta_path.insert(0, _Blocker())
        import tsugite.tools.schedule as s
        assert hasattr(s, "set_scheduler")
        print("OK")
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_core_tools_load_without_optional_deps():
    """_ensure_tools_loaded() must succeed with every optional dep blocked."""
    script = textwrap.dedent(
        """
        import sys
        BLOCKED = {"ddgs", "cronsim", "discord", "starlette", "uvicorn",
                   "readability", "youtube_transcript_api", "pywebpush", "py_vapid"}
        class _Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ModuleNotFoundError(name)
                return None
        sys.meta_path.insert(0, _Blocker())
        import tsugite.tools as t
        t._ensure_tools_loaded()
        assert "fetch_json" in t._tools
        print("OK")
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


AGENT_RUNTIME_MODULES = ("tsugite.core.agent", "tsugite.agent_runner.runner")


def modules_loaded_by_cli(*argv: str) -> set[str]:
    """Names in sys.modules after a fresh interpreter runs the CLI on argv."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "modules.json"
        script = textwrap.dedent(
            f"""
            import json
            import sys
            from pathlib import Path

            sys.argv = ["tsu", *sys.argv[1:]]
            from tsugite.tsugite import app

            try:
                app()
            except SystemExit:
                pass
            Path({str(out)!r}).write_text(json.dumps(sorted(sys.modules)))
            """
        )
        result = subprocess.run([sys.executable, "-c", script, *argv], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        return set(json.loads(out.read_text()))


def test_version_does_not_import_agent_runtime():
    loaded = modules_loaded_by_cli("--version")
    assert not loaded.intersection(AGENT_RUNTIME_MODULES)
