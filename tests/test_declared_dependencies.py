"""Module-scope imports in tsugite/ must resolve to a declared dependency.

Relying on a transitive one works until the direct dependency drops it, and the
break only shows up on a fresh install, never against the lockfile.
"""

import ast
import re
import sys
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _canonical(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _declared_dependencies() -> set[str]:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    return {_canonical(re.split(r"[<>=!~\[;\s]", spec, maxsplit=1)[0]) for spec in pyproject["project"]["dependencies"]}


def _module_scope_imports(path: Path):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Import):
            yield from (alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0]


def test_module_scope_imports_are_declared_dependencies():
    declared = _declared_dependencies()
    providers = packages_distributions()

    undeclared: dict[str, set[str]] = {}
    for path in sorted((REPO_ROOT / "tsugite").rglob("*.py")):
        for name in _module_scope_imports(path):
            if name == "tsugite" or name in sys.stdlib_module_names:
                continue
            if not {_canonical(dist) for dist in providers.get(name, [])} & declared:
                undeclared.setdefault(name, set()).add(str(path.relative_to(REPO_ROOT)))

    assert not undeclared, f"imported but not in pyproject dependencies: {undeclared}"
