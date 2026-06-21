#!/usr/bin/env python3
"""Bump the version in core + every workspace plugin (lockstep release model).

Updates:
  - pyproject.toml: project.version
  - plugins/*/pyproject.toml: project.version AND the tsugite-cli== pin in
    project.dependencies

Plain regex on single lines so formatting and comments are preserved.

Usage:
    uv run python scripts/bump_version.py 0.14.0
    uv run python scripts/bump_version.py 0.14.0 --dry-run
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT_PYPROJECT = REPO_ROOT / "pyproject.toml"
PLUGIN_GLOB = "plugins/*/pyproject.toml"

VERSION_LINE_RE = re.compile(r'^(version\s*=\s*)"[^"]+"', re.MULTILINE)
# Matches any inter-package pin like "tsugite-cli==X" / "tsugite-pty==X" so
# plugins that depend on sibling packages stay in lockstep.
SIBLING_DEP_RE = re.compile(r'"(tsugite-[a-z0-9-]+)==[^"]+"')
PEP440_RE = re.compile(r"^\d+\.\d+\.\d+([a-zA-Z0-9.+-]*)?$")


def _replace_first(text: str, pattern: re.Pattern, replacement: str, file: Path) -> str:
    new_text, count = pattern.subn(replacement, text, count=1)
    if count == 0:
        raise RuntimeError(f"No match for pattern in {file}: {pattern.pattern}")
    return new_text


def bump_root(new_version: str, dry_run: bool) -> None:
    text = ROOT_PYPROJECT.read_text()
    new_text = _replace_first(text, VERSION_LINE_RE, rf'\g<1>"{new_version}"', ROOT_PYPROJECT)
    _write(ROOT_PYPROJECT, text, new_text, dry_run, label="root version")


def bump_plugin(path: Path, new_version: str, dry_run: bool) -> None:
    text = path.read_text()
    new_text = _replace_first(text, VERSION_LINE_RE, rf'\g<1>"{new_version}"', path)
    if SIBLING_DEP_RE.search(new_text):
        new_text = SIBLING_DEP_RE.sub(rf'"\g<1>=={new_version}"', new_text)
    _write(path, text, new_text, dry_run, label=f"{path.parent.name} version + sibling pins")


def _write(path: Path, before: str, after: str, dry_run: bool, label: str) -> None:
    rel = path.relative_to(REPO_ROOT)
    if before == after:
        print(f"  unchanged: {rel} ({label})")
        return
    if dry_run:
        print(f"  would update: {rel} ({label})")
        return
    path.write_text(after)
    print(f"  updated: {rel} ({label})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version", help="New version (PEP 440 format, e.g. 0.14.0)")
    parser.add_argument("--dry-run", action="store_true", help="Print planned changes without writing")
    args = parser.parse_args()

    if not PEP440_RE.match(args.version):
        parser.error(f"version '{args.version}' does not look like PEP 440 (expected like 0.14.0 or 0.14.0a1)")

    print(f"Bumping to {args.version}{' (dry run)' if args.dry_run else ''}")
    bump_root(args.version, args.dry_run)
    plugins = sorted(REPO_ROOT.glob(PLUGIN_GLOB))
    if not plugins:
        print("  no plugins found under plugins/")
    for plugin in plugins:
        bump_plugin(plugin, args.version, args.dry_run)

    if not args.dry_run:
        print("\nNext steps:")
        print("  git diff                          # review changes")
        print(f"  git commit -am 'chore: bump version to {args.version}'")
        print(f"  git tag v{args.version}")
        print("  git push origin master --tags     # triggers PyPI + GitHub release")
    return 0


if __name__ == "__main__":
    sys.exit(main())
