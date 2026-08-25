#!/usr/bin/env python3
"""One-off migration: collapse a daemon.yaml `agents:` block into flat defaults.

The daemon no longer has a per-agent config layer. This hoists the single agent's
settings to the `default_*` keys and strips the Discord bots' `agent:` binding.

Edits the file line by line rather than re-dumping it, so comments and formatting
outside the `agents:` block survive.

Usage:
    python scripts/migrate_daemon_agents.py <path-to-daemon.yaml> [--dry-run]
"""

import argparse
import difflib
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

from tsugite.utils import atomic_write_text

# agents[<name>].<key>  ->  top-level key
HOISTED = {
    "workspace_dir": "default_workspace_dir",
    "agent_file": "default_agent_file",
    "model": "default_model",
    "compaction_model": "default_compaction_model",
    "context_limit": "default_context_limit",
    "max_turns": "default_max_turns",
}


def _validate(data: dict) -> dict:
    agents = data.get("agents")
    if not agents:
        raise SystemExit("No 'agents:' block found; nothing to migrate.")
    if len(agents) > 1:
        names = ", ".join(sorted(agents))
        raise SystemExit(
            f"Found {len(agents)} agents ({names}). The daemon now runs exactly one.\n"
            "Delete the ones you don't want (or split them across separate daemons), then re-run."
        )
    name, agent = next(iter(agents.items()))
    known = set(HOISTED) | {"timezone", "auto_compact", "sandbox"}
    unknown = sorted(set(agent) - known)
    if unknown:
        raise SystemExit(f"Agent {name!r} has keys this script cannot hoist: {', '.join(unknown)}")
    if "sandbox" in agent and "sandbox" in data and data["sandbox"] != agent["sandbox"]:
        raise SystemExit(
            f"Agent {name!r} has a sandbox block that differs from the global one. Merge them by hand, then re-run."
        )
    return agent


def _block_bounds(lines: list[str], start: int) -> int:
    """Index one past the end of the indented block that follows lines[start]."""
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if line.strip() and not line.startswith((" ", "\t")):
            break
        end += 1
    # Don't swallow trailing blank lines; they belong to whatever comes next.
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    return end


def migrate_text(text: str) -> str:
    data = yaml.safe_load(text)
    agent = _validate(data)
    lines = text.splitlines(keepends=True)

    start = next((i for i, ln in enumerate(lines) if re.match(r"^agents:\s*(#.*)?$", ln)), None)
    if start is None:
        raise SystemExit("Could not find a top-level 'agents:' line to rewrite.")
    end = _block_bounds(lines, start)

    body = lines[start + 1 : end]
    # First non-blank line is the agent name; its keys sit one level deeper.
    name_idx = next(i for i, ln in enumerate(body) if ln.strip())
    key_indent = len(body[name_idx + 1]) - len(body[name_idx + 1].lstrip())

    # A global sandbox already says everything an identical agent-level one would,
    # and dedenting the agent's copy would emit `sandbox:` twice at top level.
    drop_agent_sandbox = "sandbox" in data and "sandbox" in agent

    hoisted: list[str] = []
    skipping = False
    for line in body[name_idx + 1 :]:
        if not line.strip():
            if not skipping:
                hoisted.append(line)
            continue
        indent = len(line) - len(line.lstrip())
        if indent > key_indent and skipping:
            continue
        skipping = False
        dedented = line[key_indent:] if indent >= key_indent else line.lstrip()
        if indent == key_indent:
            key = dedented.split(":", 1)[0].strip()
            if key == "sandbox" and drop_agent_sandbox:
                skipping = True
                continue
            if key in HOISTED:
                dedented = dedented.replace(key, HOISTED[key], 1)
        hoisted.append(dedented)

    out = lines[:start] + hoisted + lines[end:]
    return "".join(_strip_bot_agent_bindings(out))


def _strip_bot_agent_bindings(lines: list[str]) -> list[str]:
    """Drop each Discord bot's `agent:` binding, and only those.

    Scoped to the `discord_bots:` block: `plugins:` is free-form, so a bare
    indented `agent:` elsewhere in the file belongs to somebody else.
    """
    start = next((i for i, ln in enumerate(lines) if re.match(r"^discord_bots:\s*(#.*)?$", ln)), None)
    if start is None:
        return lines
    # Run to the next top-level key. `_block_bounds` stops short here because a
    # YAML sequence item (`- name: ...`) sits at column 0 like a key does.
    end = start + 1
    while end < len(lines) and not re.match(r"^[A-Za-z_]", lines[end]):
        end += 1
    block = [ln for ln in lines[start:end] if not re.match(r"^\s*-?\s*agent:\s*\S+\s*$", ln)]
    return lines[:start] + block + lines[end:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path, help="path to daemon.yaml")
    ap.add_argument("--dry-run", action="store_true", help="print the diff without writing")
    args = ap.parse_args()

    if not args.config.exists():
        raise SystemExit(f"Not found: {args.config}")

    original = args.config.read_text(encoding="utf-8")
    migrated = migrate_text(original)

    # Re-parse so a broken transform fails here rather than at daemon start.
    reparsed = yaml.safe_load(migrated)
    if "agents" in reparsed:
        raise SystemExit("Migration left an 'agents:' key behind; aborting.")
    if "default_workspace_dir" not in reparsed:
        raise SystemExit("Migration produced no 'default_workspace_dir'; aborting.")
    # PyYAML keeps the last of a duplicated key without complaining, so a
    # doubled top-level key survives the parse above; count the text instead.
    top_level = [ln.split(":", 1)[0] for ln in migrated.splitlines() if re.match(r"^[A-Za-z_]", ln)]
    dupes = sorted({k for k in top_level if top_level.count(k) > 1})
    if dupes:
        raise SystemExit(f"Migration produced duplicate top-level keys: {', '.join(dupes)}; aborting.")

    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            migrated.splitlines(keepends=True),
            fromfile=str(args.config),
            tofile=f"{args.config} (migrated)",
        )
    )
    sys.stdout.write(diff or "(no changes)\n")

    if args.dry_run:
        return

    backup = args.config.with_suffix(f".yaml.bak-{datetime.now().strftime('%Y%m%d%H%M%S')}")
    shutil.copy2(args.config, backup)
    atomic_write_text(args.config, migrated)
    print(f"\nWrote {args.config} (backup: {backup})")


if __name__ == "__main__":
    main()
