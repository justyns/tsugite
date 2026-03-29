#!/usr/bin/env python3
"""Backfill usage.db from existing JSONL history files.

Usage:
    uv run python scripts/backfill_usage.py
    uv run python scripts/backfill_usage.py --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tsugite.history import SessionStorage, get_history_dir, list_session_files
from tsugite.history.models import Turn
from tsugite.usage.store import UsageStore


def backfill(dry_run: bool = False) -> None:
    store = UsageStore()
    history_dir = get_history_dir()

    if not history_dir.exists():
        print(f"No history directory found at {history_dir}")
        return

    files = list_session_files()
    print(f"Found {len(files)} history files in {history_dir}")

    existing = {r["session_id"] for r in store.query(limit=100_000) if r.get("session_id")}
    print(f"Already have {len(existing)} sessions in usage.db")

    added = 0
    skipped = 0

    for session_path in files:
        session_id = session_path.stem

        if session_id in existing:
            skipped += 1
            continue

        # Skip obvious test fixtures
        if session_id.startswith("conv-") or session_id.startswith("test-"):
            skipped += 1
            continue

        try:
            storage = SessionStorage(session_path)
            meta = storage.load_meta_fast(session_path)
            records = storage.load_records()
        except Exception as e:
            print(f"  SKIP {session_id}: {e}")
            skipped += 1
            continue

        turns = [r for r in records if isinstance(r, Turn)]
        if not turns:
            skipped += 1
            continue

        total_tokens = sum(t.tokens or 0 for t in turns)
        total_cost = sum(t.cost or 0 for t in turns)
        total_duration = sum(t.duration_ms or 0 for t in turns)
        agent = meta.agent if meta else None
        model = meta.model if meta else None

        # Determine source from metadata
        source = "cli"
        for t in turns:
            if t.metadata and t.metadata.get("source"):
                source = t.metadata["source"]
                break

        if dry_run:
            print(f"  WOULD ADD {session_id}: agent={agent} model={model} tokens={total_tokens} cost=${total_cost:.4f}")
        else:
            store.record(
                session_id=session_id,
                agent=agent,
                model=model,
                source=source,
                total_tokens=total_tokens,
                cost_usd=total_cost if total_cost > 0 else None,
                duration_ms=total_duration if total_duration > 0 else None,
                timestamp=turns[0].timestamp.isoformat() if turns[0].timestamp else None,
            )
        added += 1

    print(f"\nDone: {added} sessions {'would be ' if dry_run else ''}added, {skipped} skipped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill usage.db from JSONL history")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be added without writing")
    args = parser.parse_args()
    backfill(dry_run=args.dry_run)
