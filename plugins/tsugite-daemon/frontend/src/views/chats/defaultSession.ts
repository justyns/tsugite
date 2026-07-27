/**
 * Shared resolver for "which session a chat surface shows when it has no explicit
 * sessionId". The chat surface uses it to seed its selection; the sessions rail
 * uses the same rule to highlight the row the focused paramless surface is
 * showing, so the two never disagree.
 */
import type { SessionRow } from '$lib/stores/sessions.svelte';
import { isFinishedSession } from './sessionModel';

/** Priority: an explicit/deep-linked id (if present), then the primary session,
 *  then the first pinned, then the newest row. Null when the list is empty.
 *  Superseded sessions (compaction "view source") and finished ones are valid
 *  explicit targets but never the bare default - a live session is; when none is
 *  live the view opens empty rather than auto-selecting an ended chat. */
export function resolveDefaultSession(rows: SessionRow[], preferId?: string | null): string | null {
  if (rows.length === 0) return null;
  if (preferId && rows.some((r) => r.id === preferId)) return preferId;
  const live = rows.filter((r) => !r.superseded_by && !isFinishedSession(r));
  return (
    live.find((r) => r.is_primary)?.id ?? live.find((r) => r.pinned)?.id ?? live[0]?.id ?? null
  );
}
