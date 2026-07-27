/**
 * Build the composer's built-in `@` session source: the recent chat rows mapped
 * to RefItems the popover lists alongside workspace files. Pure and node-testable
 * (like sessionModel.ts); ChatComposer feeds it the store's ordered rows.
 *
 * A pick attaches the session as a context chip through the `session` capture
 * provider - the RefItem's kind IS `session`, matching that provider's key, so
 * the shared attach path captures it with no session-specific pick code.
 */
import type { SessionRow } from '$lib/stores/sessions.svelte';
import type { RefItem } from '$lib/components/composer/types';
import { formatWhen } from './sessionModel';

/** Cap the source at the most-recent chats so the popover stays scannable; the
 *  store already orders rows (pinned then recency), so this slices the head. */
const MAX_SESSION_REFS = 25;

/** Map ordered session rows to `@` source items, dropping the current chat (you
 *  don't reference the chat you're in) and capping to the most recent. */
export function buildSessionRefs(
  rows: SessionRow[],
  currentId: string | null,
  now: number = Date.now(),
): RefItem[] {
  const items: RefItem[] = [];
  for (const row of rows) {
    if (row.id === currentId) continue;
    const when = formatWhen(row.last_active, now);
    const detail = [row.status, when].filter(Boolean).join(' · ');
    items.push({
      id: row.id,
      kind: 'session',
      label: row.title || 'Untitled chat',
      ...(detail ? { detail } : {}),
      group: 'Sessions',
    });
    if (items.length >= MAX_SESSION_REFS) break;
  }
  return items;
}
