import { buildHash, type RouteParams } from '$lib/router.svelte';
import type { SessionRow } from '$lib/stores/sessions.svelte';
import { liveSessionId } from '../../views/chats/sessionModel';

export const SOLO_VIEW = 'solo';

/** Outcome of fetching a session id the loaded list does not hold. */
export type SoloProbe =
  'pending' | 'found' | 'missing' | 'error' | { kind: 'superseded'; successorId: string };

export type SoloResolution =
  | { state: 'none' }
  | { state: 'loading'; sessionId?: string }
  | { state: 'ready'; sessionId: string }
  | { state: 'superseded'; sessionId: string; successorId: string }
  | { state: 'not-found'; sessionId: string }
  | { state: 'error'; sessionId: string }
  | { state: 'alias-empty'; alias: string };

/** `sessionId` pins one conversation. `alias` follows whichever live session
 *  holds it, and is ignored when `sessionId` is set. */
export function resolveSolo(
  params: RouteParams,
  rows: SessionRow[],
  probe: SoloProbe,
  loaded: boolean,
): SoloResolution {
  const sessionId = params.sessionId;
  if (sessionId) {
    const row = rows.find((r) => r.id === sessionId);
    if (row?.superseded_by) {
      return { state: 'superseded', sessionId, successorId: liveSessionId(rows, sessionId) };
    }
    if (row || probe === 'found') return { state: 'ready', sessionId };
    if (typeof probe === 'object') {
      return { state: 'superseded', sessionId, successorId: probe.successorId };
    }
    if (probe === 'missing') return { state: 'not-found', sessionId };
    if (probe === 'error') return { state: 'error', sessionId };
    return { state: 'loading', sessionId };
  }
  const alias = params.alias;
  if (!alias) return { state: 'none' };
  const holder = rows.find((r) => r.alias === alias && !r.superseded_by);
  if (holder) return { state: 'ready', sessionId: holder.id };
  return loaded ? { state: 'alias-empty', alias } : { state: 'loading' };
}

/** The hash that opens `row` in a solo window, by alias when it holds one. */
export function soloHash(row: Pick<SessionRow, 'id' | 'alias'>): string {
  return row.alias
    ? buildHash(SOLO_VIEW, { alias: row.alias })
    : buildHash(SOLO_VIEW, { sessionId: row.id });
}
