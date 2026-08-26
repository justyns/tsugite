/** A backend session row with every field filled in, for the chats rail tests. */
import type { AttentionRecord, SessionRow } from '$lib/stores/sessions.svelte';

export function attentionRecord(
  source: string,
  extra: Partial<AttentionRecord> = {},
): AttentionRecord {
  return {
    id: `attn-${source}`,
    owner_kind: 'session',
    owner_id: 's1',
    source,
    ref_id: `${source}-1`,
    kind: 'needs_answer',
    created_at: '2026-07-17T00:00:00Z',
    ...extra,
  };
}

export function sessionRow(id: string, extra: Partial<SessionRow> = {}): SessionRow {
  return {
    id,
    user_id: 'u',
    label: id,
    source: 'web',
    status: 'active',
    state: 'idle',
    created_at: '2026-07-17T00:00:00Z',
    last_active: '2026-07-17T00:00:00Z',
    parent_id: null,
    prompt: '',
    model: null,
    error: null,
    result: null,
    title: id,
    is_default: false,
    metadata: {},
    pinned: false,
    pin_position: null,
    last_viewed_at: null,
    superseded_by: null,
    unread: false,
    is_primary: false,
    busy: false,
    attention: [],
    ...extra,
  };
}
