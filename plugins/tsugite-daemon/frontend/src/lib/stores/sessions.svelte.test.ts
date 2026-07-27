import { describe, expect, it } from 'vitest';
import { SessionsStore, type SessionRow } from './sessions.svelte';

function row(id: string, extra: Partial<SessionRow> = {}): SessionRow {
  return {
    id,
    user_id: 'u',
    label: id,
    source: 'web',
    status: 'active',
    state: 'idle',
    created_at: '2026-07-16T00:00:00Z',
    last_active: null,
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
    ...extra,
  };
}

// Unread lifecycle, frontend half: a session flags unread from server-sent
// row.unread on background activity (load()), and clears when viewed. The clear
// arrives two ways - the cross-tab `viewed` broadcast and this tab's own
// markViewed echo - both funnelling through patchRow(unread:false). Pin the
// broadcast reducer here (pure, no network); the local echo shares the path.
describe('SessionsStore unread lifecycle', () => {
  it('clears unread on the viewed row when a `viewed` broadcast arrives', () => {
    const store = new SessionsStore();
    store.rows = [row('a', { unread: true }), row('b', { unread: true })];

    store.applySessionUpdate({ action: 'viewed', id: 'a' });

    expect(store.rows.find((r) => r.id === 'a')?.unread).toBe(false);
    expect(store.rows.find((r) => r.id === 'b')?.unread).toBe(true); // only the viewed row clears
  });

  it('leaves unread untouched for a non-unread action (e.g. busy)', () => {
    const store = new SessionsStore();
    store.rows = [row('a', { unread: true })];

    store.applySessionUpdate({ action: 'busy', id: 'a', busy: true });

    const r = store.rows.find((x) => x.id === 'a');
    expect(r?.unread).toBe(true); // background activity marker survives a busy tick
    expect(r?.busy).toBe(true);
  });

  it('ignores a viewed broadcast for an id it does not hold', () => {
    const store = new SessionsStore();
    store.rows = [row('a', { unread: true })];

    store.applySessionUpdate({ action: 'viewed', id: 'ghost' });

    expect(store.rows.find((r) => r.id === 'a')?.unread).toBe(true);
  });
});
