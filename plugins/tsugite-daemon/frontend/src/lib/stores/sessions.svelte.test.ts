import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from '$lib/api/client';
import { SessionsStore, type SessionRow } from './sessions.svelte';

afterEach(() => vi.restoreAllMocks());

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

// The shell seeds the session list at boot while the chats rail loads it too;
// both effects run in one flush, so the store coalesces that burst.
describe('SessionsStore.load', () => {
  it('shares one GET between callers that ask in the same tick', async () => {
    const get = vi.spyOn(api, 'get').mockResolvedValue({ sessions: [row('a')] } as never);
    const store = new SessionsStore();

    await Promise.all([store.load('odyn'), store.load('odyn')]);

    expect(get).toHaveBeenCalledTimes(1);
    expect(store.rows.map((r) => r.id)).toEqual(['a']);
  });

  it('refetches for a reload that follows a mutation, not the boot burst', async () => {
    const get = vi.spyOn(api, 'get').mockResolvedValue({ sessions: [] } as never);
    const store = new SessionsStore();

    await store.load('odyn');
    await store.load('odyn');

    expect(get).toHaveBeenCalledTimes(2);
  });

  it('leaves a different agent to its own GET', async () => {
    const get = vi.spyOn(api, 'get').mockResolvedValue({ sessions: [] } as never);
    const store = new SessionsStore();

    await Promise.all([store.load('odyn'), store.load('scout')]);

    expect(get).toHaveBeenCalledTimes(2);
  });
});
