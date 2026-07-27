import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

// The store's mutations + revalidate hit the api client; mock it so the SSE
// application paths are exercised without a network (and an accidental
// revalidate is a harmless no-op).
vi.mock('$lib/api/client', () => ({
  api: {
    get: vi.fn(async () => ({ sessions: [] })),
    post: vi.fn(async () => ({})),
    patch: vi.fn(async () => ({})),
    put: vi.fn(async () => ({})),
    del: vi.fn(async () => ({})),
  },
  authHeaders: () => ({}),
}));

import { api } from '$lib/api/client';
import { SessionsStore, type SessionRow } from './sessions.svelte';

const apiGet = api.get as ReturnType<typeof vi.fn>;

function row(id: string, extra: Partial<SessionRow> = {}): SessionRow {
  return { id, pinned: false, pin_position: null, ...extra } as unknown as SessionRow;
}

describe('SessionsStore SSE application', () => {
  test('applySessionEvent folds progress ticks into the per-session overlay', () => {
    const store = new SessionsStore();
    store.rows = [row('s1')];
    store.applySessionEvent({ session_id: 's1', event_type: 'turn_start', turn: 2 });
    store.applySessionEvent({ session_id: 's1', event_type: 'tool_result', tool: 'grep' });
    const p = store.progressFor('s1');
    expect(p?.turn_count).toBe(2);
    expect(p?.tool_count).toBe(1);
    expect(p?.status_text).toBe('Tool: grep');
  });

  test('applySessionEvent ignores an event with no session id', () => {
    const store = new SessionsStore();
    store.applySessionEvent({ event_type: 'turn_start', turn: 1 });
    expect(Object.keys(store.progress)).toHaveLength(0);
  });

  test('busy / titled updates patch the matching row in place', () => {
    const store = new SessionsStore();
    store.rows = [row('s1', { busy: false, title: 'old' })];
    store.applySessionUpdate({ action: 'busy', id: 's1', busy: true });
    store.applySessionUpdate({ action: 'titled', id: 's1', title: 'new' });
    expect(store.rows[0]!.busy).toBe(true);
    expect(store.rows[0]!.title).toBe('new');
  });

  test('reordered echoes the new pin order', () => {
    const store = new SessionsStore();
    store.rows = [
      row('a', { pinned: true, pin_position: 0 }),
      row('b', { pinned: true, pin_position: 1 }),
    ];
    store.applySessionUpdate({ action: 'reordered', ids: ['b', 'a'] });
    expect(store.ordered.map((r) => r.id)).toEqual(['b', 'a']);
  });

  describe('terminal lifecycle actions', () => {
    beforeEach(() => vi.useFakeTimers());
    afterEach(() => vi.useRealTimers());

    test('completed drops the progress overlay and patches the row status', () => {
      const store = new SessionsStore();
      store.agent = 'smoke';
      store.rows = [row('s1', { status: 'running', state: 'running' })];
      store.progress = {
        s1: { turn_count: 3, tool_count: 1, status_text: 'x', last_event_time: null },
      };
      store.applySessionUpdate({ action: 'completed', id: 's1' });
      expect(store.progress.s1).toBeUndefined();
      expect(store.rows[0]!.status).toBe('completed');
    });
  });
});

describe('SessionsStore ordering', () => {
  test('ordered floats pinned rows above the recency tail', () => {
    const store = new SessionsStore();
    store.rows = [row('a'), row('b', { pinned: true, pin_position: 0 })];
    expect(store.ordered.map((r) => r.id)).toEqual(['b', 'a']);
    expect(store.pinned.map((r) => r.id)).toEqual(['b']);
    expect(store.unpinned.map((r) => r.id)).toEqual(['a']);
  });
});

describe('SessionsStore conversation broadcast bridge', () => {
  test('bindConversation forwards matching session_event frames to the sink', () => {
    const store = new SessionsStore();
    store.rows = [row('s1')];
    const seen: Record<string, unknown>[] = [];
    store.bindConversation('s1', (d) => seen.push(d));

    store.applySessionEvent({ session_id: 's1', event_type: 'thought', content: 'hi' });
    store.applySessionEvent({ session_id: 'other', event_type: 'thought', content: 'no' });

    expect(seen).toHaveLength(1);
    expect(seen[0]).toMatchObject({ session_id: 's1', event_type: 'thought' });
  });

  test('the unbind returned by bindConversation stops delivery', () => {
    const store = new SessionsStore();
    const seen: unknown[] = [];
    const off = store.bindConversation('s1', (d) => seen.push(d));
    off();
    store.applySessionEvent({ session_id: 's1', event_type: 'thought' });
    expect(seen).toHaveLength(0);
  });

  test('two surfaces on the same session both receive the broadcast', () => {
    const store = new SessionsStore();
    let a = 0;
    let b = 0;
    store.bindConversation('s1', () => (a += 1));
    store.bindConversation('s1', () => (b += 1));
    store.applySessionEvent({ session_id: 's1', event_type: 'code' });
    expect([a, b]).toEqual([1, 1]);
  });
});

describe('SessionsStore getInfo context limit', () => {
  test('a fresh session (raw context_limit null) reports the resolved agent-default limit', async () => {
    // The meter's fallback is built from getInfo().contextLimit; a fresh session's
    // raw field is null, so getInfo must prefer the resolved limit or the new
    // session shows no context meter at all until its first turn.
    const store = new SessionsStore();
    apiGet.mockResolvedValueOnce({
      agent: 'smoke',
      metadata: {},
      context_limit: null,
      context_limit_resolved: 200_000,
      cumulative_tokens: 0,
    });
    const info = await store.getInfo('s1');
    expect(info.contextLimit).toBe(200_000);
    expect(info.cumulativeTokens).toBe(0);
  });

  test('a per-session window (context_limit set) is honored via the resolved field', async () => {
    const store = new SessionsStore();
    apiGet.mockResolvedValueOnce({
      context_limit: 1_000_000,
      context_limit_resolved: 1_000_000,
      cumulative_tokens: 12,
    });
    const info = await store.getInfo('s1');
    expect(info.contextLimit).toBe(1_000_000);
  });
});

describe('SessionsStore settings broadcast', () => {
  test('a settings session_update bumps settingsRev so the pickers refetch', () => {
    const store = new SessionsStore();
    store.rows = [row('s1', { model: null } as Partial<SessionRow>)];
    expect(store.settingsRev['s1'] ?? 0).toBe(0);
    store.applySessionUpdate({
      action: 'settings',
      id: 's1',
      model: 'anthropic:claude-opus-4-8',
      reasoning_effort: 'high',
    });
    expect(store.settingsRev['s1']).toBe(1);
    store.applySessionUpdate({ action: 'settings', id: 's1', model: null, reasoning_effort: null });
    expect(store.settingsRev['s1']).toBe(2);
  });
});
