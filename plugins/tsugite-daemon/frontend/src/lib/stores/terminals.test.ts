import { afterEach, describe, expect, test, vi } from 'vitest';
import { api } from '$lib/api/client';
import { TerminalsStore, type Terminal, type TerminalState } from './terminals.svelte';

function term(id: string, state: TerminalState = 'running'): Terminal {
  return { id, state } as unknown as Terminal;
}

describe('TerminalsStore.applyTerminalState', () => {
  test('records the live state and patches the matching list row', () => {
    const store = new TerminalsStore();
    store.list = [term('t1', 'running')];
    store.applyTerminalState({ terminal_id: 't1', state: 'succeeded' });
    expect(store.stateOf('t1')).toBe('succeeded');
    expect(store.list[0]!.state).toBe('succeeded');
  });

  test('stateOf falls back to the record state when no overlay exists', () => {
    const store = new TerminalsStore();
    store.list = [term('t1', 'starting')];
    expect(store.stateOf('t1')).toBe('starting');
    expect(store.stateOf('missing')).toBeNull();
  });

  test('isLive is true only for starting/running', () => {
    const store = new TerminalsStore();
    store.list = [term('t1', 'running'), term('t2', 'failed')];
    expect(store.isLive('t1')).toBe(true);
    expect(store.isLive('t2')).toBe(false);
  });

  test('ignores a malformed state event', () => {
    const store = new TerminalsStore();
    store.applyTerminalState({ terminal_id: 't1' });
    expect(store.stateOf('t1')).toBeNull();
  });
});

describe('TerminalsStore follow axis (client-only)', () => {
  test('follow defaults to true and toggles independently of server state', () => {
    const store = new TerminalsStore();
    expect(store.isFollowing('t1')).toBe(true);
    store.setFollow('t1', false);
    expect(store.isFollowing('t1')).toBe(false);
  });

  test('re-enabling follow clears any queued-line count', () => {
    const store = new TerminalsStore();
    store.queuedLines = { t1: 5 };
    store.setFollow('t1', true);
    expect(store.queuedLines.t1).toBe(0);
  });
});

describe('TerminalsStore.load restarted_from carry-forward', () => {
  afterEach(() => vi.restoreAllMocks());

  test('preserves a locally-restarted terminal restarted_from across a poll refresh that omits it', async () => {
    const store = new TerminalsStore();
    // restart() left the new terminal carrying the chip source; the metrics
    // poll's GET /api/terminals/ never serves restarted_from back.
    store.list = [{ ...term('new-1'), restarted_from: 'old-0' }];
    vi.spyOn(api, 'get').mockResolvedValue({ terminals: [term('new-1')] });

    await store.load();

    expect(store.list.find((t) => t.id === 'new-1')?.restarted_from).toBe('old-0');
  });

  test('never invents restarted_from for a terminal that never had one', async () => {
    const store = new TerminalsStore();
    store.list = [term('t1')];
    vi.spyOn(api, 'get').mockResolvedValue({ terminals: [term('t1')] });

    await store.load();

    expect(store.list[0]!.restarted_from).toBeUndefined();
  });

  test('a value present on the fetched record wins over the carried one', async () => {
    const store = new TerminalsStore();
    store.list = [{ ...term('t1'), restarted_from: 'stale' }];
    vi.spyOn(api, 'get').mockResolvedValue({
      terminals: [{ ...term('t1'), restarted_from: 'fresh' }],
    });

    await store.load();

    expect(store.list[0]!.restarted_from).toBe('fresh');
  });
});
