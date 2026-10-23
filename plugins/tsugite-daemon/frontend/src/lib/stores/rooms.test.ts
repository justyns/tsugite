import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from '$lib/api/client';
import { RoomsStore, type Room } from './rooms.svelte';

afterEach(() => {
  vi.restoreAllMocks();
});

const room = (id: string, extra: Partial<Room> = {}): Room => ({
  id,
  title: `room ${id}`,
  participants: [],
  max_rounds: 12,
  rounds_used: 0,
  job_id: null,
  needs_attention: false,
  attention: [],
  created_at: '2026-10-22T00:00:00Z',
  status: 'active',
  ...extra,
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (err: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe('RoomsStore.load', () => {
  it('shares one GET between callers that ask while it is in flight', async () => {
    const get = vi.spyOn(api, 'get').mockResolvedValue({ rooms: [room('r1')] } as never);
    const store = new RoomsStore();

    await Promise.all([store.load(), store.load()]);

    expect(get).toHaveBeenCalledTimes(1);
    expect(store.list.map((r) => r.id)).toEqual(['r1']);
    expect(store.loaded).toBe(true);
  });

  it('rejects, records the error and keeps the list when the fetch fails', async () => {
    const get = vi.spyOn(api, 'get').mockResolvedValue({ rooms: [room('r1')] } as never);
    const store = new RoomsStore();
    await store.load();
    get.mockRejectedValue(new Error('daemon down'));

    await expect(store.load()).rejects.toThrow('daemon down');

    expect(store.error).toBe('daemon down');
    expect(store.list.map((r) => r.id)).toEqual(['r1']);
  });
});

describe('RoomsStore.surfaceRef', () => {
  it('returns a room surface for a room id and null for any other session', async () => {
    vi.spyOn(api, 'get').mockResolvedValue({ rooms: [room('r1')] } as never);
    const store = new RoomsStore();

    expect(await store.surfaceRef('r1')).toEqual({
      kind: 'room',
      params: { roomId: 'r1' },
      title: 'room r1',
    });
    expect(await store.surfaceRef('s1')).toBeNull();
  });

  it('waits for the load already in flight before deciding', async () => {
    const pending = deferred<unknown>();
    const get = vi.spyOn(api, 'get').mockReturnValue(pending.promise as never);
    const store = new RoomsStore();
    const first = store.load();

    const ref = store.surfaceRef('r1');
    pending.resolve({ rooms: [room('r1')] });

    await first;
    expect((await ref)?.kind).toBe('room');
    expect(get).toHaveBeenCalledTimes(1);
  });

  it('rejects when the list cannot be loaded', async () => {
    vi.spyOn(api, 'get').mockRejectedValue(new Error('daemon down'));
    const store = new RoomsStore();

    await expect(store.surfaceRef('r1')).rejects.toThrow('daemon down');
  });
});

describe('RoomsStore.applySessionUpdate', () => {
  async function loadedStore() {
    const get = vi.spyOn(api, 'get').mockResolvedValue({ rooms: [room('r1')] } as never);
    const store = new RoomsStore();
    await store.load();
    get.mockClear();
    return { store, get };
  }

  it('replaces a known room from room_updated without refetching', async () => {
    const { store, get } = await loadedStore();

    store.applySessionUpdate({
      action: 'room_updated',
      id: 'r1',
      room: room('r1', { rounds_used: 5 }),
    });

    expect(store.byId('r1')?.rounds_used).toBe(5);
    expect(get).not.toHaveBeenCalled();
  });

  it('adds a room it did not know', async () => {
    const { store } = await loadedStore();

    store.applySessionUpdate({ action: 'room_updated', id: 'r2', room: room('r2') });

    expect(store.list.map((r) => r.id)).toEqual(['r1', 'r2']);
  });

  it('ignores other session updates', async () => {
    const { store, get } = await loadedStore();

    store.applySessionUpdate({ action: 'completed', id: 'r1' });

    expect(store.byId('r1')?.status).toBe('active');
    expect(get).not.toHaveBeenCalled();
  });
});

describe('RoomsStore mutations', () => {
  it('replaces the changed room and keeps the others', async () => {
    vi.spyOn(api, 'get').mockResolvedValue({ rooms: [room('r1'), room('r2')] } as never);
    const store = new RoomsStore();
    await store.load();
    vi.spyOn(api, 'patch').mockResolvedValue(room('r2', { max_rounds: 24 }) as never);

    await store.setMaxRounds('r2', 24);

    expect(store.list.map((r) => [r.id, r.max_rounds])).toEqual([
      ['r1', 12],
      ['r2', 24],
    ]);
  });
});
