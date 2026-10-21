import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from '$lib/api/client';
import { AgentsStore, type Agent } from './agents.svelte';

afterEach(() => vi.restoreAllMocks());

const atlas = { id: 'main', name: 'atlas', is_default: true } as Agent;
const homelab = { id: 'ops', name: 'homelab', is_default: false } as Agent;

describe('AgentsStore.load', () => {
  it('shares one GET between callers that ask while it is in flight', async () => {
    const get = vi.spyOn(api, 'get').mockResolvedValue({ agents: [atlas] } as never);
    const store = new AgentsStore();

    await Promise.all([store.load(), store.load()]);

    expect(get).toHaveBeenCalledTimes(1);
    expect(store.list).toEqual([atlas]);
  });

  it('records the error and keeps the list when the fetch fails', async () => {
    const get = vi.spyOn(api, 'get').mockResolvedValue({ agents: [atlas] } as never);
    const store = new AgentsStore();
    await store.load();
    get.mockRejectedValue(new Error('daemon down'));

    await store.load();

    expect(store.error).toBe('daemon down');
    expect(store.list).toEqual([atlas]);
  });
});

describe('AgentsStore.resolve', () => {
  it('refetches once for an agent id the list does not hold', async () => {
    const get = vi.spyOn(api, 'get').mockResolvedValue({ agents: [atlas] } as never);
    const store = new AgentsStore();
    await store.load();
    get.mockResolvedValue({ agents: [atlas, homelab] } as never);

    await store.resolve(['main', 'ops']);

    expect(get).toHaveBeenCalledTimes(2);
    expect(store.list).toEqual([atlas, homelab]);
  });

  it('does not refetch again for an id that stayed unknown', async () => {
    const get = vi.spyOn(api, 'get').mockResolvedValue({ agents: [atlas] } as never);
    const store = new AgentsStore();
    await store.load();

    await store.resolve(['gone']);
    await store.resolve(['gone']);

    expect(get).toHaveBeenCalledTimes(2);
  });

  it('fetches again rather than joining a load that started before the agent existed', async () => {
    let settle!: (v: unknown) => void;
    const get = vi
      .spyOn(api, 'get')
      .mockReturnValueOnce(new Promise((r) => (settle = r)) as never)
      .mockResolvedValue({ agents: [atlas, homelab] } as never);
    const store = new AgentsStore();
    const first = store.load();

    const resolving = store.resolve(['ops']);
    settle({ agents: [atlas] });
    await Promise.all([first, resolving]);

    expect(get).toHaveBeenCalledTimes(2);
    expect(store.list).toEqual([atlas, homelab]);
  });

  it('retries an unknown id after a failed fetch', async () => {
    const get = vi.spyOn(api, 'get').mockRejectedValueOnce(new Error('daemon down'));
    const store = new AgentsStore();
    await store.resolve(['ops']);
    get.mockResolvedValue({ agents: [atlas, homelab] } as never);

    await store.resolve(['ops']);

    expect(get).toHaveBeenCalledTimes(2);
    expect(store.list).toEqual([atlas, homelab]);
  });

  it('skips the fetch when every id is known', async () => {
    const get = vi.spyOn(api, 'get').mockResolvedValue({ agents: [atlas] } as never);
    const store = new AgentsStore();
    await store.load();

    await store.resolve(['main']);

    expect(get).toHaveBeenCalledTimes(1);
  });
});
