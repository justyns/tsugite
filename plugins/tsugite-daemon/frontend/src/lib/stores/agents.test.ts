import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from '$lib/api/client';
import { AgentsStore, validateAgentId, type Agent } from './agents.svelte';

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

describe('AgentsStore.create and update', () => {
  it('posts the new agent, then refetches the list', async () => {
    const created = { id: 'atlas', name: 'Atlas' } as Agent;
    const post = vi.spyOn(api, 'post').mockResolvedValue(created as never);
    const get = vi.spyOn(api, 'get').mockResolvedValue({ agents: [atlas, created] } as never);
    const store = new AgentsStore();

    const result = await store.create({ id: 'atlas', name: 'Atlas', color: 'teal' });

    expect(post).toHaveBeenCalledWith('/api/agents', { id: 'atlas', name: 'Atlas', color: 'teal' });
    expect(result).toBe(created);
    expect(get).toHaveBeenCalledTimes(1);
    expect(store.list).toEqual([atlas, created]);
  });

  it('patches an agent by encoded id, then refetches the list', async () => {
    const patch = vi.spyOn(api, 'patch').mockResolvedValue(homelab as never);
    vi.spyOn(api, 'get').mockResolvedValue({ agents: [homelab] } as never);
    const store = new AgentsStore();

    await store.update('a/b', { model: null });

    expect(patch).toHaveBeenCalledWith('/api/agents/a%2Fb', { model: null });
    expect(store.list).toEqual([homelab]);
  });

  it('rejects with the server error and does not change the list', async () => {
    vi.spyOn(api, 'post').mockRejectedValue(new Error('Agent already exists: atlas'));
    const get = vi.spyOn(api, 'get');
    const store = new AgentsStore();

    await expect(store.create({ id: 'atlas' })).rejects.toThrow('already exists');
    expect(get).not.toHaveBeenCalled();
  });

  it('refetches after a load that started before the change', async () => {
    let settle!: (v: unknown) => void;
    const get = vi
      .spyOn(api, 'get')
      .mockImplementationOnce(() => new Promise((r) => (settle = r)))
      .mockResolvedValue({ agents: [atlas, homelab] } as never);
    vi.spyOn(api, 'patch').mockResolvedValue(homelab as never);
    const store = new AgentsStore();
    const stale = store.load();

    const update = store.update('ops', { name: 'homelab' });
    settle({ agents: [atlas] });
    await Promise.all([stale, update]);

    expect(get).toHaveBeenCalledTimes(2);
    expect(store.list).toEqual([atlas, homelab]);
  });
});

describe('validateAgentId', () => {
  it.each(['atlas', 'a', '0day', 'ops-2_b', 'a'.repeat(64)])('accepts %s', (id) => {
    expect(validateAgentId(id)).toBeNull();
  });

  it.each(['', 'Atlas', '-atlas', '_atlas', 'a b', '../x', 'a/b', 'a'.repeat(65)])(
    'rejects %j',
    (id) => {
      expect(validateAgentId(id)).not.toBeNull();
    },
  );
});
