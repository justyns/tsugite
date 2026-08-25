import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from '$lib/api/client';
import { AgentsMetaStore } from './agentsMeta.svelte';

afterEach(() => vi.restoreAllMocks());

// The shell seeds the roster at boot while the chats rail asks for it too, so
// both fire in one effect flush.
describe('AgentsMetaStore.load', () => {
  it('shares one GET between callers that ask while it is in flight', async () => {
    const get = vi
      .spyOn(api, 'get')
      .mockResolvedValue({ agent_file: 'odyn', workspace_dir: '/ws' } as never);
    const store = new AgentsMetaStore();

    await Promise.all([store.load(), store.load()]);

    expect(get).toHaveBeenCalledTimes(1);
  });

  it('refetches for a caller that asks after the first load settled', async () => {
    const get = vi
      .spyOn(api, 'get')
      .mockResolvedValue({ agent_file: 'odyn', workspace_dir: '/ws' } as never);
    const store = new AgentsMetaStore();

    await store.load();
    await store.load();

    expect(get).toHaveBeenCalledTimes(2);
  });
});
