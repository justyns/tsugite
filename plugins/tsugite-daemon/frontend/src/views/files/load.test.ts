import { beforeEach, describe, expect, test, vi } from 'vitest';

// loadWorkspace hits the workspace endpoint through the api client; mock it so
// the single-call contract is asserted without a network.
vi.mock('$lib/api/client', () => ({
  api: { get: vi.fn() },
}));

import { api } from '$lib/api/client';
import { loadWorkspace } from './load';

const apiGet = api.get as ReturnType<typeof vi.fn>;

const TREE = {
  entries: [
    { path: 'top.md', name: 'top.md', is_dir: false },
    { path: 'sub', name: 'sub', is_dir: true },
    { path: 'sub/nested.md', name: 'nested.md', is_dir: false },
  ],
  workspace_dir: '/ws',
};

describe('loadWorkspace', () => {
  beforeEach(() => {
    apiGet.mockReset();
    apiGet.mockResolvedValue(TREE);
  });

  test('fetches the whole tree in a single recursive call', async () => {
    const ws = await loadWorkspace('agent-x');

    expect(apiGet).toHaveBeenCalledTimes(1);
    const url = apiGet.mock.calls[0]?.[0] as string;
    expect(url).toContain('/api/agents/agent-x/workspace');
    expect(url).toContain('recursive=1');

    expect(ws.entries.map((e) => e.path)).toEqual(['top.md', 'sub', 'sub/nested.md']);
    expect(ws.workspaceDir).toBe('/ws');
  });

  test('builds tree and markdown index from the flat entries', async () => {
    const ws = await loadWorkspace('agent-x');

    expect(ws.tree.map((n) => n.path)).toEqual(['sub', 'top.md']);
    expect(ws.index.files).toEqual(expect.arrayContaining(['top.md', 'sub/nested.md']));
  });
});
