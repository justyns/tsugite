import { beforeEach, describe, expect, test, vi } from 'vitest';

vi.mock('$lib/api/client', () => ({
  api: {
    get: vi.fn(async () => ({ path: 'a.md', content: 'x', is_text: true })),
    post: vi.fn(async () => ({ files: [] })),
    put: vi.fn(async () => ({})),
  },
  authHeaders: () => ({}),
}));

import { api } from '$lib/api/client';
import { FilesStore } from './files.svelte';

const apiGet = api.get as ReturnType<typeof vi.fn>;

describe('FilesStore.read', () => {
  beforeEach(() => apiGet.mockClear());

  test('a session id scopes the read to that session workspace', async () => {
    await new FilesStore().read('reports/cov.html', 'sess-1');

    expect(apiGet).toHaveBeenCalledWith(
      '/api/workspace/content?path=reports%2Fcov.html&session_id=sess-1',
    );
  });

  test('no session id leaves the read on the daemon workspace', async () => {
    await new FilesStore().read('reports/cov.html');

    expect(apiGet).toHaveBeenCalledWith('/api/workspace/content?path=reports%2Fcov.html');
  });
});
