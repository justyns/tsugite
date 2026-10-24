import { beforeEach, describe, expect, test, vi } from 'vitest';

vi.mock('$lib/api/client', () => ({
  api: {
    get: vi.fn(async () => ({ path: 'a.md', content: 'x', is_text: true })),
    post: vi.fn(async () => ({ files: [] })),
    put: vi.fn(async () => ({})),
    getRaw: vi.fn(),
  },
}));

import { api } from '$lib/api/client';
import { FilesStore } from './files.svelte';

const apiGet = api.get as ReturnType<typeof vi.fn>;
const apiGetRaw = api.getRaw as ReturnType<typeof vi.fn>;

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

describe('FilesStore.download', () => {
  function stubBrowser() {
    const anchor = { href: '', download: '', click: vi.fn(), remove: vi.fn() };
    vi.stubGlobal('document', { createElement: () => anchor, body: { append: vi.fn() } });
    const revoke = vi.fn();
    vi.stubGlobal('URL', { createObjectURL: () => 'blob:x', revokeObjectURL: revoke });
    return Object.assign(anchor, { revoke });
  }

  beforeEach(() => {
    apiGetRaw.mockReset();
    vi.unstubAllGlobals();
  });

  test('fetches the attachment through the authenticated client and saves it under its own name', async () => {
    vi.useFakeTimers();
    const anchor = stubBrowser();
    apiGetRaw.mockResolvedValue({ blob: async () => new Blob(['x']) });

    await new FilesStore().download('out/report v2.pdf');

    expect(apiGetRaw).toHaveBeenCalledWith(
      '/api/workspace/raw?path=out%2Freport+v2.pdf&download=1',
    );
    expect(anchor.href).toBe('blob:x');
    expect(anchor.download).toBe('report v2.pdf');
    expect(anchor.click).toHaveBeenCalledOnce();
    expect(anchor.revoke).not.toHaveBeenCalled();
    vi.runAllTimers();
    expect(anchor.revoke).toHaveBeenCalledWith('blob:x');
    vi.useRealTimers();
  });

  test('a failed response throws the server error and saves nothing', async () => {
    const anchor = stubBrowser();
    apiGetRaw.mockRejectedValue(new Error('file not found'));

    await expect(new FilesStore().download('gone.bin')).rejects.toThrow('file not found');
    expect(anchor.click).not.toHaveBeenCalled();
  });
});
