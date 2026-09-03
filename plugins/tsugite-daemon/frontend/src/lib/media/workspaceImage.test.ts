import { afterEach, describe, expect, test, vi } from 'vitest';

vi.mock('$lib/api/client', () => ({ authHeaders: () => ({}) }));

import { loadWorkspaceDataURL } from './workspaceImage';

/** The fetch never succeeds: the assertion is the URL it was asked for, and a
 *  non-OK reply keeps the node project clear of a Blob decoder. */
function captureFetch(): string[] {
  const seen: string[] = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      seen.push(url);
      return { ok: false, status: 404 } as unknown as Response;
    }),
  );
  return seen;
}

afterEach(() => vi.unstubAllGlobals());

describe('loadWorkspaceDataURL', () => {
  test('a session id scopes the raw read to that session workspace', async () => {
    const seen = captureFetch();

    await expect(loadWorkspaceDataURL('img/logo.png', 'sess-1')).rejects.toThrow('workspace raw');

    expect(seen).toEqual(['/api/workspace/raw?path=img%2Flogo.png&session_id=sess-1']);
  });

  test('no session id leaves the raw read on the daemon workspace', async () => {
    const seen = captureFetch();

    await expect(loadWorkspaceDataURL('img/logo.png')).rejects.toThrow('workspace raw');

    expect(seen).toEqual(['/api/workspace/raw?path=img%2Flogo.png']);
  });
});
