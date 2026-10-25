import { afterEach, expect, test, vi } from 'vitest';

vi.mock('$lib/api/client', () => ({
  api: { get: vi.fn() },
}));
vi.mock('$lib/stores/filesShowIgnored.svelte', () => ({
  filesShowIgnored: { enabled: true },
}));

import { api } from '$lib/api/client';
import { filesShowIgnored } from '$lib/stores/filesShowIgnored.svelte';
import { FilesWorkspaceStore } from './workspace.svelte';

const apiGet = api.get as ReturnType<typeof vi.fn>;

function serve(dirs: string[]) {
  const handler = async (url: string) => {
    const sub = dirs.find((d) => url.includes(`subdir=${d}`));
    return {
      entries: sub
        ? [{ path: `${sub}/lib.bin`, name: 'lib.bin', is_dir: false, ignored: true }]
        : url.includes('include_ignored=1')
          ? dirs.map((d) => ({ path: d, name: d, is_dir: true, ignored: true }))
          : [],
      workspace_dir: '/ws',
    };
  };
  apiGet.mockImplementation(handler);
  return handler;
}

function children(store: FilesWorkspaceStore, path: string) {
  return store.ws?.tree.find((n) => n.path === path)?.children.map((n) => n.path);
}

afterEach(() => {
  filesShowIgnored.enabled = true;
});

test('reload lists the ignored directories that were opened before', async () => {
  serve(['vendor']);
  const store = new FilesWorkspaceStore();
  await store.reload();
  await store.loadIgnoredDir('vendor');

  await store.reload();

  expect(children(store, 'vendor')).toEqual(['vendor/lib.bin']);
});

test('opening two ignored directories at once lists both', async () => {
  serve(['a', 'b']);
  const store = new FilesWorkspaceStore();
  await store.reload();

  await Promise.all([store.loadIgnoredDir('a'), store.loadIgnoredDir('b')]);

  expect(children(store, 'a')).toEqual(['a/lib.bin']);
  expect(children(store, 'b')).toEqual(['b/lib.bin']);
});

test('opened ignored directories are listed again after hiding and showing ignored files', async () => {
  serve(['vendor']);
  const store = new FilesWorkspaceStore();
  await store.reload();
  await store.loadIgnoredDir('vendor');

  filesShowIgnored.enabled = false;
  await store.reload();
  filesShowIgnored.enabled = true;
  await store.reload();

  expect(children(store, 'vendor')).toEqual(['vendor/lib.bin']);
});

test('a listing that lands after a reload does not duplicate entries', async () => {
  const handler = serve(['vendor']);
  const store = new FilesWorkspaceStore();
  await store.reload();
  let release!: () => void;
  const gate = new Promise<void>((r) => (release = r));
  apiGet.mockImplementationOnce(async (url: string) => {
    await gate;
    return handler(url);
  });
  const stale = store.loadIgnoredDir('vendor');

  await store.reload();
  release();
  await stale;

  expect(children(store, 'vendor')).toEqual(['vendor/lib.bin']);
});
