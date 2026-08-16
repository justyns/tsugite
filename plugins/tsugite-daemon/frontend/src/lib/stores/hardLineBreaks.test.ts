// Node project: a fake `window.localStorage` lets storage.ts persist, and
// resetModules + re-import rebuilds the store against staged storage so the
// construction-time read (the persist-across-reload contract) is observable.
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { fakeLocalStorage, preloadStore } from './testLocalStorage';

const KEY = 'tsugite_hard_line_breaks';

preloadStore(() => import('./hardLineBreaks.svelte'));

let ls: ReturnType<typeof fakeLocalStorage>;
beforeEach(() => {
  ls = fakeLocalStorage();
  vi.stubGlobal('window', { localStorage: ls });
  vi.resetModules();
});
afterEach(() => vi.unstubAllGlobals());

async function freshStore() {
  return (await import('./hardLineBreaks.svelte')).hardLineBreaks;
}

test('defaults on when nothing is stored', async () => {
  expect((await freshStore()).enabled).toBe(true);
});

test('reads a stored "false" back as off across a reload', async () => {
  ls.setItem(KEY, 'false');
  vi.resetModules();
  expect((await freshStore()).enabled).toBe(false);
});

test('turning it off persists, so a reload stays off', async () => {
  (await freshStore()).set(false);
  expect(ls.getItem(KEY)).toBe('false');
  vi.resetModules();
  expect((await freshStore()).enabled).toBe(false);
});

test('turning it back on persists too', async () => {
  ls.setItem(KEY, 'false');
  vi.resetModules();
  const store = await freshStore();
  store.set(true);
  expect(ls.getItem(KEY)).toBe('true');
  vi.resetModules();
  expect((await freshStore()).enabled).toBe(true);
});
