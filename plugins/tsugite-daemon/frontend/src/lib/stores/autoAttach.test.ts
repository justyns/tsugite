// Node project: a fake `window.localStorage` lets storage.ts persist, and
// resetModules + re-import rebuilds the store against staged storage so the
// construction-time read (the persist-across-reload contract) is observable.
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

const KEY = 'tsugite_geo_autoattach';

function fakeLocalStorage() {
  const map = new Map<string, string>();
  return {
    getItem: (k: string) => (map.has(k) ? (map.get(k) as string) : null),
    setItem: (k: string, v: string) => void map.set(k, v),
    removeItem: (k: string) => void map.delete(k),
    clear: () => map.clear(),
  };
}

let ls: ReturnType<typeof fakeLocalStorage>;
beforeEach(() => {
  ls = fakeLocalStorage();
  vi.stubGlobal('window', { localStorage: ls });
  vi.resetModules();
});
afterEach(() => vi.unstubAllGlobals());

async function freshStore(key = KEY) {
  return (await import('./autoAttach.svelte')).autoAttachStore(key);
}

test('defaults off when nothing is stored', async () => {
  expect((await freshStore()).enabled).toBe(false);
});

test('reads a stored "true" back as on across a reload', async () => {
  ls.setItem(KEY, 'true');
  vi.resetModules();
  expect((await freshStore()).enabled).toBe(true);
});

test('any non-"true" stored value reads as off', async () => {
  ls.setItem(KEY, 'yes');
  vi.resetModules();
  expect((await freshStore()).enabled).toBe(false);
});

test('set() flips the state and persists it under the store key', async () => {
  const store = await freshStore();
  store.set(true);
  expect(store.enabled).toBe(true);
  expect(ls.getItem(KEY)).toBe('true');
  store.set(false);
  expect(store.enabled).toBe(false);
  expect(ls.getItem(KEY)).toBe('false');
});

test('the same key returns one shared instance; distinct keys are independent', async () => {
  const mod = await import('./autoAttach.svelte');
  expect(mod.autoAttachStore(KEY)).toBe(mod.autoAttachStore(KEY));
  const other = mod.autoAttachStore('tsugite_other_autoattach');
  other.set(true);
  expect(mod.autoAttachStore(KEY).enabled).toBe(false);
});
