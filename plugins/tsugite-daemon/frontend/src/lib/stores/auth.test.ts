// The store reads its token at import, so each test stubs `window` and then imports a fresh module.
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { fakeLocalStorage } from './testLocalStorage';

let ls: ReturnType<typeof fakeLocalStorage>;
beforeEach(() => {
  ls = fakeLocalStorage();
  vi.resetModules();
});
afterEach(() => vi.unstubAllGlobals());

async function load(win: Partial<Window>) {
  vi.stubGlobal('window', { localStorage: ls, ...win });
  return (await import('./auth.svelte')).auth;
}

test('a desktop bridge token is used ahead of localStorage and never written there', async () => {
  ls.setItem('tsugite_token', 'stored');
  const auth = await load({ tsugiteDesktop: { token: 'from-desktop' } });
  expect(auth.token).toBe('from-desktop');
  expect(auth.gated).toBe(false);
  auth.save('typed');
  expect(ls.getItem('tsugite_token')).toBe('stored');
});

test('without the bridge the token comes from and is saved to localStorage', async () => {
  ls.setItem('tsugite_token', 'stored');
  const auth = await load({});
  expect(auth.token).toBe('stored');
  auth.save('typed');
  expect(ls.getItem('tsugite_token')).toBe('typed');
});
