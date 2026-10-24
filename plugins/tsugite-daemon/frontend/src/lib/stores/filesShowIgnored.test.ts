import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { fakeLocalStorage } from './testLocalStorage';

let ls: ReturnType<typeof fakeLocalStorage>;
beforeEach(() => {
  ls = fakeLocalStorage();
  vi.stubGlobal('window', { localStorage: ls });
  vi.resetModules();
});
afterEach(() => vi.unstubAllGlobals());

test('defaults off and persists a change', async () => {
  const { filesShowIgnored } = await import('./filesShowIgnored.svelte');
  expect(filesShowIgnored.enabled).toBe(false);

  filesShowIgnored.set(true);
  expect(ls.getItem('tsugite_files_show_ignored')).toBe('true');
});

test('reads a stored on state', async () => {
  ls.setItem('tsugite_files_show_ignored', 'true');
  const { filesShowIgnored } = await import('./filesShowIgnored.svelte');
  expect(filesShowIgnored.enabled).toBe(true);
});
