// Node project: `storage.ts` no-ops without a `window`, so a fake localStorage
// has to be stubbed in before a pref is constructed.
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { fakeLocalStorage } from './testLocalStorage';
import { booleanPref } from './booleanPref.svelte';

let ls: ReturnType<typeof fakeLocalStorage>;
beforeEach(() => {
  ls = fakeLocalStorage();
  vi.stubGlobal('window', { localStorage: ls });
});
afterEach(() => vi.unstubAllGlobals());

test('an unset key reads as the fallback, either way round', () => {
  expect(booleanPref('k', true).enabled).toBe(true);
  expect(booleanPref('k', false).enabled).toBe(false);
});

test('a stored value wins over either fallback', () => {
  ls.setItem('k', 'false');
  expect(booleanPref('k', true).enabled).toBe(false);
  ls.setItem('k', 'true');
  expect(booleanPref('k', false).enabled).toBe(true);
});

test('a value that is neither "true" nor "false" reads as the fallback', () => {
  ls.setItem('k', 'sometimes');
  expect(booleanPref('k', true).enabled).toBe(true);
  expect(booleanPref('k', false).enabled).toBe(false);
});

test('set persists under the key, so a reload reads it back', () => {
  const pref = booleanPref('k', true);

  pref.set(false);
  expect(pref.enabled).toBe(false);
  expect(ls.getItem('k')).toBe('false');
  expect(booleanPref('k', true).enabled).toBe(false);

  pref.set(true);
  expect(ls.getItem('k')).toBe('true');
  expect(booleanPref('k', true).enabled).toBe(true);
});
