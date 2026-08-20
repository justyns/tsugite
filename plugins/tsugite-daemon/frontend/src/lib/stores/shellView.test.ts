// Node project: storage.ts no-ops without a `window`, so stub one, and
// resetModules + re-import rebuilds the store against staged storage - the
// constructor's read is what decides which view the shell opens on.
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { fakeLocalStorage, preloadStore } from './testLocalStorage';

const VIEW_KEY = 'tsugite_shell_view';

preloadStore(() => import('./shellView.svelte'));

let ls: ReturnType<typeof fakeLocalStorage>;
beforeEach(() => {
  ls = fakeLocalStorage();
  vi.stubGlobal('window', { localStorage: ls });
  vi.resetModules();
});
afterEach(() => vi.unstubAllGlobals());

async function freshShell() {
  return (await import('./shellView.svelte')).shellView;
}

test('a first run with nothing stored opens on chats', async () => {
  expect((await freshShell()).activeViewId).toBe('chats');
});

test('a returning visitor lands back on the view they left', async () => {
  ls.setItem(VIEW_KEY, 'schedules');
  expect((await freshShell()).activeViewId).toBe('schedules');
});

test('a stored workspace view restores the context rail with it', async () => {
  ls.setItem(VIEW_KEY, 'terminals');
  const shell = await freshShell();
  expect(shell.activeViewId).toBe('terminals');
  expect(shell.workspaceView).toBe('terminals');
});

test('activating a view persists it, so the next load lands there', async () => {
  (await freshShell()).activate('usage');
  expect(ls.getItem(VIEW_KEY)).toBe('usage');
  vi.resetModules();
  expect((await freshShell()).activeViewId).toBe('usage');
});
