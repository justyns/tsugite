import { readFileSync } from 'node:fs';
import { expect, it, vi } from 'vitest';

const origin = 'https://example.test';
const src = readFileSync('public/sw.js', 'utf8');

function click(data: unknown, clients: unknown[]) {
  const handlers: Record<string, (e: unknown) => void> = {};
  const openWindow = vi.fn();
  const self = {
    location: { origin },
    addEventListener: (type: string, fn: (e: unknown) => void) => (handlers[type] = fn),
    clients: { matchAll: async () => clients, openWindow },
  };
  new Function('self', src)(self);
  let done: Promise<unknown> = Promise.resolve();
  handlers.notificationclick!({
    notification: { close() {}, data },
    waitUntil: (p: Promise<unknown>) => (done = p),
  });
  return { done: () => done, openWindow };
}

it('navigates the open window to the notification url', async () => {
  const win = {
    url: `${origin}/#chats?sessionId=x`,
    focus: async () => win,
    navigate: vi.fn(),
  };
  const { done, openWindow } = click({ url: '#chats?sessionId=y' }, [win]);
  await done();
  expect(win.navigate).toHaveBeenCalledWith(`${origin}/#chats?sessionId=y`);
  expect(openWindow).not.toHaveBeenCalled();
});

it('opens a window when the open window cannot be navigated', async () => {
  const win = {
    url: `${origin}/#chats?sessionId=x`,
    focus: async () => win,
    navigate: () => Promise.reject(new TypeError('not controlled')),
  };
  const { done, openWindow } = click({ url: '#chats?sessionId=y' }, [win]);
  await done();
  expect(openWindow).toHaveBeenCalledWith(`${origin}/#chats?sessionId=y`);
});

it('opens a window at the absolute url when none is open', async () => {
  const { done, openWindow } = click({ url: '#chats?sessionId=y' }, []);
  await done();
  expect(openWindow).toHaveBeenCalledWith(`${origin}/#chats?sessionId=y`);
});
